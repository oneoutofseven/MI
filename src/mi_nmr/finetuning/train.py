"""Fit and evaluate the five-phenotype readout and NMR encoder jointly."""

import math
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn.utils import clip_grad_norm_

from mi_nmr.shared.checkpoints import digest, encoder_record, make_encoder, numpy_dict, tensor_dict
from mi_nmr.shared.data import (
    TASKS,
    clinical_values,
    cohort,
    covariate_z,
    data_digest,
    fit_clinical_scales,
    measurement_batch,
    partition_key,
    readout_inputs,
    rows_for,
    transformed_targets,
)
from mi_nmr.shared.engine import autocast, batches, fit_epochs
from mi_nmr.finetuning.readout import InflammationModel


@torch.inference_mode()
def embedding_moments(encoder, raw, training, stats, device, batch_size):
    encoder.eval()
    width = encoder.settings["embedding_width"]
    total, square = np.zeros(width), np.zeros(width)
    for rows in batches(training, batch_size):
        b = measurement_batch(raw, rows, stats, device)
        z = encoder(b["values"], b["observed"], check=False).double().cpu().numpy()
        total += z.sum(0)
        square += np.square(z).sum(0)
    center = total / len(training)
    return center, np.sqrt(np.maximum(square / len(training) - center**2, 1e-12))


@torch.inference_mode()
def predict_rows(model, raw, cov, rows, layout, stats, device, batch_size):
    model.eval()
    out = []
    for ix in batches(rows, batch_size):
        args = readout_inputs(raw, cov, ix, layout, stats, device)
        out.append(model(*args).float().mean(1).cpu().numpy())
    return np.concatenate(out)


def scores(pred, transformed, scales):
    if pred.shape != transformed.shape or not np.isfinite(pred).all():
        raise ValueError("Expected finite, aligned predictions")
    mean, sd = scales["mean"], scales["sd"]
    per_task, standardized_mse = {}, []
    for j, task in enumerate(TASKS):
        valid = np.isfinite(transformed[:, j])
        y = transformed[valid, j]
        if len(y) < 2:
            raise ValueError(f"Need at least two observed {task} values for evaluation")
        p = pred[valid, j].astype(np.float64) * sd[j] + mean[j]
        mse = float(np.mean(np.square((p - y) / sd[j])))
        rho = float(spearmanr(p, y).statistic) if np.ptp(p) and np.ptp(y) else None
        per_task[task] = {
            "n": len(y),
            "rmse_transformed": float(math.sqrt(mse) * sd[j]),
            "rmse_standardized": math.sqrt(mse),
            "spearman": rho,
        }
        standardized_mse.append(mse)
    return {
        "five_task_mse": float(np.mean(standardized_mse)),
        "macro_standardized_rmse": float(np.mean(np.sqrt(standardized_mse))),
        "per_task": per_task,
    }


def finetune(path, encoder_path, out, config, device, resume=False):
    frame, raw = cohort(path, roles=True, clinical=True, targets=True)
    key = partition_key(frame)
    pretrained = encoder_record(encoder_path)
    if "partition_key" in pretrained and pretrained["partition_key"] != key:
        raise ValueError("Use the same participant/split manifest for both stages")
    keep = frame.finetune_split.isin(("train", "val")).to_numpy()
    frame, raw = frame.loc[keep].reset_index(drop=True), raw[keep]
    tr, va = [rows_for(frame, "finetune_split", r) for r in ("train", "val")]
    y, cov = transformed_targets(frame), clinical_values(frame)
    if not np.isfinite(y[tr]).any(1).all():
        raise ValueError(
            "Each supervised training row needs at least one observed outcome; mark unlabelled rows unused"
        )
    scales = fit_clinical_scales(cov, y, tr)
    cov = covariate_z(cov, scales)
    y_z = (y - scales["mean"]) / scales["sd"]
    layout, stats = pretrained["layout"], numpy_dict(pretrained["statistics"])
    s = config["finetuning"]
    encoder = make_encoder(pretrained).to(device)
    center, scale = embedding_moments(encoder, raw, tr, stats, device, s["micro_batch_size"])
    model = InflammationModel(encoder, center, scale, s["seed"]).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": s["encoder_lr"],
                "weight_decay": s["encoder_weight_decay"],
            },
            {
                "params": model.head.parameters(),
                "lr": s["head_lr"],
                "weight_decay": s["head_weight_decay"],
            },
        ]
    )

    def train_epoch(epoch):
        seed = s["seed"] + epoch
        torch.manual_seed(seed)
        order = np.random.default_rng(seed).permutation(tr)
        for rows in batches(order, s["effective_batch_size"]):
            counts = np.isfinite(y_z[rows]).sum(0)
            coeff = 1.0 / (max(int((counts > 0).sum()), 1) * model.head.k * np.maximum(counts, 1))
            weights = torch.as_tensor(coeff, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            for ix in batches(rows, s["micro_batch_size"]):
                target = torch.as_tensor(y_z[ix], dtype=torch.float32, device=device)
                valid = torch.isfinite(target)
                with autocast(device, s["amp_bfloat16"]):
                    members = model(*readout_inputs(raw, cov, ix, layout, stats, device))
                    delta = members.float() - torch.where(valid, target, 0.0)[:, None, :]
                    loss = (torch.where(valid[:, None, :], delta, 0.0).square() * weights).sum()
                loss.backward()
            clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

    def validate():
        pred = predict_rows(model, raw, cov, va, layout, stats, device, s["micro_batch_size"])
        report = scores(pred, y[va], scales)
        return report["five_task_mse"], report

    def selected():
        return {
            "format": "mi.readout.v1",
            "layout": layout,
            "encoder": pretrained["encoder"],
            "statistics": pretrained["statistics"],
            "state": model.state_dict(),
            "tasks": list(TASKS),
            "scales": tensor_dict(scales),
            "partition_key": key,
        }

    identity = {
        "stage": "finetune",
        "settings": s,
        "encoder_sha256": digest(encoder_path),
        "partition_key": key,
        "data": data_digest((raw, cov, y), np.arange(len(raw))),
    }
    return fit_epochs(
        out, resume, identity, s, model, optimizer, train_epoch, validate, selected, device
    )


if __name__ == "__main__":
    from mi_nmr.shared.cli import main

    main(["finetune", *sys.argv[1:]])
