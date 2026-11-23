"""Single-device pretraining and joint five-phenotype fine-tuning."""

import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn.utils import clip_grad_norm_

from .checkpoints import (
    digest,
    encoder_record,
    load,
    make_encoder,
    numpy_dict,
    output_run,
    save,
    save_json,
    tensor_dict,
)
from .composition import train_statistics
from .data import (
    TASKS,
    clinical_values,
    cohort,
    covariate_z,
    data_digest,
    fit_clinical_scales,
    measurement_batch,
    partition_key,
    readout_inputs,
    resource,
    rows_for,
    transformed_targets,
)
from .masks import training_bank, validation_plans
from .model import FieldSupervisedModel, MeasurementModel
from .objectives import FieldObjective, weighted_loss
from .readout import InflammationModel


def batches(rows, size):
    if size < 1:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def schedule(step, settings, per_epoch):
    warmup = max(1, round(settings["warmup_epochs"] * per_epoch))
    peak, floor = settings["learning_rate"], settings["minimum_learning_rate"]
    if step < warmup:
        return peak * (step + 1) / warmup
    duration = max(settings["max_epochs"] * per_epoch - warmup, 1)
    fraction = min((step - warmup) / duration, 1.0)
    return floor + (peak - floor) * (1.0 + math.cos(math.pi * fraction)) / 2


def autocast(device, enabled):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda"
    )


def fit_epochs(
    out, resume, identity, settings, model, optimizer, train_epoch, validate, best_payload, device
):
    """Persist the full optimizer, generator and history at epoch boundaries."""
    with output_run(out, resume) as folder:
        last = folder / "last.pt"
        history = []
        best_at, best = 0, math.inf

        def checkpoint():
            save(
                last,
                {
                    "identity": identity,
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "best_at": best_at,
                    "best": best,
                    "rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                },
            )

        if resume:
            state = load(last)
            if state["identity"] != identity or (state["gpu_rng"] is None) != (
                device.type == "cpu"
            ):
                raise ValueError(
                    "Resume requires the original recipe, data, source and device type"
                )
            model.load_state_dict(state["state"])
            optimizer.load_state_dict(state["optimizer"])
            history, best_at, best = state["history"], state["best_at"], state["best"]
            if best_at and not (folder / "best.pt").is_file():
                raise ValueError("Missing best.pt for this run")
            torch.set_rng_state(state["rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["gpu_rng"], device)
            del state
        else:
            checkpoint()
        for epoch in range(len(history) + 1, settings["max_epochs"] + 1):
            if (
                best_at
                and len(history) >= settings.get("minimum_epochs", 1)
                and len(history) - best_at >= settings["patience"]
            ):
                break
            model.train()
            train_epoch(epoch)
            model.eval()
            with torch.inference_mode():
                score, metrics = validate()
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite validation criterion")
            if score < best - settings.get("minimum_improvement", 0.0):
                best, best_at = score, epoch
                save(folder / "best.pt", {**best_payload(), "epoch": epoch, "score": score})
            history.append({"epoch": epoch, "score": score, **metrics})
            checkpoint()
            save_json(folder / "history.json", history)
            print(
                f"epoch {epoch}  validation {score:.6f}  best {best:.6f} (epoch {best_at})",
                flush=True,
            )
        save_json(folder / "history.json", history)
        return history


def pretrain(path, out, config, device, resume=False):
    frame, raw = cohort(path, roles=True)
    layout = resource("schema.json")
    tr, va = [rows_for(frame, "pretrain_split", r) for r in ("train", "val")]
    stats = train_statistics(raw, tr, layout)
    bank = training_bank(layout, config["masking"], config["mask"])
    masks, views = validation_plans(layout, config["masking"], len(va))
    torch.manual_seed(config["seed"])
    core = MeasurementModel(
        layout, stats, config["encoder"], config["decoder"], config["decoder_mode"]
    )
    model = FieldSupervisedModel(core).to(device)
    objective = FieldObjective(layout, stats, config["field_weight"]).to(device)
    s = config["pretraining"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=s["learning_rate"],
        betas=tuple(s["betas"]),
        eps=s["eps"],
        weight_decay=s["weight_decay"],
    )
    logical = s["effective_batch_size"]
    steps = math.ceil(len(tr) / logical)

    def train_epoch(epoch):
        order = np.random.default_rng(config["seed"] + epoch).permutation(tr)
        mask_rng = np.random.default_rng(config["masking"]["seed"] + epoch)
        for step, rows in enumerate(batches(order, logical)):
            selected = bank[mask_rng.integers(len(bank), size=len(rows))]
            batch = measurement_batch(raw, rows, stats, device, selected)
            n_people, n_compositions = map(int, objective.denominators(batch))
            if n_people == 0:
                continue
            for group in optimizer.param_groups:
                group["lr"] = schedule((epoch - 1) * steps + step, s, steps)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(rows), s["micro_batch_size"]):
                micro = {k: v[start : start + s["micro_batch_size"]] for k, v in batch.items()}
                with autocast(device, s["amp_bfloat16"]):
                    pred = model(micro["values"], micro["observed"], micro["visible"], check=False)
                    loss = weighted_loss(
                        objective(pred, micro),
                        n_people,
                        n_compositions,
                        config["composition_weight"],
                    )
                loss.backward()
            clip_grad_norm_(model.parameters(), s["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()

    def validate():
        view_metrics = {}
        for j, name in enumerate(views):
            sums = np.zeros(5, dtype=np.float64)
            for ix in batches(np.arange(len(va)), s["validation_micro_batch_size"]):
                b = measurement_batch(raw, va[ix], stats, device, masks[j, ix])
                with autocast(device, s["amp_bfloat16"]):
                    pred = model(b["values"], b["observed"], b["visible"], check=False)
                    part = objective(pred, b)
                sums += [
                    float(part[k])
                    for k in (
                        "global_numeric_sum",
                        "local_sum",
                        "numeric_people",
                        "composition_sum",
                        "composition_people",
                    )
                ]
            global_sum, field_sum, people, comp_sum, comp_people = map(float, sums)
            if not people:
                raise ValueError(f"No observed hidden fields for {name}")
            view_metrics[name] = {
                "global_mae": global_sum / people,
                "field_mae": field_sum / people,
                "composition_mae": comp_sum / comp_people if comp_people else None,
            }
        score = np.mean(
            [v["global_mae"] for k, v in view_metrics.items() if k.startswith("random_")]
        )
        return float(score), {"views": view_metrics}

    key = partition_key(frame)

    def selected():
        return {
            "format": "mi.pretrain.v1",
            "layout": layout,
            "encoder": config["encoder"],
            "decoder": config["decoder"],
            "decoder_mode": config["decoder_mode"],
            "statistics": tensor_dict(stats),
            "state": model.state_dict(),
            "partition_key": key,
        }

    identity = {
        "stage": "pretrain",
        "recipe": config,
        "data": data_digest((raw,), np.concatenate((tr, va))),
        "partition_key": key,
    }
    return fit_epochs(
        out, resume, identity, s, model, optimizer, train_epoch, validate, selected, device
    )


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
