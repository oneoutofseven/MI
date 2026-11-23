"""Train the participant encoder by reconstructing hidden NMR measurements."""

import math
import sys

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from mi_nmr.shared.checkpoints import tensor_dict
from mi_nmr.shared.composition import train_statistics
from mi_nmr.shared.data import cohort, data_digest, measurement_batch, partition_key, resource, rows_for
from mi_nmr.shared.engine import autocast, batches, fit_epochs
from mi_nmr.pretraining.masks import training_bank, validation_plans
from mi_nmr.pretraining.model import FieldSupervisedModel, MeasurementModel
from mi_nmr.pretraining.objectives import FieldObjective, weighted_loss


def schedule(step, settings, per_epoch):
    warmup = max(1, round(settings["warmup_epochs"] * per_epoch))
    peak, floor = settings["learning_rate"], settings["minimum_learning_rate"]
    if step < warmup:
        return peak * (step + 1) / warmup
    duration = max(settings["max_epochs"] * per_epoch - warmup, 1)
    fraction = min((step - warmup) / duration, 1.0)
    return floor + (peak - floor) * (1.0 + math.cos(math.pi * fraction)) / 2


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


if __name__ == "__main__":
    from mi_nmr.shared.cli import main

    main(["pretrain", *sys.argv[1:]])
