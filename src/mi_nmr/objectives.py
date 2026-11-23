"""Observed hidden-value MAE and complete-hidden composition contrasts."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MaskedObjective(nn.Module):
    def __init__(self, layout, statistics):
        super().__init__()
        self.register_buffer("families", torch.tensor(layout["family_indices"], dtype=torch.long))
        self.register_buffer(
            "helmert", torch.as_tensor(np.asarray(statistics["helmert"]), dtype=torch.float32)
        )
        self.register_buffer(
            "composition_std",
            torch.as_tensor(np.asarray(statistics["composition_std"]), dtype=torch.float32),
        )

    def denominators(self, batch):
        hidden = batch["observed"] & ~batch["visible"]
        raw = batch["raw"][:, self.families[:, 1:]]
        observed = batch["observed"][:, self.families[:, 1:]].all(-1)
        hidden_components = hidden[:, self.families[:, 1:]].all(-1)
        clean = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw))
        valid = observed & hidden_components & (clean >= 0).all(-1) & (clean.sum(-1) > 0)
        return hidden.any(1).sum(), valid.any(1).sum()

    def forward(self, output, batch):
        hidden = batch["observed"] & ~batch["visible"]
        target = torch.where(hidden, batch["values"], torch.zeros_like(batch["values"]))
        error = torch.where(hidden, (output["prediction"].float() - target).abs(), 0.0)
        count = hidden.sum(1)
        per_person = error.sum(1) / count.clamp_min(1)
        lipid_indices = self.families[:, 1:]
        raw = batch["raw"][:, lipid_indices]
        hidden_group = hidden[:, lipid_indices].all(-1)
        observed_group = batch["observed"][:, lipid_indices].all(-1)
        clean = torch.where(torch.isfinite(raw), raw, torch.zeros_like(raw))
        total = clean.sum(-1)
        eligible = hidden_group & observed_group & (clean >= 0).all(-1) & (total > 0)
        safe = torch.where(eligible[..., None], clean, torch.zeros_like(clean))
        fractions = safe / torch.where(eligible, total, torch.ones_like(total))[..., None]
        contrast_error = (
            (output["fractions"].float() - fractions) @ self.helmert.T
        ).abs() / self.composition_std
        group_error = torch.where(eligible, contrast_error.mean(-1), 0.0)
        group_count = eligible.sum(1)
        comp_person = group_error.sum(1) / group_count.clamp_min(1)
        squared = torch.where(hidden, (output["prediction"].float() - target).square(), 0.0)
        return {
            "numeric_sum": per_person.sum(),
            "numeric_people": count.gt(0).sum(),
            "token_absolute_sum": error.sum(),
            "token_squared_sum": squared.sum(),
            "targets": count.sum(),
            "composition_sum": comp_person.sum(),
            "composition_people": group_count.gt(0).sum(),
            "composition_groups": group_count.sum(),
        }


def weighted_loss(parts, numeric_denominator, composition_denominator, composition_weight):
    numeric = parts["numeric_sum"] / max(int(numeric_denominator), 1)
    composition = parts["composition_sum"] / max(int(composition_denominator), 1)
    return numeric + composition_weight * composition


class FieldObjective(MaskedObjective):
    def __init__(self, layout, statistics, weight):
        super().__init__(layout, statistics)
        self.weight = weight

    def forward(self, output, batch):
        parts = super().forward(output, batch)
        hidden = batch["observed"] & ~batch["visible"]
        target = torch.where(hidden, batch["values"], 0.0)
        err = torch.where(hidden, (output["local"].float() - target).abs(), 0.0)
        local = (err.sum(1) / hidden.sum(1).clamp_min(1)).sum()
        parts["global_numeric_sum"] = parts["numeric_sum"]
        parts["local_sum"] = local
        # The trainer divides numeric_sum by the number of valid participants
        # in the whole logical batch, including when microbatch sizes differ.
        parts["numeric_sum"] = parts["numeric_sum"] + self.weight * local
        return parts
