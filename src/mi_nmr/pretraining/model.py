"""Participant and field decoders for masked NMR reconstruction."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mi_nmr.shared.encoder import ParticipantEncoder


class NumericDecoder(nn.Module):
    def __init__(self, layout, statistics, embedding_width, settings, mode):
        super().__init__()
        if mode not in ["measurement", "ordinary"]:
            raise ValueError("Expected measurement or ordinary decoder")
        self.mode, self.n_fields = mode, len(layout["columns"])
        self.families = len(layout["family_indices"])
        queries = self.families + len(layout["other_indices"])
        self.query = nn.Embedding(queries, settings["query_width"])
        self.z_projection = nn.Linear(embedding_width, settings["hidden"])
        self.film = nn.Linear(settings["query_width"], 2 * settings["hidden"])
        self.hidden = nn.Linear(settings["hidden"], settings["hidden"])
        self.output = nn.Linear(settings["hidden"], 6)
        self.dropout = nn.Dropout(settings["dropout"])
        self.register_buffer(
            "family_indices", torch.tensor(layout["family_indices"], dtype=torch.long)
        )
        self.register_buffer(
            "other_indices", torch.tensor(layout["other_indices"], dtype=torch.long)
        )
        for name in ["mean", "std", "particle_scale", "lipid_scale", "composition_prior"]:
            self.register_buffer(
                name, torch.as_tensor(np.asarray(statistics[name]), dtype=torch.float32)
            )

    def forward(self, z):
        gamma, beta = self.film(self.query.weight).chunk(2, -1)
        h = F.silu(self.z_projection(z)[:, None] * (1 + gamma) + beta)
        out = self.output(self.dropout(F.silu(self.hidden(h)))).float()
        block = out[:, : self.families]
        indices = self.family_indices
        if self.mode == "measurement":
            offset = math.log(math.expm1(1.0))
            particle = F.softplus(block[..., 0] + offset) * self.particle_scale
            total = F.softplus(block[..., 1] + offset) * self.lipid_scale
            fractions = F.softmax(
                block[..., 2:] + self.composition_prior.clamp_min(1e-12).log(), -1
            )
            components = total[..., None] * fractions
            raw_groups = torch.cat([particle[..., None], components], -1)
            standardized_groups = (raw_groups - self.mean[indices]) / self.std[indices]
        else:
            # Unconstrained point regression, with a shared subclass offset.
            # All six logits participate; decoder parameter count is matched.
            standardized_groups = torch.cat([block[..., :1], block[..., 1:2] + block[..., 2:]], -1)
            raw_groups = standardized_groups * self.std[indices] + self.mean[indices]
            components = raw_groups[..., 1:].clamp_min(0.0)
            total = components.sum(-1)
            fractions = torch.where(
                total[..., None] > 0,
                components / total[..., None].clamp_min(1e-30),
                torch.full_like(components, 0.25),
            )
        prediction = z.new_zeros(len(z), self.n_fields, dtype=torch.float32)
        prediction = prediction.index_copy(1, indices.flatten(), standardized_groups.flatten(1))
        prediction = prediction.index_copy(1, self.other_indices, out[:, self.families :, 0])
        return {
            "prediction": prediction,
            "fractions": fractions,
            "lipid_sum": total,
            "raw_groups": raw_groups,
        }


class MeasurementModel(nn.Module):
    def __init__(self, layout, statistics, encoder, decoder, mode="measurement"):
        super().__init__()
        self.encoder = ParticipantEncoder(layout, encoder)
        self.decoder = NumericDecoder(layout, statistics, encoder["embedding_width"], decoder, mode)
        self.apply(self._initialize)
        nn.init.uniform_(self.encoder.tokenizer.value_w1.weight, -1.0, 1.0)
        nn.init.uniform_(self.encoder.tokenizer.value_w2.weight, -0.1, 0.1)
        nn.init.normal_(self.encoder.tokenizer.value_alpha, 0.0, 1.0)
        scale = math.sqrt(2 * encoder["layers"])
        for block in self.encoder.blocks:
            nn.init.normal_(block.attention.output.weight, 0.0, 0.02 / scale)
            nn.init.normal_(block.down.weight, 0.0, 0.02 / scale)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, values, observed=None, visible=None, field_ids=None, *, check=True):
        z = self.encoder(values, observed, visible, field_ids, check=check)
        return {"embedding": z, **self.decoder(z)}

    def encode(self, values, observed=None, *, field_ids=None):
        return self.encoder(values, observed, field_ids=field_ids)

    def parameter_counts(self):
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


class FieldSupervisedModel(nn.Module):
    """One encoder pass supplies both the participant and the hidden-field predictions."""

    def __init__(self, core):
        super().__init__()
        self.core = core
        with torch.random.fork_rng(devices=[]):
            self.field_head = nn.Linear(core.encoder.settings["width"], 1)
            nn.init.zeros_(self.field_head.weight)
            nn.init.zeros_(self.field_head.bias)

    def forward(self, values, observed, visible, field_ids=None, *, check=True):
        z, fields = self.core.encoder(
            values, observed, visible, field_ids, check=check, return_fields=True
        )
        return {
            "embedding": z,
            "local": self.field_head(fields).squeeze(-1),
            **self.core.decoder(z),
        }
