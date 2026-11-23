"""Field-token encoder and measurement-constrained decoder for plasma NMR."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FieldTokenizer(nn.Module):
    def __init__(self, layout, width, bases=100):
        super().__init__()
        self.bases = bases
        self.identity = nn.Embedding(len(layout["columns"]), width)
        self.metadata = nn.ModuleDict(
            {
                name + "_embedding": nn.Embedding(len(vocab), width)
                for name, vocab in layout["vocabularies"].items()
            }
        )
        for name, values in layout["metadata_ids"].items():
            self.register_buffer(name + "_ids", torch.tensor(values, dtype=torch.long))
        self.value_w1 = nn.Linear(1, bases, bias=False)
        self.value_w2 = nn.Linear(bases, bases, bias=False)
        self.value_alpha = nn.Parameter(torch.empty(bases))
        self.value_lookup = nn.Linear(bases, width, bias=False)
        # Row0=CLS,row1=true missing,row2=artificially hidden.
        self.special = nn.Embedding(3, width)

    def forward(self, values, observed, visible, field_ids):
        clean = torch.where(visible, values, torch.zeros_like(values))
        v1 = F.leaky_relu(self.value_w1(clean[..., None]))
        weights = F.softmax(self.value_w2(v1) + self.value_alpha * v1, dim=-1)
        numeric = self.value_lookup(weights) * math.sqrt(self.bases)
        state = torch.where(observed, 2, 1)
        numeric = torch.where(visible[..., None], numeric, self.special(state))
        fields = 0.5 * (numeric + self.identity(field_ids)[None])
        metadata = sum(
            layer(getattr(self, name.removesuffix("_embedding") + "_ids")[field_ids])[None]
            for name, layer in self.metadata.items()
        )
        fields = fields + 0.1 * metadata / math.sqrt(len(self.metadata))
        return torch.cat([self.special.weight[0][None, None].expand(len(values), 1, -1), fields], 1)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by its number of heads")
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, states, allowed_keys):
        batch, length, width = states.shape
        q, k, v = (
            self.qkv(states)
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed_keys[:, None, None, :], dropout_p=0.0, is_causal=False
        )
        return self.output(value.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads, hidden, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, allowed_keys):
        states = states + self.dropout(self.attention(self.attention_norm(states), allowed_keys))
        gate, value = self.gate_up(self.ffn_norm(states)).chunk(2, -1)
        return states + self.dropout(self.down(F.silu(gate) * value))


class ParticipantEncoder(nn.Module):
    def __init__(self, layout, settings):
        super().__init__()
        self.n_fields = len(layout["columns"])
        self.settings = dict(settings)
        width = settings["width"]
        self.tokenizer = FieldTokenizer(layout, width, settings["soft_bases"])
        self.blocks = nn.ModuleList(
            [
                Block(width, settings["heads"], settings["ffn_hidden"], settings["dropout"])
                for _ in range(settings["layers"])
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, settings["embedding_width"]),
            nn.LayerNorm(settings["embedding_width"]),
        )

    def forward(
        self,
        values,
        observed=None,
        visible=None,
        field_ids=None,
        *,
        check=True,
        return_fields=False,
    ):
        observed = torch.isfinite(values) if observed is None else observed
        visible = observed if visible is None else visible
        if field_ids is None:
            field_ids = torch.arange(values.shape[1], device=values.device)
        if check:
            if values.ndim != 2 or observed.shape != values.shape or visible.shape != values.shape:
                raise ValueError("Expected aligned participant-by-field inputs")
            if observed.dtype != torch.bool or visible.dtype != torch.bool:
                raise ValueError("Observation masks must be Boolean")
            if (
                field_ids.ndim != 1
                or len(field_ids) != values.shape[1]
                or len(field_ids.unique()) != len(field_ids)
            ):
                raise ValueError("Field identities must be unique and follow values when permuted")
            if (field_ids < 0).any() or (field_ids >= self.n_fields).any():
                raise ValueError("Unknown field identity")
            if (visible & ~observed).any() or not torch.isfinite(values[visible]).all():
                raise ValueError("Visible values must have actually been observed and be finite")
        states = self.tokenizer(values, observed, visible, field_ids)
        allowed = torch.cat(
            [torch.ones(len(values), 1, dtype=torch.bool, device=values.device), visible], 1
        )
        for block in self.blocks:
            if (
                self.settings.get("activation_checkpointing", False)
                and self.training
                and torch.is_grad_enabled()
            ):
                states = checkpoint(
                    block, states, allowed, use_reentrant=False, preserve_rng_state=True
                )
            else:
                states = block(states, allowed)
        participant = self.readout(states[:, 0])
        if return_fields:
            return participant, states[:, 1:]
        return participant


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
