"""Five-outcome ensemble with a lipid composition gate after the first block."""

import torch
from tabm import TabM
from torch import nn

RANK = 8
BOUND = 0.25
BRANCH_SEED = 20261122


class CompositionAdjustment(nn.Module):
    def __init__(self):
        super().__init__()
        # A fresh branch cannot perturb the stock head or the dropout generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(BRANCH_SEED)
            self.context = nn.Linear(42, RANK, bias=False)
            self.projection = nn.Linear(RANK, 512, bias=False)
            self.members = nn.Parameter(1.0 + 0.04 * torch.randn(32, RANK))
            nn.init.zeros_(self.projection.weight)

    def forward(self, context):
        factors = self.context(context).tanh()[:, None, :] * self.members
        # BF16 cannot represent 1 + many useful near-zero adjustments. Keep
        # the residual and its application in FP32 in both experimental arms.
        return BOUND * self.projection(factors).float().tanh()


class CompositionReadout(nn.Module):
    def __init__(self, n_features, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 31)
            stock = TabM.make(
                n_num_features=n_features,
                cat_cardinalities=[],
                d_out=5,
                num_embeddings=None,
                n_blocks=2,
                d_block=512,
                k=32,
                dropout=0.1,
            )
        self.k = stock.k
        self.ensemble_view = stock.ensemble_view
        self.first, self.second = stock.backbone.blocks
        self.output = stock.output
        self.adjustment = CompositionAdjustment()

    def forward(self, features, context):
        hidden = self.first(self.ensemble_view(features))
        return self.output(self.second(hidden * (1.0 + self.adjustment(context))))


class InflammationModel(nn.Module):
    def __init__(self, encoder, center, scale, seed):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("z_mean", torch.as_tensor(center, dtype=torch.float32).clone())
        self.register_buffer("z_std", torch.as_tensor(scale, dtype=torch.float32).clone())
        self.head = CompositionReadout(self.z_mean.numel() + 7, seed)

    def forward(self, values, covariates, contrasts):
        z = (self.encoder(values) - self.z_mean) / self.z_std
        return self.head(torch.cat((z, covariates), dim=-1), contrasts)
