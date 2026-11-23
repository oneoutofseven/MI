"""Generate a small, entirely synthetic cohort for the command-line example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mi_nmr.shared.data import TASKS, resource


def generate(path):
    rng = np.random.default_rng(27)
    n = 96
    layout = resource("schema.json")
    sizes = np.asarray(layout["family_indices"])
    g = rng.normal(size=(n, 3))
    x = rng.lognormal(mean=0, sigma=0.35, size=(n, len(layout["columns"])))
    for family in sizes:
        total = np.exp(0.3 * g[:, 0] + rng.normal(scale=0.2, size=n))
        x[:, family[0]] = total * rng.lognormal(-3.0, 0.2, size=n)
        x[:, family[1:]] = total[:, None] * rng.dirichlet([3.0, 4.0, 1.0, 2.0], size=n)
    x[rng.random(x.shape) < 0.025] = np.nan
    f = pd.DataFrame(x, columns=layout["columns"])
    f.insert(0, "id", [f"sim-{i:04d}" for i in range(n)])
    f["pretrain_split"] = ["train"] * 64 + ["val"] * 16 + ["unused"] * 16
    f["finetune_split"] = ["train"] * 56 + ["unused"] * 8 + ["val"] * 16 + ["test"] * 16
    f["age"] = rng.integers(40, 70, n)
    f["male"] = rng.integers(0, 2, n)
    f["bmi"] = rng.normal(27.0, 3.0, n)
    f.loc[::11, "bmi"] = np.nan
    f["smoking"] = rng.choice(("never", "previous", "current", "unknown"), n)
    for j, task in enumerate(TASKS):
        f[task] = np.exp(0.2 * g[:, j % 3] + rng.normal(scale=0.3, size=n))
    # Unlabelled participants can contribute to NMR pretraining.
    f.loc[f.finetune_split == "unused", list(TASKS)] = np.nan
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    generate(p.parse_args().output)
