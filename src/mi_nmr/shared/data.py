"""Canonical columns, participant partitions and train-fitted transformations."""

import csv
import hashlib
import json
from importlib.resources import files

import numpy as np
import pandas as pd
import torch

from .composition import composition_coordinates

TASKS = ("CRP", "neutrophils", "monocytes", "lymphocytes", "eosinophils")
COVARIATES = ("age", "male", "bmi", "smoking")
ROLES = ("pretrain_split", "finetune_split")


def resource(name):
    return json.loads(files(__package__).joinpath(name).read_text(encoding="utf8"))


def cohort(path, *, roles=False, clinical=False, targets=False):
    layout = resource("schema.json")
    columns = ["id", *layout["columns"]]
    if roles:
        columns.extend(ROLES)
    if clinical:
        columns.extend(COVARIATES)
    if targets:
        columns.extend(TASKS)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV headers make field identity ambiguous")
    missing = sorted(set(columns).difference(header))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    types = {field: np.float32 for field in layout["columns"]}
    types["id"] = "string"
    frame = pd.read_csv(path, usecols=columns, dtype=types)
    if frame.empty or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Supply exactly one row per unique participant id")
    if roles:
        for field in ROLES:
            if not frame[field].isin(("train", "val", "test", "unused")).all():
                raise ValueError(f"Unknown or missing split in {field}")
        supervised_test = frame.finetune_split == "test"
        if (supervised_test & frame.pretrain_split.isin(("train", "val"))).any():
            raise ValueError(
                "Downstream test participants must be held out of NMR training and selection"
            )
    raw = np.ascontiguousarray(frame[layout["columns"]].to_numpy(dtype=np.float32))
    if np.isinf(raw).any():
        raise ValueError("Use empty cells for missing NMR measurements")
    return frame.drop(columns=layout["columns"]), raw


def partition_key(frame):
    values = pd.util.hash_pandas_object(frame[["id", *ROLES]], index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


def data_digest(arrays, rows):
    """Fingerprint only the selected rows and columns consumed by a training stage."""
    sha = hashlib.sha256()
    for a in arrays:
        sha.update(str((len(rows), a.shape[1:], a.dtype.str)).encode("ascii"))
        for start in range(0, len(rows), 8192):
            sha.update(np.ascontiguousarray(a[rows[start : start + 8192]]).tobytes())
    return sha.hexdigest()


def rows_for(frame, split_column, role):
    rows = np.flatnonzero(frame[split_column].to_numpy() == role)
    if not len(rows):
        raise ValueError(f"No {role} participants in {split_column}")
    return rows


def tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def measurement_batch(raw, rows, stats, device, plans=None):
    x = raw[rows]
    observed = np.isfinite(x)
    visible = observed if plans is None else observed & ~plans
    z = (x - stats["mean"]) / stats["std"]
    return {
        "raw": tensor(x, device),
        "values": tensor(z, device),
        "observed": torch.as_tensor(observed, device=device),
        "visible": torch.as_tensor(visible, device=device),
    }


def clinical_values(frame):
    age, male, bmi = (frame[k].to_numpy(dtype=float) for k in COVARIATES[:3])
    smoking = frame.smoking.fillna("unknown")
    if (
        not np.isfinite(age).all()
        or not np.isin(male, (0, 1)).all()
        or np.isinf(bmi).any()
        or not smoking.isin(("never", "previous", "current", "unknown")).all()
    ):
        raise ValueError("Invalid age, sex coding, BMI or smoking category")
    return np.column_stack(
        (
            age,
            male,
            bmi,
            np.isnan(bmi),
            smoking == "previous",
            smoking == "current",
            smoking == "unknown",
        )
    )


def transformed_targets(frame):
    y = frame[list(TASKS)].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(y).any() or (y[:, 0] <= 0).any() or (y[:, 1:] < 0).any():
        raise ValueError("Observed CRP must be positive and cell counts nonnegative")
    y[:, 0] = np.log(y[:, 0])
    y[:, 1:] = np.log1p(y[:, 1:])
    return y


def fit_clinical_scales(cov, y, rows):
    cov_mean, cov_sd = np.nanmean(cov[rows], axis=0), np.nanstd(cov[rows], axis=0)
    mean, sd = np.nanmean(y[rows], axis=0), np.nanstd(y[rows], axis=0)
    if not np.isfinite(np.r_[cov_mean, cov_sd, mean, sd]).all() or (sd <= 0).any():
        raise ValueError(
            "Supervised training needs observations for each covariate and variation for each outcome"
        )
    return {
        "cov_mean": cov_mean,
        "cov_sd": np.where(cov_sd > 1e-8, cov_sd, 1.0),
        "mean": mean,
        "sd": sd,
    }


def covariate_z(cov, scales):
    return np.nan_to_num((cov - scales["cov_mean"]) / scales["cov_sd"], nan=0.0).astype(np.float32)


def readout_inputs(raw, cov, rows, layout, stats, device):
    z = (raw[rows] - stats["mean"]) / stats["std"]
    contrasts = composition_coordinates(raw[rows], layout["family_indices"], stats)
    return tensor(z, device), tensor(cov[rows], device), tensor(contrasts, device)
