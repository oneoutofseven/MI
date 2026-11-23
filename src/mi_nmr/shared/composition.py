"""Training-set moments and within-subclass lipid coordinates."""

import numpy as np


def helmert_four():
    return (
        np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], np.float64)
        / np.sqrt(np.array([2, 6, 12], np.float64))[:, None]
    )


def train_statistics(raw, rows, layout, block=8192):
    width = raw.shape[1]
    total, square, count = np.zeros(width), np.zeros(width), np.zeros(width, np.int64)
    k = len(layout["family_indices"])
    comp_total, comp_square = np.zeros((k, 3)), np.zeros((k, 3))
    comp_count = np.zeros(k, np.int64)
    fraction_total = np.zeros((k, 4))
    lipid_total, particle_total, particle_count = np.zeros(k), np.zeros(k), np.zeros(k, np.int64)
    h = helmert_four()
    indices = np.asarray(layout["family_indices"])
    for start in range(0, len(rows), block):
        x = np.array(raw[rows[start : start + block]], np.float64)
        observed = np.isfinite(x)
        clean = np.where(observed, x, 0.0)
        total += clean.sum(0)
        square += np.square(clean).sum(0)
        count += observed.sum(0)
        groups = x[:, indices]
        p, c = groups[:, :, 0], groups[:, :, 1:]
        valid_p = np.isfinite(p) & (p > 0)
        particle_total += np.where(valid_p, p, 0.0).sum(0)
        particle_count += valid_p.sum(0)
        valid = np.isfinite(c).all(-1) & (c >= 0).all(-1)
        s = np.where(valid[..., None], c, 0.0).sum(-1)
        valid &= s > 0
        fractions = np.where(valid[..., None], c, 0.0) / np.maximum(
            s[..., None], np.finfo(np.float64).tiny
        )
        contrasts = fractions @ h.T
        comp_total += contrasts.sum(0)
        comp_square += np.square(contrasts).sum(0)
        comp_count += valid.sum(0)
        fraction_total += fractions.sum(0)
        lipid_total += np.where(valid, s, 0.0).sum(0)
    if (count < 2).any() or (comp_count < 2).any() or (particle_count < 2).any():
        raise ValueError("Insufficient measured TRAIN values for normalization")
    mean = total / count
    variance = np.maximum((square - total * total / count) / (count - 1), 0.0)
    sd = np.sqrt(variance)
    sd[sd < 1e-12] = 1.0
    cm = comp_total / comp_count[:, None]
    cv = np.maximum(
        (comp_square - comp_total * comp_total / comp_count[:, None]) / (comp_count[:, None] - 1),
        0.0,
    )
    cs = np.maximum(np.sqrt(cv), 1e-3)
    prior = fraction_total / comp_count[:, None]
    return {
        "mean": mean,
        "std": sd,
        "observed_count": count,
        "composition_mean": cm,
        "composition_std": cs,
        "composition_count": comp_count,
        "composition_prior": prior,
        "particle_scale": particle_total / particle_count,
        "lipid_scale": lipid_total / comp_count,
        "helmert": h,
    }


def composition_coordinates(raw, family_indices, statistics):
    """Return 14 x 3 standardized Helmert contrasts in canonical subclass order.

    Use original concentrations. A missing, negative or zero-total subclass has
    undefined composition and contributes its TRAIN-center (zero after scaling).
    """
    raw = np.asarray(raw)
    groups = np.asarray(family_indices)
    if raw.ndim != 2 or raw.shape[1] != 107 or groups.shape != (14, 5):
        raise ValueError("Expected the registered 107-field schema and 14 five-field subclasses")
    lipids = raw[:, groups[:, 1:]].astype(np.float64)
    totals = lipids.sum(axis=-1)
    valid = np.isfinite(lipids).all(-1) & (lipids >= 0).all(-1) & (totals > 0)
    fractions = np.divide(
        lipids, totals[..., None], out=np.zeros_like(lipids), where=valid[..., None]
    )
    contrast = fractions @ statistics["helmert"].T
    standardized = (contrast - statistics["composition_mean"]) / statistics["composition_std"]
    return np.where(valid[..., None], standardized, 0.0).reshape(len(raw), 42).astype(np.float32)
