"""Matched semantic/random mask banks; statistics never depend on outcomes."""

from __future__ import annotations

import numpy as np


def random_plan(width, count, rng):
    result = np.zeros(width, bool)
    result[rng.choice(width, count, replace=False)] = True
    return result


def biological_plan(width, count, families, other, rng):
    if count < 5 or not len(families):
        return random_plan(width, count, rng)
    first = int(rng.integers(len(families)))
    chosen = list(families[first])
    units = [list(group) for j, group in enumerate(families) if j != first]
    units += [[int(j)] for j in other]
    for index in rng.permutation(len(units)):
        unit = units[index]
        remaining = count - len(chosen)
        if remaining <= 0:
            break
        chosen.extend(
            unit if len(unit) <= remaining else rng.permutation(unit)[:remaining].tolist()
        )
    result = np.zeros(width, bool)
    result[chosen] = True
    if result.sum() != count:
        raise ValueError("Biological mask does not cover the requested number of fields")
    return result


def randomize_degrees(bank, rng, attempts):
    result = bank.copy()
    accepted = 0
    for _ in range(attempts):
        a, b = rng.integers(len(result), size=2)
        if a == b:
            continue
        forward = np.flatnonzero(result[a] & ~result[b])
        backward = np.flatnonzero(result[b] & ~result[a])
        if not len(forward) or not len(backward):
            continue
        u, v = int(rng.choice(forward)), int(rng.choice(backward))
        result[a, u] = result[b, v] = False
        result[a, v] = result[b, u] = True
        accepted += 1
    np.testing.assert_array_equal(bank.sum(0), result.sum(0))
    np.testing.assert_array_equal(bank.sum(1), result.sum(1))
    return result, accepted


def build_banks(layout, settings):
    rng = np.random.default_rng(settings["seed"])
    width = len(layout["columns"])
    low, high = settings["rate_range"]
    if not 0 < low <= high < 1:
        raise ValueError("Mask rates must be between zero and one")
    families, other = layout["family_indices"], layout["other_indices"]
    masks = []
    for index in range(settings["bank_size"]):
        count = min(width - 1, max(1, round(width * rng.uniform(low, high))))
        if index % 2 == 0:
            masks.append(random_plan(width, count, rng))
        else:
            masks.append(biological_plan(width, count, families, other, rng))
    biological = np.stack(masks)
    control, accepted = randomize_degrees(
        biological, np.random.default_rng(settings["seed"] + 1), settings["swap_attempts"]
    )

    def composition_fraction(bank):
        return float(np.mean([bank[:, np.array(group)[1:]].all(1).mean() for group in families]))

    audit = {
        "plans": len(biological),
        "fields": width,
        "same_plan_target_counts": True,
        "same_per_field_target_frequency": True,
        "accepted_degree_preserving_swaps": accepted,
        "fraction_changed_cells": float(np.mean(biological != control)),
        "composition_hidden_fraction_measurement": composition_fraction(biological),
        "composition_hidden_fraction_random": composition_fraction(control),
        "per_field_target_frequency": biological.mean(0).tolist(),
        "minimum_targets": int(biological.sum(1).min()),
        "maximum_targets": int(biological.sum(1).max()),
    }
    return biological, control, audit


def validation_plans(layout, settings, people):
    rng = np.random.default_rng(settings["validation_seed"])
    width = len(layout["columns"])
    output, names = [], []
    for rate in settings["validation_rates"]:
        count = min(width - 1, max(1, round(width * rate)))
        output.append(np.stack([random_plan(width, count, rng) for _ in range(people)]))
        names.append(f"random_{rate:g}")
    count = min(width - 1, max(1, round(width * 0.3)))
    output.append(
        np.stack(
            [
                biological_plan(
                    width, count, layout["family_indices"], layout["other_indices"], rng
                )
                for _ in range(people)
            ]
        )
    )
    names.append("biological_0.3")
    return np.stack(output), names


def training_bank(layout, settings, mode):
    mixed, matched, _ = build_banks(layout, settings)
    if mode == "mixed":
        return mixed
    if mode == "ordinary":
        return matched
    if mode != "subclass":
        raise ValueError("mask must be mixed, ordinary or subclass")
    rng = np.random.default_rng([settings["seed"], 1])
    for i in range(0, len(mixed), 2):
        mixed[i] = biological_plan(
            mixed.shape[1],
            int(mixed[i].sum()),
            layout["family_indices"],
            layout["other_indices"],
            rng,
        )
    return mixed
