"""Portable model checkpoints and atomic run records."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .data import TASKS, resource
from .encoder import ParticipantEncoder


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save(path, payload):
    atomic_write(path, lambda tmp: torch.save(payload, tmp))


def save_json(path, payload):
    atomic_write(
        path, lambda tmp: tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    )


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def output_run(directory, resume):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exists = (directory / "last.pt").exists()
        if exists != resume:
            raise ValueError(
                "Use --resume with an existing last.pt, or use a fresh output directory"
            )
        if not resume and (directory / "best.pt").exists():
            raise ValueError("Output already contains a selected checkpoint")
        yield directory


def numpy_dict(record):
    return {k: np.asarray(v) for k, v in record.items()}


def tensor_dict(record):
    return {k: torch.as_tensor(v) for k, v in record.items()}


def encoder_record(path):
    source = load(path)
    kind = source.get("format")
    if kind == "mi.encoder.v1":
        result = source
    elif kind == "mi.pretrain.v1":
        state = {
            k.removeprefix("core.encoder."): v
            for k, v in source["state"].items()
            if k.startswith("core.encoder.")
        }
        result = {
            **{k: source[k] for k in ("layout", "encoder", "statistics", "partition_key")},
            "format": "mi.encoder.v1",
            "state": state,
        }
    elif kind == "nmr_measure_pretraining_v1":
        state = {
            k.removeprefix("encoder."): v
            for k, v in source["model"].items()
            if k.startswith("encoder.")
        }
        result = {
            "format": "mi.encoder.v1",
            "layout": source["layout"],
            "encoder": source["encoder"],
            "statistics": source["statistics"],
            "state": state,
        }
    else:
        raise ValueError("Expected an exported NMR participant encoder")
    if result["layout"] != resource("schema.json"):
        raise ValueError("Checkpoint and input field definitions differ")
    return result


def make_encoder(record):
    encoder = ParticipantEncoder(record["layout"], record["encoder"])
    encoder.load_state_dict(record["state"], strict=True)
    return encoder


def readout_record(path, encoder_path=None):
    record = load(path)
    if record.get("format") == "mi.readout.v1":
        result = record
    elif (
        encoder_path
        and record.get("arm") == "C"
        and record.get("source_sha256") == digest(encoder_path)
    ):
        source = encoder_record(encoder_path)
        result = {
            "format": "mi.readout.v1",
            "layout": source["layout"],
            "encoder": source["encoder"],
            "statistics": source["statistics"],
            "state": record["state"],
            "tasks": record["tasks"],
            "scales": {k: record[k] for k in ("mean", "sd", "cov_mean", "cov_sd")},
        }
    else:
        raise ValueError(
            "Expected a readout checkpoint; an older five-task checkpoint also needs its matching --encoder"
        )
    if result["layout"] != resource("schema.json") or tuple(result["tasks"]) != TASKS:
        raise ValueError("Checkpoint field or phenotype order differs")
    return result


def make_readout(record):
    from ..finetuning.readout import InflammationModel

    encoder = ParticipantEncoder(record["layout"], record["encoder"])
    state = record["state"]
    model = InflammationModel(encoder, state["z_mean"], state["z_std"], 0)
    model.load_state_dict(state, strict=True)
    return model
