"""Command-line entry points for fitting, inference and held-out evaluation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoints import (
    atomic_write,
    encoder_record,
    make_encoder,
    make_readout,
    numpy_dict,
    readout_record,
    save_json,
)
from .data import (
    TASKS,
    clinical_values,
    cohort,
    covariate_z,
    measurement_batch,
    partition_key,
    resource,
    rows_for,
    transformed_targets,
)
from .engine import batches


def arguments(argv=None):
    parser = argparse.ArgumentParser(prog="mi-nmr", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("pretrain", "finetune", "encode", "predict", "evaluate"):
        p = actions.add_parser(name)
        p.add_argument(
            "--data", type=Path, required=True, help="CSV with named baseline NMR fields"
        )
        p.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Run directory for fitting; output file for inference",
        )
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--threads", type=int, default=4)
        if name in ("pretrain", "finetune"):
            p.add_argument(
                "--config", type=Path, help="Full JSON recipe (default: study architecture)"
            )
            p.add_argument(
                "--resume", action="store_true", help="Resume at the last completed epoch"
            )
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=64)
        if name in ("finetune", "evaluate", "predict"):
            p.add_argument(
                "--encoder",
                type=Path,
                required=name == "finetune",
                help="Pretraining checkpoint; required for older two-file readouts",
            )
        if name == "evaluate":
            p.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args(argv)


@torch.inference_mode()
def inference(args, device):
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    for source in (args.data, args.checkpoint, getattr(args, "encoder", None)):
        if source is not None and args.out.resolve() == source.resolve():
            raise ValueError("Output cannot overwrite an input file")
    if args.action == "encode":
        checkpoint = encoder_record(args.checkpoint)
        encoder = make_encoder(checkpoint).to(device).eval()
        frame, raw = cohort(args.data)
        stats = numpy_dict(checkpoint["statistics"])
        embeddings = np.empty(
            (len(raw), checkpoint["encoder"]["embedding_width"]), dtype=np.float32
        )
        for rows in batches(np.arange(len(raw)), args.batch_size):
            b = measurement_batch(raw, rows, stats, device)
            embeddings[rows] = (
                encoder(b["values"], b["observed"], check=False).float().cpu().numpy()
            )

        def write_npz(path):
            with path.open("wb") as handle:
                np.savez(handle, id=frame.id.to_numpy(dtype=str), embedding=embeddings)

        atomic_write(args.out, write_npz)
        return
    from ..finetuning.train import predict_rows, scores

    evaluate = args.action == "evaluate"
    record = readout_record(args.checkpoint, args.encoder)
    model = make_readout(record).eval().to(device)
    frame, raw = cohort(args.data, clinical=True, roles=evaluate, targets=evaluate)
    if evaluate:
        if "partition_key" in record and partition_key(frame) != record["partition_key"]:
            raise ValueError(
                "Use the same participant/split manifest to evaluate a study partition"
            )
        selected = rows_for(frame, "finetune_split", args.split)
        frame, raw = frame.iloc[selected], raw[selected]
    scales = numpy_dict(record["scales"])
    cov = covariate_z(clinical_values(frame), scales)
    predictions = predict_rows(
        model,
        raw,
        cov,
        np.arange(len(raw)),
        record["layout"],
        numpy_dict(record["statistics"]),
        device,
        args.batch_size,
    )
    if evaluate:
        result = scores(predictions, transformed_targets(frame), scales)
        save_json(args.out, {"split": args.split, "participants": len(frame), **result})
        print(
            f"{args.split}: five-task MSE {result['five_task_mse']:.6f}  "
            f"macro standardized RMSE {result['macro_standardized_rmse']:.6f}"
        )
    else:
        columns = ["ln_CRP", *["log1p_" + task for task in TASKS[1:]]]
        values = predictions.astype(np.float64) * scales["sd"] + scales["mean"]
        output = pd.DataFrame(values, columns=columns)
        output.insert(0, "id", frame.id.to_numpy())
        atomic_write(args.out, lambda path: output.to_csv(path, index=False))


def main(argv=None):
    args = arguments(argv)
    if args.threads < 1:
        raise ValueError("CPU threads must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported device types are cpu and cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if args.action == "finetune":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
    if args.action in ("pretrain", "finetune"):
        cfg = json.loads(args.config.read_text()) if args.config else resource("recipe.json")
        if args.action == "pretrain":
            from ..pretraining.train import pretrain

            pretrain(args.data, args.out, cfg, device, args.resume)
        else:
            from ..finetuning.train import finetune

            finetune(args.data, args.encoder, args.out, cfg, device, args.resume)
    else:
        inference(args, device)


if __name__ == "__main__":
    main()
