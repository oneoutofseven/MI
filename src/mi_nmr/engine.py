"""Shared batching, mixed precision, early stopping and epoch checkpoints."""

import math

import torch

from .checkpoints import load, output_run, save, save_json


def batches(rows, size):
    if size < 1:
        raise ValueError("Batch size must be positive")
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def autocast(device, enabled):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda"
    )


def fit_epochs(
    out, resume, identity, settings, model, optimizer, train_epoch, validate, best_payload, device
):
    """Persist the full optimizer, generator and history at epoch boundaries."""
    with output_run(out, resume) as folder:
        last = folder / "last.pt"
        history = []
        best_at, best = 0, math.inf

        def checkpoint():
            save(
                last,
                {
                    "identity": identity,
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "best_at": best_at,
                    "best": best,
                    "rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                },
            )

        if resume:
            state = load(last)
            if state["identity"] != identity or (state["gpu_rng"] is None) != (
                device.type == "cpu"
            ):
                raise ValueError(
                    "Resume requires the original recipe, data, source and device type"
                )
            model.load_state_dict(state["state"])
            optimizer.load_state_dict(state["optimizer"])
            history, best_at, best = state["history"], state["best_at"], state["best"]
            if best_at and not (folder / "best.pt").is_file():
                raise ValueError("Missing best.pt for this run")
            torch.set_rng_state(state["rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["gpu_rng"], device)
            del state
        else:
            checkpoint()
        for epoch in range(len(history) + 1, settings["max_epochs"] + 1):
            if (
                best_at
                and len(history) >= settings.get("minimum_epochs", 1)
                and len(history) - best_at >= settings["patience"]
            ):
                break
            model.train()
            train_epoch(epoch)
            model.eval()
            with torch.inference_mode():
                score, metrics = validate()
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite validation criterion")
            if score < best - settings.get("minimum_improvement", 0.0):
                best, best_at = score, epoch
                save(folder / "best.pt", {**best_payload(), "epoch": epoch, "score": score})
            history.append({"epoch": epoch, "score": score, **metrics})
            checkpoint()
            save_json(folder / "history.json", history)
            print(
                f"epoch {epoch}  validation {score:.6f}  best {best:.6f} (epoch {best_at})",
                flush=True,
            )
        save_json(folder / "history.json", history)
        return history
