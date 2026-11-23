import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mi_nmr.cli import main
from mi_nmr.composition import composition_coordinates, train_statistics
from mi_nmr.data import TASKS, cohort, measurement_batch, resource
from mi_nmr.masks import training_bank
from mi_nmr.model import FieldSupervisedModel, MeasurementModel
from mi_nmr.objectives import FieldObjective
from mi_nmr.training import scores

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "small.json"


@pytest.fixture()
def data(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "input.csv"
    runpy.run_path(str(ROOT / "examples" / "make_demo.py"))["generate"](path)
    return path


def test_schema_mask_budget_and_frequency():
    layout = resource("schema.json")
    settings = json.loads(SMALL.read_text())["masking"]
    family = np.asarray(layout["family_indices"])
    assert family.shape == (14, 5)
    assert sorted([*family.ravel(), *layout["other_indices"]]) == list(range(107))
    mixed, ordinary, subclass = [
        training_bank(layout, settings, m) for m in ("mixed", "ordinary", "subclass")
    ]
    np.testing.assert_array_equal(mixed.sum(0), ordinary.sum(0))
    np.testing.assert_array_equal(mixed.sum(1), ordinary.sum(1))
    np.testing.assert_array_equal(mixed.sum(1), subclass.sum(1))
    np.testing.assert_array_equal(mixed[1::2], subclass[1::2])
    assert subclass[:, family].all(-1).any(-1).all()


def test_field_identity_hidden_information_and_subtotals(data):
    layout = resource("schema.json")
    cfg = json.loads(SMALL.read_text())
    frame, raw = cohort(data, roles=True)
    tr = np.flatnonzero(frame.pretrain_split == "train")
    stats = train_statistics(raw, tr, layout)
    core = MeasurementModel(layout, stats, cfg["encoder"], cfg["decoder"])
    model = FieldSupervisedModel(core).eval()
    # Make the field head sensitive to the hidden states even before training.
    torch.nn.init.normal_(model.field_head.weight)
    b = measurement_batch(raw, np.arange(6), stats, "cpu")
    visible = b["observed"].clone()
    visible[:, ::3] = False
    changed = torch.where(b["observed"] & ~visible, b["values"] + 11.0, b["values"])
    ids = torch.arange(106, -1, -1)
    with torch.no_grad():
        reference = model(b["values"], b["observed"], visible)
        altered = model(changed, b["observed"], visible)
        reordered = model(changed[:, ids], b["observed"][:, ids], visible[:, ids], ids)
    for key in ("embedding", "prediction", "local"):
        torch.testing.assert_close(altered[key], reference[key], rtol=0, atol=0)
    torch.testing.assert_close(reordered["embedding"], reference["embedding"])
    torch.testing.assert_close(reordered["local"], reference["local"][:, ids], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(reference["raw_groups"][..., 1:].sum(-1), reference["lipid_sum"])
    torch.testing.assert_close(
        reference["fractions"].sum(-1), torch.ones_like(reference["lipid_sum"])
    )


def test_composition_eligibility_and_field_targets(data):
    frame, raw = cohort(data, roles=True)
    layout = resource("schema.json")
    stats = train_statistics(raw, np.flatnonzero(frame.pretrain_split == "train"), layout)
    subset = np.array(raw[:3], copy=True)
    lipids = layout["family_indices"][0][1:]
    subset[:, lipids] = [1.0, 2.0, 3.0, 4.0]
    subset[2, lipids[0]] = np.nan
    mask = np.zeros(subset.shape, dtype=bool)
    mask[:, lipids] = True
    mask[1, lipids[0]] = False
    b = measurement_batch(subset, np.arange(3), stats, "cpu", mask)
    loss = FieldObjective(layout, stats, 0.2)
    n, groups = loss.denominators(b)
    assert int(n) == 3 and int(groups) == 1
    out = {
        "prediction": torch.zeros_like(b["values"]),
        "local": torch.ones_like(b["values"]),
        "fractions": torch.full((3, 14, 4), 0.25),
    }
    parts = loss(out, b)
    hidden = b["observed"] & ~b["visible"]
    expected = sum(
        (out["local"][i, h] - b["values"][i, h]).abs().mean() for i, h in enumerate(hidden)
    )
    torch.testing.assert_close(parts["local_sum"], expected)
    torch.testing.assert_close(parts["numeric_sum"], parts["global_numeric_sum"] + 0.2 * expected)
    context = composition_coordinates(subset, layout["family_indices"], stats)
    assert np.all(context[2, :3] == 0)


def test_new_people_and_partitions(data, tmp_path):
    frame = pd.read_csv(data)
    columns = resource("schema.json")["columns"]
    new = tmp_path / "new.csv"
    frame[["id", *reversed(columns)]].to_csv(new, index=False)
    np.testing.assert_allclose(cohort(new)[1], cohort(data)[1], equal_nan=True)
    frame.loc[0, "finetune_split"] = "test"
    frame.to_csv(new, index=False)
    with pytest.raises(ValueError, match="held out"):
        cohort(new, roles=True)


def test_cli_and_resume(data, tmp_path, monkeypatch):
    from mi_nmr import engine

    def run_stage_script(stage, args):
        script = ROOT / stage / "train.py"
        with monkeypatch.context() as patch:
            patch.setattr("sys.argv", [str(script), *args])
            runpy.run_path(str(script), run_name="__main__")

    base = ["--data", str(data), "--device", "cpu", "--threads", "1"]
    cfg = ["--config", str(SMALL)]
    p, f = tmp_path / "pre", tmp_path / "fit"
    pretrain = ["pretrain", *base, *cfg, "--out", str(p)]
    real_save = engine.save_json

    def interrupt_after_epoch(path, value):
        if len(value) == 1:
            raise KeyboardInterrupt
        real_save(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "save_json", interrupt_after_epoch)
        with pytest.raises(KeyboardInterrupt):
            main(pretrain)
    main([*pretrain, "--resume"])
    clean = tmp_path / "clean"
    run_stage_script("pretraining", [*base, *cfg, "--out", str(clean)])
    assert json.loads((p / "history.json").read_text()) == json.loads(
        (clean / "history.json").read_text()
    )
    tune = ["finetune", *base, *cfg, "--encoder", str(p / "best.pt"), "--out", str(f)]
    run_stage_script("finetuning", tune[1:])
    main([*tune, "--resume"])
    checkpoint = ["--checkpoint", str(f / "best.pt")]
    result = tmp_path / "metrics.json"
    main(["evaluate", *base, *checkpoint, "--out", str(result)])
    metrics = json.loads(result.read_text())
    assert metrics["participants"] == 16 and set(metrics["per_task"]) == set(TASKS)
    csv_path = tmp_path / "pred.csv"
    main(["predict", *base, *checkpoint, "--out", str(csv_path)])
    predictions = pd.read_csv(csv_path)
    table = pd.read_csv(data)
    assert predictions.id.tolist() == table.id.tolist()
    assert np.isfinite(predictions.iloc[:, 1:].to_numpy()).all()
    out = tmp_path / "embedding.npz"
    main(["encode", *base, "--checkpoint", str(p / "best.pt"), "--out", str(out)])
    with np.load(out, allow_pickle=False) as z:
        assert z["id"].tolist() == table.id.tolist()
        assert z["embedding"].shape == (96, 16)


def test_phenotype_metrics_use_training_scales():
    absolute = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 5.0, 4.0, 7.0, 6.0]])
    mean, sd = np.array([2.0, 3.0, 1.0, 0.0, 5.0]), np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    pred = (absolute - mean) / sd + 0.5
    report = scores(pred, absolute, {"mean": mean, "sd": sd})
    assert report["five_task_mse"] == pytest.approx(0.25)
    assert report["macro_standardized_rmse"] == pytest.approx(0.5)
    for j, task in enumerate(TASKS):
        assert report["per_task"][task]["rmse_transformed"] == pytest.approx(0.5 * sd[j])
