"""Tests for the experiment harness: reproducibility, optimiser groups, held-out
selection, gradient telemetry and run comparison.

None of these check that the model is good. They check that a *comparison between two
models* means something - that identical runs agree, that runs record what produced
them, that "best" is chosen on held-out data, and that the comparison tool reports its
own noise floor. An ablation without these is a table of anecdotes.
"""
import csv
import json
import math
import statistics

import pytest
import torch

from src.models.world_on_rails import WorldOnRailsPolicy, QwenWorldOnRailsPolicy
from src.training.wor_trainer import WorldOnRailsTrainer


def test_trainer_excludes_gates_and_norms_from_weight_decay(tmp_path):
    """AdamW weight decay on the trunk's alpha residual gates pulls every block's
    contribution toward zero - it penalizes the model for letting its layers
    contribute at all. Gates, norm gains, embeddings and learned tokens belong in a
    weight_decay=0 group."""
    policy = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4)
    trainer = WorldOnRailsTrainer(
        model=policy, data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False
    )

    gate_ids = {id(p) for n, p in policy.named_parameters()
                if n.endswith(("alpha_attn", "alpha_ffn", "policy_token", "vision_pos", "type_embed"))}
    assert gate_ids, "expected the Qwen trunk to expose residual gates and learned tokens"

    for group in trainer.optimizer.param_groups:
        if any(id(p) in gate_ids for p in group["params"]):
            assert group["weight_decay"] == 0.0

    # And the opt-out still reproduces the original behaviour for comparison runs.
    legacy = WorldOnRailsTrainer(
        model=QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck2"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False,
        decay_gates_and_norms=True
    )
    assert all(len(g["params"]) == 0 for g in legacy.optimizer.param_groups if g["weight_decay"] == 0.0)


def test_scheduler_warms_up_then_decays(tmp_path):
    """A bare cosine schedule starts the heads at full LR on batch 1. A conv head
    tolerates that; a 12-layer pre-norm transformer characteristically does not."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=2, num_workers=0, device="cpu", synthetic_samples=64,
        use_mlflow=False, warmup_frac=0.1, lr_heads=3e-4
    )
    scheduler = trainer._build_scheduler(num_epochs=20)

    lrs = []
    for _ in range(20 * len(trainer.train_loader)):
        lrs.append(trainer.optimizer.param_groups[1]["lr"])
        scheduler.step()

    peak = max(lrs)
    assert lrs[0] < peak / 2, "LR did not start low - warmup is not happening"
    assert peak == pytest.approx(3e-4, rel=1e-3), "warmup should reach the configured head LR"
    assert lrs[-1] < peak / 10, "LR did not decay after warmup"
    assert lrs.index(peak) == pytest.approx(int(0.1 * len(lrs)), abs=2)


def test_trainer_validates_and_selects_best_on_val(tmp_path):
    """`val_data_dir` used to be stored and never read, and 'best' was chosen on
    training loss - which cannot distinguish generalization from memorization."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=40,
        use_mlflow=False, val_split=0.25
    )
    assert trainer.val_loader is not None
    assert len(trainer.val_loader.dataset) > 0
    assert len(trainer.train_loader.dataset) + len(trainer.val_loader.dataset) == 40

    val = trainer.validate()
    for key in ("val_loss", "val_wp_ade_m", "val_wp_lateral_error_m", "val_wp_longitudinal_error_m"):
        assert key in val and math.isfinite(val[key])


def test_grad_norm_telemetry_survives_a_nonfinite_batch(tmp_path):
    """One non-finite batch used to poison the whole epoch's grad_norm mean to inf,
    which is what made that column unreadable in the first telemetry files. The AMP
    GradScaler produces those deliberately, so they are counted, not averaged."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=16,
        use_mlflow=False, grad_clip=0.5
    )
    metrics = trainer.train_epoch(epoch=1)

    assert math.isfinite(metrics["grad_norm"])
    assert metrics["nonfinite_grad_batches"] == 0
    assert metrics["clipped_grad_batches"] > 0, "a 0.5 clip should bite on an untrained model"
    assert metrics["grad_clip"] == 0.5


def test_checkpoint_records_vision_geometry(tmp_path):
    """A Qwen checkpoint trained on one pooled vision token loads into a 64-token
    model without error under strict=False, then drives on untrained positional
    embeddings. Stamping the geometry into the checkpoint is what prevents that."""
    from src.models.world_on_rails.wor_loader import load_wor_model

    save_dir = tmp_path / "ck"
    trainer = WorldOnRailsTrainer(
        model=QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4),
        data_dir=str(tmp_path / "none"), save_dir=str(save_dir),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False
    )
    trainer.train(num_epochs=1, save_freq=1)

    ckpt = torch.load(str(save_dir / "latest_model.pth"), map_location="cpu")
    assert ckpt["config"]["vision_grid"] == 4
    assert ckpt["config"]["num_vision_tokens"] == 16

    # The caller does not have to remember the geometry - it is read back off disk.
    restored = load_wor_model(
        checkpoint_path=str(save_dir / "latest_model.pth"), backbone_name="resnet18",
        pretrained_backbone=False, policy_arch="qwen100m"
    )
    assert restored.vision_grid == 4
    assert restored.num_vision_tokens == 16


def _seeded_first_batch_loss(tmp_path, seed, name):
    """Trains one epoch under a fixed seed and returns the resulting metrics."""
    from src.training.seeding import seed_everything

    seed_everything(seed)
    trainer = WorldOnRailsTrainer(
        model=QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / name),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=16,
        use_mlflow=False, val_split=0.25, seed=seed
    )
    return trainer.train_epoch(epoch=1)


def test_seeding_makes_runs_reproducible(tmp_path):
    """Nothing in this pipeline was seeded before. An ablation cannot attribute a
    difference to a component until two identical runs are known to agree, so this is
    a prerequisite for every comparison that follows - not a convenience."""
    a = _seeded_first_batch_loss(tmp_path, 0, "a")
    b = _seeded_first_batch_loss(tmp_path, 0, "b")
    c = _seeded_first_batch_loss(tmp_path, 1, "c")

    assert a["total_loss"] == pytest.approx(b["total_loss"], rel=1e-9), \
        "same seed produced different results - runs are not reproducible"
    assert a["wp_lateral_error_m"] == pytest.approx(b["wp_lateral_error_m"], rel=1e-9)
    assert a["total_loss"] != pytest.approx(c["total_loss"], rel=1e-6), \
        "different seeds produced identical results - the seed is not being used"


def test_worker_init_fn_decorrelates_numpy_across_workers():
    """Torch reseeds its own RNG per DataLoader worker; NumPy's is inherited unchanged,
    so without this every worker draws the same NumPy numbers."""
    import numpy as np
    from src.training.seeding import make_worker_init_fn

    init = make_worker_init_fn(1234)
    draws = []
    for worker_id in range(4):
        init(worker_id)
        draws.append(float(np.random.rand()))

    assert len(set(draws)) == len(draws), "workers drew identical NumPy values"

    init(0)
    assert float(np.random.rand()) == draws[0], "worker seeding is not reproducible"


def test_run_config_and_telemetry_identify_the_run(tmp_path):
    """A sweep whose rows cannot be traced back to the flags that produced them is not
    an experiment. compare_wor_runs.py reads both of these."""
    save_dir = tmp_path / "run"
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(save_dir),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=16,
        use_mlflow=False, val_split=0.25, seed=7
    )
    trainer.train(num_epochs=1, save_freq=1)

    assert "seed" in trainer.TELEMETRY_FIELDS
    with open(save_dir / "wor_training_telemetry.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and rows[-1]["seed"] == "7"

    ckpt = torch.load(str(save_dir / "latest_model.pth"), map_location="cpu")
    assert ckpt["config"]["seed"] == 7


def test_compare_runs_reports_seed_spread_as_a_noise_floor(tmp_path):
    """The comparison must group repeats of one configuration and report their spread.
    A component that moves the metric less than that spread has not been shown to do
    anything, and the tool has to say so rather than rank it."""
    import compare_wor_runs

    fields = ["epoch", "num_batches", "total_loss", "val_loss", "val_wp_ade_m",
              "val_wp_fde_m", "val_wp_lateral_error_m", "val_wp_longitudinal_error_m",
              "grad_norm", "clipped_grad_batches", "nonfinite_grad_batches",
              "samples_per_sec", "wall_time_s", "seed"]

    def make_run(name, seed, val_loss, epochs=2):
        run_dir = tmp_path / name
        run_dir.mkdir()
        with open(run_dir / "wor_training_telemetry.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                writer.writerow({
                    "epoch": epoch, "num_batches": 10, "total_loss": val_loss + 0.1,
                    "val_loss": val_loss + (0.5 if epoch < epochs else 0.0),
                    "val_wp_ade_m": 0.3, "val_wp_fde_m": 0.6,
                    "val_wp_lateral_error_m": 0.02, "val_wp_longitudinal_error_m": 0.25,
                    "grad_norm": 4.0, "clipped_grad_batches": 5,
                    "nonfinite_grad_batches": 0, "samples_per_sec": 1000,
                    "wall_time_s": 100 * epoch, "seed": seed,
                })
        with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
            json.dump({"run_label": name.rsplit("_s", 1)[0], "seed": seed,
                       "policy_arch": "qwen30m"}, handle)
        return str(run_dir)

    dirs = [make_run("base_s0", 0, 1.00), make_run("base_s1", 1, 1.20),
            make_run("variant_s0", 0, 1.05)]

    summaries = [compare_wor_runs.summarize(d) for d in dirs]
    assert all(s is not None for s in summaries)
    assert summaries[0]["best_loss"] == pytest.approx(1.00)
    assert summaries[0]["selected_on"] == "val_loss"
    assert summaries[0]["best_epoch"] == 2  # the epoch that actually scored best

    labels = {s["label"] for s in summaries}
    assert labels == {"base", "variant"}

    # base spread is 0.20; variant beats base's mean (1.10) by only 0.05, i.e. less
    # than the noise floor - exactly the case the tool exists to make visible.
    base = [s for s in summaries if s["label"] == "base"]
    spread = max(s["best_loss"] for s in base) - min(s["best_loss"] for s in base)
    variant = [s for s in summaries if s["label"] == "variant"][0]
    assert spread == pytest.approx(0.20)
    assert abs(variant["best_loss"] - statistics.mean(s["best_loss"] for s in base)) < spread


def test_compare_runs_splits_appended_runs(tmp_path):
    """Runs append to the same CSV, so one file can hold several - the first shipped
    telemetry file held an 8-epoch run followed by a 50-epoch one."""
    import compare_wor_runs

    rows = ([{"epoch": str(e)} for e in range(1, 9)]
            + [{"epoch": str(e)} for e in range(1, 51)])
    runs = compare_wor_runs.split_runs(rows)

    assert len(runs) == 2
    assert [len(r) for r in runs] == [8, 50]
