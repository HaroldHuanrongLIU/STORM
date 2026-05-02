from __future__ import annotations

from types import SimpleNamespace

from storm_surgwmbench.adapter import eval_adapter, train_adapter


def test_storm_adapter_train_eval_smoke(tmp_path, toy_surgwmbench_root):
    output_dir = tmp_path / "run"
    train_result = train_adapter(
        SimpleNamespace(
            dataset_root=str(toy_surgwmbench_root),
            manifest="manifests/train.jsonl",
            train_manifest="manifests/train.jsonl",
            val_manifest="manifests/val.jsonl",
            target="sparse_20_anchor",
            interpolation_method="linear",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            image_size=32,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            recon_weight=0.01,
            kl_weight=1e-5,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
            seed=7,
        )
    )
    result = eval_adapter(
        SimpleNamespace(
            dataset_root=str(toy_surgwmbench_root),
            manifest="manifests/test.jsonl",
            checkpoint=train_result["checkpoint"],
            target="sparse_20_anchor",
            interpolation_method="linear",
            output=str(output_dir / "metrics.json"),
            batch_size=1,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
        )
    )
    assert result["baseline"] == "storm"
    assert result["experiment_target"] == "sparse_20_anchor"
    assert "ade" in result["metrics_overall"]
