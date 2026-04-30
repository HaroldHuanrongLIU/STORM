from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from storm_surgwmbench.data.surgwmbench import (
    SOURCE_TO_CODE,
    SurgWMBenchClipDataset,
    SurgWMBenchFrameDataset,
)


def test_sparse_dataset_loads_exactly_20_anchors(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        dataset_root=toy_surgwmbench_root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["frames"].shape == (20, 3, 32, 32)
    assert sample["human_anchor_coords_px"].shape == (20, 2)
    assert sample["human_anchor_coords_norm"].shape == (20, 2)
    assert sample["sampled_indices"].shape == (20,)
    assert sample["frame_indices"].tolist() == sample["sampled_indices"].tolist()
    assert sample["selected_coord_sources"].tolist() == [SOURCE_TO_CODE["human"]] * 20
    assert all(path.endswith(".png") for path in sample["frame_paths"])


def test_dense_dataset_loads_variable_length_sources(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        dataset_root=toy_surgwmbench_root,
        manifest="manifests/train.jsonl",
        image_size=16,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample0 = dataset[0]
    sample1 = dataset[1]

    assert sample0["frames"].shape[0] == 25
    assert sample1["frames"].shape[0] == 31
    assert sample0["dense_coords_norm"].shape == (25, 2)
    assert sample0["selected_coord_sources"].shape == (25,)
    assert SOURCE_TO_CODE["interpolated"] in sample0["selected_coord_sources"].tolist()
    assert sample0["selected_label_weights"][0].item() == pytest.approx(1.0)
    first_interpolated = sample0["selected_coord_sources"].tolist().index(SOURCE_TO_CODE["interpolated"])
    assert sample0["selected_label_weights"][first_interpolated].item() == pytest.approx(0.5)


def test_window_dataset_returns_deterministic_contiguous_window(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        dataset_root=toy_surgwmbench_root,
        manifest="manifests/train.jsonl",
        image_size=16,
        frame_sampling="window",
        max_frames=8,
    )

    sample = dataset[1]
    indices = sample["frame_indices"]

    assert indices.shape == (8,)
    assert torch.equal(indices[1:] - indices[:-1], torch.ones(7, dtype=torch.long))


def test_interpolation_method_switching_loads_selected_file(toy_surgwmbench_root: Path) -> None:
    linear = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="linear",
        frame_sampling="dense",
        image_size=16,
    )[0]
    pchip = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="pchip",
        frame_sampling="dense",
        image_size=16,
    )[0]

    assert linear["interpolation_path"].endswith(".linear.json")
    assert pchip["interpolation_path"].endswith(".pchip.json")
    first_interpolated = linear["selected_coord_sources"].tolist().index(SOURCE_TO_CODE["interpolated"])
    assert not torch.allclose(linear["selected_coords_norm"][first_interpolated], pchip["selected_coords_norm"][first_interpolated])
    assert torch.allclose(linear["selected_coords_norm"][0], pchip["selected_coords_norm"][0])


def test_strict_mode_rejects_missing_interpolation_file(toy_surgwmbench_root: Path) -> None:
    missing = toy_surgwmbench_root / "interpolations" / "video_01" / "traj_001.linear.json"
    missing.unlink()
    dataset = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="linear",
        frame_sampling="sparse_anchors",
        image_size=16,
    )

    with pytest.raises(FileNotFoundError, match="Interpolation file not found"):
        _ = dataset[0]


def test_strict_mode_rejects_wrong_dataset_version(toy_surgwmbench_root: Path) -> None:
    manifest = toy_surgwmbench_root / "manifests" / "train.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["dataset_version"] = "Legacy"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_version"):
        SurgWMBenchClipDataset(toy_surgwmbench_root, "manifests/train.jsonl")

    dataset = SurgWMBenchClipDataset(toy_surgwmbench_root, "manifests/train.jsonl", allow_legacy_version=True)
    assert len(dataset) == 2


def test_frame_dataset_returns_image_and_metadata(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchFrameDataset(toy_surgwmbench_root, "manifests/train.jsonl", image_size=20, max_frames_per_clip=3)

    image, metadata = dataset[0]

    assert len(dataset) == 6
    assert image.shape == (3, 20, 20)
    assert metadata["local_frame_idx"] == 0
    assert metadata["frame_path"].endswith(".png")
