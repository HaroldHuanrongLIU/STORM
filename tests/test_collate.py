from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from storm_surgwmbench.data.collate import (
    collate_dense_variable_length,
    collate_frame_vae,
    collate_sparse_anchors,
    collate_window_sequences,
    direction_classes_from_delta,
)
from storm_surgwmbench.data.surgwmbench import SOURCE_TO_CODE, SurgWMBenchClipDataset, SurgWMBenchFrameDataset


def test_sparse_collate_shapes_and_actions(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        image_size=16,
        frame_sampling="sparse_anchors",
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_sparse_anchors)))

    assert batch["frames"].shape == (2, 20, 3, 16, 16)
    assert batch["coords_norm"].shape == (2, 20, 2)
    assert batch["actions_delta"].shape == (2, 19, 2)
    assert batch["actions_delta_dt"].shape == (2, 19, 3)
    assert batch["anchor_dt"].shape == (2, 19)
    assert batch["direction_classes"].shape == (2, 19)
    assert batch["magnitudes"].shape == (2, 19)
    assert batch["human_anchor_mask"].all()
    assert len(batch["metadata"]) == 2


def test_dense_collate_pads_masks_and_actions(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        image_size=12,
        frame_sampling="dense",
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_dense_variable_length)))

    assert batch["frames"].shape == (2, 31, 3, 12, 12)
    assert batch["frame_mask"][0].sum().item() == 25
    assert batch["frame_mask"][1].sum().item() == 31
    assert not batch["frame_mask"][0, 25:].any()
    assert batch["action_mask"][0].sum().item() == 24
    assert batch["action_mask"][1].sum().item() == 30
    assert batch["coord_source"][0, 0].item() == SOURCE_TO_CODE["human"]
    assert SOURCE_TO_CODE["interpolated"] in batch["coord_source"][0].tolist()
    assert batch["actions_delta_dt"].shape == (2, 30, 3)
    assert batch["direction_classes"].shape == (2, 30)


def test_window_collate_uses_dense_contract(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchClipDataset(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        image_size=12,
        frame_sampling="window",
        max_frames=7,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_window_sequences)))

    assert batch["frames"].shape == (2, 7, 3, 12, 12)
    assert batch["frame_mask"].all()
    assert batch["action_mask"].shape == (2, 6)


def test_direction_class_mapping_image_plane() -> None:
    deltas = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
        ]
    )

    assert direction_classes_from_delta(deltas).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_frame_vae_collate(toy_surgwmbench_root: Path) -> None:
    dataset = SurgWMBenchFrameDataset(toy_surgwmbench_root, "manifests/train.jsonl", image_size=10, max_frames_per_clip=2)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_frame_vae)))

    assert batch["image"].shape == (3, 3, 10, 10)
    assert len(batch["metadata"]) == 3
