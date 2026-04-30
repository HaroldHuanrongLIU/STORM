"""Collators for SurgWMBench datasets."""

from __future__ import annotations

from typing import Any

import torch

STAY_THRESHOLD = 0.002


def _sample_metadata(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "patient_id",
        "source_video_id",
        "source_video_path",
        "trajectory_id",
        "difficulty",
        "num_frames",
        "image_size_original",
        "annotation_path",
        "interpolation_path",
        "interpolation_method",
    )
    return {key: item.get(key) for key in keys}


def _require_frames(item: dict[str, Any]) -> torch.Tensor:
    frames = item.get("frames")
    if frames is None:
        raise ValueError("Collators require samples created with return_images=True")
    return frames


def direction_classes_from_delta(dxdy: torch.Tensor, stay_threshold: float = STAY_THRESHOLD) -> torch.Tensor:
    """Map ``[...,2]`` normalized deltas to 0=stay, 1..8 image-plane compass bins."""

    if dxdy.shape[-1] != 2:
        raise ValueError(f"dxdy must end in dimension 2, got {tuple(dxdy.shape)}")
    magnitude = torch.linalg.norm(dxdy, dim=-1)
    angle = torch.atan2(dxdy[..., 1], dxdy[..., 0])
    bins = torch.remainder(torch.round(angle / (torch.pi / 4.0)).to(torch.long), 8) + 1
    return torch.where(magnitude <= stay_threshold, torch.zeros_like(bins), bins)


def _actions_from_coords(
    coords_norm: torch.Tensor,
    frame_indices: torch.Tensor,
    num_frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions_delta = coords_norm[:, 1:] - coords_norm[:, :-1]
    denom = torch.clamp(num_frames.to(torch.float32) - 1.0, min=1.0).unsqueeze(1)
    dt = (frame_indices[:, 1:] - frame_indices[:, :-1]).to(torch.float32) / denom
    actions_delta_dt = torch.cat([actions_delta, dt.unsqueeze(-1)], dim=-1)
    direction_classes = direction_classes_from_delta(actions_delta)
    magnitudes = torch.linalg.norm(actions_delta, dim=-1)
    return actions_delta, actions_delta_dt, direction_classes, magnitudes


def collate_sparse_anchors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate ``frame_sampling='sparse_anchors'`` samples."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    frames = torch.stack([_require_frames(item) for item in batch], dim=0)
    coords_norm = torch.stack([item["selected_coords_norm"] for item in batch], dim=0)
    coords_px = torch.stack([item["selected_coords_px"] for item in batch], dim=0)
    sampled_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    frame_indices = torch.stack([item["frame_indices"] for item in batch], dim=0)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    if frames.shape[1] != 20 or coords_norm.shape[1] != 20 or sampled_indices.shape[1] != 20:
        raise ValueError("Sparse SurgWMBench batches must contain exactly 20 anchor timesteps")

    actions_delta, actions_delta_dt, direction_classes, magnitudes = _actions_from_coords(
        coords_norm, sampled_indices, num_frames
    )
    anchor_dt = actions_delta_dt[..., 2]
    metadata = [_sample_metadata(item) for item in batch]

    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "sampled_indices": sampled_indices,
        "frame_indices": frame_indices,
        "human_anchor_mask": torch.ones(coords_norm.shape[:2], dtype=torch.bool),
        "num_frames": num_frames,
        "anchor_dt": anchor_dt,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "direction_classes": direction_classes,
        "magnitudes": magnitudes,
        "coord_source": torch.stack([item["selected_coord_sources"] for item in batch], dim=0),
        "label_weight": torch.stack([item["selected_label_weights"] for item in batch], dim=0),
        "difficulty": [item["difficulty"] for item in batch],
        "metadata": metadata,
    }


def collate_dense_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate dense variable-length or windowed samples with padding."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(batch)
    max_t = max(int(_require_frames(item).shape[0]) for item in batch)
    channels, height, width = _require_frames(batch[0]).shape[1:]

    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    coords_norm = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    coords_px = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    coord_source = torch.zeros(batch_size, max_t, dtype=torch.long)
    label_weight = torch.zeros(batch_size, max_t, dtype=torch.float32)
    confidence = torch.zeros(batch_size, max_t, dtype=torch.float32)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    action_t = max(max_t - 1, 0)
    actions_delta = torch.zeros(batch_size, action_t, 2, dtype=torch.float32)
    actions_delta_dt = torch.zeros(batch_size, action_t, 3, dtype=torch.float32)
    action_mask = torch.zeros(batch_size, action_t, dtype=torch.bool)
    direction_classes = torch.zeros(batch_size, action_t, dtype=torch.long)
    magnitudes = torch.zeros(batch_size, action_t, dtype=torch.float32)

    for row, item in enumerate(batch):
        item_frames = _require_frames(item)
        t = int(item_frames.shape[0])
        frames[row, :t] = item_frames
        coords_norm[row, :t] = item["selected_coords_norm"]
        coords_px[row, :t] = item["selected_coords_px"]
        frame_mask[row, :t] = True
        coord_source[row, :t] = item["selected_coord_sources"]
        label_weight[row, :t] = item["selected_label_weights"]
        confidence[row, :t] = item["selected_confidence"]
        frame_indices[row, :t] = item["frame_indices"]

        if t > 1:
            local_delta = item["selected_coords_norm"][1:] - item["selected_coords_norm"][:-1]
            denom = max(float(item["num_frames"] - 1), 1.0)
            dt = (item["frame_indices"][1:] - item["frame_indices"][:-1]).to(torch.float32) / denom
            actions_delta[row, : t - 1] = local_delta
            actions_delta_dt[row, : t - 1] = torch.cat([local_delta, dt.unsqueeze(-1)], dim=-1)
            action_mask[row, : t - 1] = True
            direction_classes[row, : t - 1] = direction_classes_from_delta(local_delta)
            magnitudes[row, : t - 1] = torch.linalg.norm(local_delta, dim=-1)

    metadata = [_sample_metadata(item) for item in batch]
    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "frame_mask": frame_mask,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "action_mask": action_mask,
        "direction_classes": direction_classes,
        "magnitudes": magnitudes,
        "num_frames": num_frames,
        "difficulty": [item["difficulty"] for item in batch],
        "metadata": metadata,
    }


def collate_window_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate window samples; identical padding contract to dense batches."""

    return collate_dense_variable_length(batch)


def collate_frame_vae(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """Collate frame-level samples for VAE pretraining."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    images, metadata = zip(*batch)
    return {
        "image": torch.stack(list(images), dim=0),
        "metadata": [dict(item) for item in metadata],
    }


__all__ = [
    "collate_dense_variable_length",
    "collate_frame_vae",
    "collate_sparse_anchors",
    "collate_window_sequences",
    "direction_classes_from_delta",
]
