"""Read-only SurgWMBench dataset validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from storm_surgwmbench.data.surgwmbench import (
    DATASET_VERSION,
    INTERPOLATION_METHODS,
    _coord_from_item,
    _frame_local_index,
    _frame_path_value,
    _parse_image_size,
    load_json,
    read_jsonl_manifest,
    resolve_dataset_path,
)


@dataclass(frozen=True)
class ValidationSummary:
    dataset_root: str
    manifest: str
    checked_entries: int
    checked_frames: int
    checked_human_anchors: int
    checked_interpolation_coordinates: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def _interpolation_files(entry: dict[str, Any], annotation: dict[str, Any]) -> dict[str, str]:
    files = entry.get("interpolation_files") or annotation.get("interpolation_files")
    if not isinstance(files, dict):
        raise ValueError("interpolation_files must be a mapping")
    return {str(key): str(value) for key, value in files.items()}


def _frame_path(dataset_root: Path, frame: Any, fallback_idx: int) -> Path | None:
    value = _frame_path_value(frame)
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else dataset_root / candidate


def _validate_interpolation(
    *,
    interpolation_path: Path,
    method: str,
    num_frames: int,
    image_size_hw: tuple[int, int],
    human_anchors: list[dict[str, Any]],
    errors: list[str],
) -> int:
    try:
        interpolation = load_json(interpolation_path)
    except Exception as exc:
        _add_error(errors, f"{interpolation_path}: cannot read interpolation file: {exc}")
        return 0
    if not isinstance(interpolation, dict):
        _add_error(errors, f"{interpolation_path}: interpolation JSON is not an object")
        return 0
    if interpolation.get("dataset_version") != DATASET_VERSION:
        _add_error(errors, f"{interpolation_path}: dataset_version is not {DATASET_VERSION!r}")
    if interpolation.get("interpolation_method") not in (None, method):
        _add_error(errors, f"{interpolation_path}: interpolation_method does not match {method!r}")
    coordinates = interpolation.get("coordinates")
    if not isinstance(coordinates, list):
        _add_error(errors, f"{interpolation_path}: missing coordinates[]")
        return 0
    if len(coordinates) != num_frames:
        _add_error(errors, f"{interpolation_path}: method {method} has {len(coordinates)} coordinates, expected {num_frames}")

    by_local_idx: dict[int, dict[str, Any]] = {}
    for fallback_idx, coord in enumerate(coordinates):
        if not isinstance(coord, dict):
            _add_error(errors, f"{interpolation_path}: coordinate {fallback_idx} is not an object")
            continue
        local_idx = int(coord.get("local_frame_idx", fallback_idx))
        if local_idx in by_local_idx:
            _add_error(errors, f"{interpolation_path}: duplicate local_frame_idx={local_idx}")
        by_local_idx[local_idx] = coord

    missing = sorted(set(range(num_frames)) - set(by_local_idx))
    if missing:
        _add_error(errors, f"{interpolation_path}: missing local_frame_idx values: {missing[:5]}")

    anchor_indices = {int(anchor.get("local_frame_idx", pos)) for pos, anchor in enumerate(human_anchors)}
    for frame_idx, coord in by_local_idx.items():
        source = str(coord.get("source")).lower()
        confidence = float(coord.get("confidence", 0.0))
        label_weight = float(coord.get("label_weight", 0.0))
        if frame_idx in anchor_indices:
            if source != "human":
                _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={frame_idx} source is not 'human'")
            if not np.isclose(confidence, 1.0):
                _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={frame_idx} confidence is not 1.0")
            if not np.isclose(label_weight, 1.0):
                _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={frame_idx} label_weight is not 1.0")
        else:
            if source != "interpolated":
                _add_error(errors, f"{interpolation_path}: non-anchor local_frame_idx={frame_idx} source is not 'interpolated'")
            if not np.isclose(confidence, 0.6):
                _add_error(errors, f"{interpolation_path}: non-anchor local_frame_idx={frame_idx} confidence is not 0.6")
            if not np.isclose(label_weight, 0.5):
                _add_error(errors, f"{interpolation_path}: non-anchor local_frame_idx={frame_idx} label_weight is not 0.5")

    for anchor_pos, anchor in enumerate(human_anchors):
        local_idx = int(anchor.get("local_frame_idx", anchor_pos))
        coord = by_local_idx.get(local_idx)
        if coord is None:
            _add_error(errors, f"{interpolation_path}: missing coordinate for human anchor local_frame_idx={local_idx}")
            continue
        anchor_px, _ = _coord_from_item(anchor, image_size_hw)
        coord_px, _ = _coord_from_item(coord, image_size_hw)
        if not np.allclose(np.asarray(anchor_px), np.asarray(coord_px), atol=1e-5):
            _add_error(errors, f"{interpolation_path}: anchor local_frame_idx={local_idx} coordinate differs from human label")

    return len(coordinates)


def validate_surgwmbench(
    dataset_root: str | Path,
    manifest: str | Path,
    interpolation_method: str | None = None,
    check_files: bool = False,
    num_samples: int | None = None,
) -> ValidationSummary:
    dataset_root = Path(dataset_root).expanduser()
    manifest_path = Path(manifest).expanduser()
    manifest_path = manifest_path if manifest_path.is_absolute() else dataset_root / manifest_path
    errors: list[str] = []
    checked_frames = 0
    checked_anchors = 0
    checked_interpolation_coordinates = 0

    try:
        entries = read_jsonl_manifest(manifest_path)
    except Exception as exc:
        return ValidationSummary(str(dataset_root), str(manifest_path), 0, 0, 0, 0, [f"{manifest_path}: cannot read manifest: {exc}"])
    if num_samples is not None:
        entries = entries[: max(int(num_samples), 0)]

    for entry_idx, entry in enumerate(entries):
        prefix = f"manifest entry {entry_idx}"
        if entry.get("dataset_version") != DATASET_VERSION:
            _add_error(errors, f"{prefix}: dataset_version={entry.get('dataset_version')!r}, expected {DATASET_VERSION!r}")
        if int(entry.get("num_human_anchors", -1)) != 20:
            _add_error(errors, f"{prefix}: num_human_anchors is not 20")

        sampled = entry.get("sampled_indices")
        if not isinstance(sampled, list) or len(sampled) != 20:
            _add_error(errors, f"{prefix}: sampled_indices must contain 20 values")
        elif list(map(int, sampled)) != sorted(map(int, sampled)):
            _add_error(errors, f"{prefix}: sampled_indices are not sorted")

        annotation_path = resolve_dataset_path(dataset_root, entry.get("annotation_path"))
        if annotation_path is None:
            _add_error(errors, f"{prefix}: missing annotation_path")
            continue
        if not annotation_path.exists():
            _add_error(errors, f"{prefix}: annotation not found: {annotation_path}")
            continue

        try:
            annotation = load_json(annotation_path)
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: cannot read annotation: {exc}")
            continue
        if not isinstance(annotation, dict):
            _add_error(errors, f"{annotation_path}: annotation is not an object")
            continue
        if annotation.get("dataset_version", entry.get("dataset_version")) != DATASET_VERSION:
            _add_error(errors, f"{annotation_path}: dataset_version is not {DATASET_VERSION!r}")

        frames = annotation.get("frames")
        if not isinstance(frames, list):
            _add_error(errors, f"{annotation_path}: missing frames[]")
            frames = []
        human_anchors = annotation.get("human_anchors")
        if not isinstance(human_anchors, list):
            _add_error(errors, f"{annotation_path}: missing human_anchors[]")
            human_anchors = []
        if len(human_anchors) != 20:
            _add_error(errors, f"{annotation_path}: expected 20 human anchors, got {len(human_anchors)}")
        checked_anchors += len(human_anchors)

        annotation_sampled = annotation.get("sampled_indices", entry.get("sampled_indices"))
        if not isinstance(annotation_sampled, list):
            _add_error(errors, f"{annotation_path}: sampled_indices is not a list")
            annotation_sampled = []
        if len(annotation_sampled) != 20:
            _add_error(errors, f"{annotation_path}: expected 20 sampled_indices, got {len(annotation_sampled)}")
        if isinstance(entry.get("sampled_indices"), list) and list(map(int, entry["sampled_indices"])) != list(
            map(int, annotation_sampled)
        ):
            _add_error(errors, f"{annotation_path}: sampled_indices differ from manifest")

        try:
            image_size_hw = _parse_image_size(annotation.get("image_size"))
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: invalid image_size: {exc}")
            image_size_hw = (1, 1)

        num_frames = int(entry.get("num_frames", annotation.get("num_frames", len(frames))))
        checked_frames += num_frames
        if frames and len(frames) != num_frames:
            _add_error(errors, f"{annotation_path}: frames[] length {len(frames)} does not match num_frames={num_frames}")

        if check_files:
            source_video_path = resolve_dataset_path(dataset_root, annotation.get("source_video_path", entry.get("source_video_path")))
            if source_video_path is None or not source_video_path.exists():
                _add_error(errors, f"{prefix}: source video not found: {source_video_path}")
            frames_dir = resolve_dataset_path(dataset_root, entry.get("frames_dir"))
            if frames_dir is None or not frames_dir.exists():
                _add_error(errors, f"{prefix}: frames_dir not found: {frames_dir}")
            for fallback_idx, frame in enumerate(frames):
                path = _frame_path(dataset_root, frame, fallback_idx)
                if path is None:
                    _add_error(errors, f"{annotation_path}: frame {fallback_idx} has no frame_path")
                elif not path.exists():
                    _add_error(errors, f"{annotation_path}: frame not found: {path}")

        try:
            files = _interpolation_files(entry, annotation)
        except Exception as exc:
            _add_error(errors, f"{annotation_path}: invalid interpolation_files: {exc}")
            continue
        selected_method = interpolation_method or entry.get("default_interpolation_method") or annotation.get(
            "default_interpolation_method"
        ) or "linear"
        if selected_method not in files:
            _add_error(errors, f"{annotation_path}: selected interpolation method {selected_method!r} not listed")
        for method in INTERPOLATION_METHODS:
            if method not in files:
                _add_error(errors, f"{annotation_path}: interpolation method {method!r} not listed")
                continue
            path = resolve_dataset_path(dataset_root, files[method])
            if path is None or not path.exists():
                _add_error(errors, f"{annotation_path}: interpolation file not found for {method}: {path}")
                continue
            checked_interpolation_coordinates += _validate_interpolation(
                interpolation_path=path,
                method=method,
                num_frames=num_frames,
                image_size_hw=image_size_hw,
                human_anchors=human_anchors,
                errors=errors,
            )

    return ValidationSummary(
        dataset_root=str(dataset_root),
        manifest=str(manifest_path),
        checked_entries=len(entries),
        checked_frames=checked_frames,
        checked_human_anchors=checked_anchors,
        checked_interpolation_coordinates=checked_interpolation_coordinates,
        errors=errors,
    )


__all__ = ["ValidationSummary", "validate_surgwmbench"]
