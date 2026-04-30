"""Create a synthetic SurgWMBench dataset for tests and smoke checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

METHODS = ("linear", "pchip", "akima", "cubic_spline")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _sampled_indices(num_frames: int) -> list[int]:
    if num_frames < 20:
        raise ValueError("Toy SurgWMBench clips must have at least 20 frames")
    indices = np.linspace(0, num_frames - 1, 20).round().astype(int).tolist()
    indices[0] = 0
    indices[-1] = num_frames - 1
    for pos in range(1, len(indices)):
        if indices[pos] <= indices[pos - 1]:
            indices[pos] = indices[pos - 1] + 1
    for pos in range(len(indices) - 2, -1, -1):
        if indices[pos] >= indices[pos + 1]:
            indices[pos] = indices[pos + 1] - 1
    return [int(value) for value in indices]


def _coord_for_frame(frame_idx: int, num_frames: int, width: int, height: int, offset: float = 0.0) -> tuple[float, float]:
    denom = max(num_frames - 1, 1)
    x = 5.0 + (width - 10.0) * frame_idx / denom
    phase = (frame_idx + offset) / denom
    y = 8.0 + (height - 16.0) * (0.5 + 0.35 * np.sin(2.0 * np.pi * phase))
    return float(x), float(y)


def _try_write_toy_video(path: Path, width: int, height: int, num_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
    except ImportError:
        path.write_bytes(b"synthetic video placeholder")
        return

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    if not writer.isOpened():
        path.write_bytes(b"synthetic video placeholder")
        return
    for frame_idx in range(num_frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = 25
        image[:, :, 1] = np.uint8(20 + frame_idx % 120)
        image[:, :, 2] = 80
        writer.write(image)
    writer.release()


def create_toy_surgwmbench(root: str | Path, num_clips: int = 2) -> Path:
    """Create a final-layout toy SurgWMBench root and return it."""

    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)

    width, height = 64, 48
    source_video_id = "video_01"
    source_video_path = "videos/video_01/video_left.avi"
    _try_write_toy_video(root / source_video_path, width, height, num_frames=64)

    rows: list[dict[str, Any]] = []
    lengths = [25, 31, 28, 33]
    difficulties = ["low", "medium", "high", None]

    for clip_idx in range(num_clips):
        patient_id = source_video_id if clip_idx == 0 else f"video_{clip_idx + 1:02d}"
        trajectory_id = f"traj_{clip_idx + 1:03d}"
        num_frames = lengths[clip_idx % len(lengths)]
        difficulty = difficulties[clip_idx % len(difficulties)]
        frames_dir_rel = f"clips/{patient_id}/{trajectory_id}/frames"
        annotation_rel = f"clips/{patient_id}/{trajectory_id}/annotation.json"
        sampled = _sampled_indices(num_frames)
        sampled_set = set(sampled)

        frame_records: list[dict[str, Any]] = []
        for frame_idx in range(num_frames):
            x, y = _coord_for_frame(frame_idx, num_frames, width, height, offset=clip_idx)
            image = Image.new("RGB", (width, height), color=(20 + clip_idx * 20, 20, 35))
            draw = ImageDraw.Draw(image)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 80, 40))
            frame_rel = f"{frames_dir_rel}/{frame_idx:06d}.png"
            (root / frame_rel).parent.mkdir(parents=True, exist_ok=True)
            image.save(root / frame_rel)
            frame_records.append(
                {
                    "local_frame_idx": frame_idx,
                    "source_frame_idx": 1000 + frame_idx,
                    "frame_path": frame_rel,
                    "is_human_labeled": frame_idx in sampled_set,
                    "anchor_idx": sampled.index(frame_idx) if frame_idx in sampled_set else None,
                    "human_coord_px": None,
                    "human_coord_norm": None,
                    "coord_source": "human" if frame_idx in sampled_set else "unlabeled",
                }
            )

        human_anchors: list[dict[str, Any]] = []
        for anchor_idx, local_idx in enumerate(sampled):
            x, y = _coord_for_frame(local_idx, num_frames, width, height, offset=clip_idx)
            coord_px = [x, y]
            coord_norm = [x / width, y / height]
            human_anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "old_frame_idx": anchor_idx,
                    "local_frame_idx": int(local_idx),
                    "source_frame_idx": 1000 + int(local_idx),
                    "label_name": f"Label {anchor_idx + 1}",
                    "value": anchor_idx + 1,
                    "coord_px": coord_px,
                    "coord_norm": coord_norm,
                }
            )
            frame_records[local_idx]["human_coord_px"] = coord_px
            frame_records[local_idx]["human_coord_norm"] = coord_norm

        interpolation_files = {
            method: f"interpolations/{patient_id}/{trajectory_id}.{method}.json" for method in METHODS
        }
        anchor_by_frame = {anchor["local_frame_idx"]: anchor for anchor in human_anchors}
        for method_idx, method in enumerate(METHODS):
            coords: list[dict[str, Any]] = []
            for frame_idx in range(num_frames):
                if frame_idx in sampled_set:
                    anchor = anchor_by_frame[frame_idx]
                    coord_px = anchor["coord_px"]
                    coord_norm = anchor["coord_norm"]
                    source = "human"
                    anchor_idx = anchor["anchor_idx"]
                    confidence = 1.0
                    label_weight = 1.0
                else:
                    x, y = _coord_for_frame(frame_idx, num_frames, width, height, offset=clip_idx + method_idx * 0.35)
                    coord_px = [x, y]
                    coord_norm = [x / width, y / height]
                    source = "interpolated"
                    anchor_idx = None
                    confidence = 0.6
                    label_weight = 0.5
                coords.append(
                    {
                        "local_frame_idx": frame_idx,
                        "coord_px": coord_px,
                        "coord_norm": coord_norm,
                        "source": source,
                        "anchor_idx": anchor_idx,
                        "confidence": confidence,
                        "label_weight": label_weight,
                        "is_out_of_bounds": False,
                    }
                )
            _write_json(
                root / interpolation_files[method],
                {
                    "dataset_version": "SurgWMBench",
                    "patient_id": patient_id,
                    "trajectory_id": trajectory_id,
                    "interpolation_method": method,
                    "num_frames": num_frames,
                    "image_size": {"width": width, "height": height},
                    "coordinates": coords,
                },
            )

        annotation = {
            "dataset_version": "SurgWMBench",
            "patient_id": patient_id,
            "source_video_id": source_video_id,
            "source_video_path": source_video_path,
            "trajectory_id": trajectory_id,
            "difficulty": difficulty,
            "num_frames": num_frames,
            "image_size": {"width": width, "height": height},
            "coordinate_format": "pixel_xy",
            "coordinate_origin": "top_left",
            "num_human_anchors": 20,
            "sampled_indices": sampled,
            "available_interpolation_methods": list(METHODS),
            "default_interpolation_method": "linear",
            "frames": frame_records,
            "human_anchors": human_anchors,
            "interpolation_files": interpolation_files,
        }
        _write_json(root / annotation_rel, annotation)

        rows.append(
            {
                "dataset_version": "SurgWMBench",
                "patient_id": patient_id,
                "source_video_id": source_video_id,
                "source_video_path": source_video_path,
                "trajectory_id": trajectory_id,
                "difficulty": difficulty,
                "num_frames": num_frames,
                "annotation_path": annotation_rel,
                "frames_dir": frames_dir_rel,
                "interpolation_files": interpolation_files,
                "default_interpolation_method": "linear",
                "num_human_anchors": 20,
                "sampled_indices": sampled,
            }
        )

    for split in ("train", "val", "test", "all"):
        _write_jsonl(root / "manifests" / f"{split}.jsonl", rows)

    _write_json(
        root / "metadata" / "source_videos.json",
        {
            "dataset_version": "SurgWMBench",
            "number_of_source_videos": 1,
            "video_filename": "video_left.avi",
            "videos": {
                source_video_id: {
                    "source_video_id": source_video_id,
                    "source_video_path": source_video_path,
                    "filename": "video_left.avi",
                    "source_dataset_split": "train",
                    "size_bytes": (root / source_video_path).stat().st_size,
                }
            },
        },
    )
    _write_json(
        root / "metadata" / "validation_report.json",
        {"dataset_version": "SurgWMBench", "total_clips": len(rows), "error_count": 0, "warning_count": 0},
    )
    _write_json(
        root / "metadata" / "dataset_stats.json",
        {
            "dataset_version": "SurgWMBench",
            "number_of_clips": len(rows),
            "number_of_frames": sum(row["num_frames"] for row in rows),
            "number_of_human_anchors": 20 * len(rows),
        },
    )
    _write_json(root / "metadata" / "difficulty_rubric.json", {})
    _write_json(
        root / "metadata" / "interpolation_config.json",
        {"default_interpolation_method": "linear", "interpolation_methods": list(METHODS)},
    )
    (root / "README.md").write_text(
        "# SurgWMBench Toy Dataset\n\nSynthetic final-layout SurgWMBench fixture for loader tests.\n",
        encoding="utf-8",
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-clips", type=int, default=2)
    args = parser.parse_args()
    root = create_toy_surgwmbench(args.output, num_clips=args.num_clips)
    print(f"Created toy SurgWMBench dataset at {root}")


if __name__ == "__main__":
    main()
