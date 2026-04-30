"""Raw source-video and extracted-frame window datasets for SurgWMBench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from storm_surgwmbench.data.surgwmbench import (
    SurgWMBenchClipDataset,
    load_json,
    read_jsonl_manifest,
    resolve_dataset_path,
)
from storm_surgwmbench.data.transforms import pil_to_float_tensor, target_size_hw


class SurgWMBenchRawVideoDataset(Dataset):
    """Load fixed-length windows from source videos or extracted clip frames."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        source_video_manifest: str | Path | None = None,
        clip_length: int = 16,
        stride: int = 4,
        image_size: int | tuple[int, int] = 128,
        backend: str = "opencv",
        max_videos: int | None = None,
        max_clips_per_video: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.split = split
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.image_size = image_size
        self.backend = backend
        self.max_videos = max_videos
        self.max_clips_per_video = max_clips_per_video

        if self.clip_length <= 0:
            raise ValueError("clip_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if backend not in ("opencv", "clip_frames"):
            raise ValueError("backend must be 'opencv' or 'clip_frames'")

        if backend == "opencv":
            self.records = self._build_opencv_records(source_video_manifest)
        else:
            self.records = self._build_clip_frame_records()
        if not self.records:
            raise ValueError(f"No raw-video windows found for split={split!r} with backend={backend!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.backend == "opencv":
            frames = self._read_opencv_window(record)
        else:
            frames = torch.stack([self._load_frame_path(path) for path in record["frame_paths"]], dim=0)
        return {
            "frames": frames,
            "source_video_id": record["source_video_id"],
            "source_video_path": record["source_video_path"],
            "start_frame": int(record["start_frame"]),
            "frame_indices": torch.as_tensor(record["frame_indices"], dtype=torch.long),
        }

    def _build_opencv_records(self, source_video_manifest: str | Path | None) -> list[dict[str, Any]]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError("SurgWMBenchRawVideoDataset backend='opencv' requires opencv-python") from exc
        self._cv2 = cv2

        videos = self._source_video_records(source_video_manifest)
        if self.max_videos is not None:
            videos = videos[: max(int(self.max_videos), 0)]

        records: list[dict[str, Any]] = []
        for video in videos:
            path = resolve_dataset_path(self.dataset_root, video["source_video_path"])
            if path is None or not path.exists():
                raise FileNotFoundError(f"Source video not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"OpenCV could not open source video: {path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count < self.clip_length:
                continue
            starts = list(range(0, frame_count - self.clip_length + 1, self.stride))
            if self.max_clips_per_video is not None:
                starts = starts[: max(int(self.max_clips_per_video), 0)]
            for start in starts:
                records.append(
                    {
                        "source_video_id": str(video["source_video_id"]),
                        "source_video_path": str(video["source_video_path"]),
                        "start_frame": int(start),
                        "frame_indices": list(range(start, start + self.clip_length)),
                    }
                )
        return records

    def _source_video_records(self, source_video_manifest: str | Path | None) -> list[dict[str, Any]]:
        metadata_path = (
            resolve_dataset_path(self.dataset_root, source_video_manifest)
            if source_video_manifest is not None
            else self.dataset_root / "metadata" / "source_videos.json"
        )
        if metadata_path is None or not metadata_path.exists():
            raise FileNotFoundError(f"Source video manifest not found: {metadata_path}")

        payload = load_json(metadata_path)
        if isinstance(payload, dict) and isinstance(payload.get("videos"), dict):
            videos = list(payload["videos"].values())
        elif isinstance(payload, list):
            videos = payload
        else:
            raise ValueError(f"Unsupported source video manifest schema: {metadata_path}")

        split_ids = self._source_ids_for_split()
        filtered: list[dict[str, Any]] = []
        for video in videos:
            if not isinstance(video, dict):
                continue
            video_id = str(video.get("source_video_id", ""))
            if split_ids is not None and video_id not in split_ids:
                continue
            source_split = video.get("source_dataset_split")
            if split_ids is None and self.split != "all" and source_split is not None and source_split != self.split:
                continue
            if "source_video_path" not in video:
                continue
            filtered.append({"source_video_id": video_id, "source_video_path": str(video["source_video_path"])})
        return filtered

    def _source_ids_for_split(self) -> set[str] | None:
        if self.split == "all":
            return None
        manifest_path = self.dataset_root / "manifests" / f"{self.split}.jsonl"
        if not manifest_path.exists():
            return None
        return {str(row["source_video_id"]) for row in read_jsonl_manifest(manifest_path)}

    def _build_clip_frame_records(self) -> list[dict[str, Any]]:
        manifest = self.dataset_root / "manifests" / f"{self.split}.jsonl"
        if self.split == "all":
            manifest = self.dataset_root / "manifests" / "all.jsonl"
        dataset = SurgWMBenchClipDataset(
            dataset_root=self.dataset_root,
            manifest=manifest,
            frame_sampling="dense",
            image_size=self.image_size,
            return_images=False,
            use_dense_pseudo=False,
            strict=True,
        )
        records: list[dict[str, Any]] = []
        for clip_index, entry in enumerate(dataset.entries):
            annotation_path = dataset._annotation_path(entry)
            annotation = dataset._load_annotation(annotation_path)
            frame_records = dataset._frame_records(annotation)
            num_frames = int(entry.get("num_frames", annotation.get("num_frames", len(frame_records))))
            if num_frames < self.clip_length:
                continue
            starts = list(range(0, num_frames - self.clip_length + 1, self.stride))
            if self.max_clips_per_video is not None:
                starts = starts[: max(int(self.max_clips_per_video), 0)]
            for start in starts:
                indices = list(range(start, start + self.clip_length))
                paths = dataset._paths_for_indices(frame_records, indices)
                records.append(
                    {
                        "clip_index": clip_index,
                        "source_video_id": str(entry["source_video_id"]),
                        "source_video_path": str(entry["source_video_path"]),
                        "start_frame": int(start),
                        "frame_indices": indices,
                        "frame_paths": paths,
                    }
                )
        return records

    def _read_opencv_window(self, record: dict[str, Any]) -> torch.Tensor:
        cv2 = self._cv2
        video_path = resolve_dataset_path(self.dataset_root, record["source_video_path"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not open source video: {video_path}")
        frames: list[torch.Tensor] = []
        for frame_idx in record["frame_indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                cap.release()
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(self._array_to_tensor(frame_rgb))
        cap.release()
        return torch.stack(frames, dim=0)

    def _array_to_tensor(self, frame_rgb: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(frame_rgb)
        size_hw = target_size_hw(self.image_size)
        if size_hw is not None and (image.height, image.width) != size_hw:
            resampling = getattr(Image, "Resampling", Image)
            image = image.resize((size_hw[1], size_hw[0]), int(resampling.BILINEAR))
        return pil_to_float_tensor(image)

    def _load_frame_path(self, path: str | Path) -> torch.Tensor:
        from storm_surgwmbench.data.transforms import load_rgb_frame

        return load_rgb_frame(path, self.image_size)[0]


__all__ = ["SurgWMBenchRawVideoDataset"]
