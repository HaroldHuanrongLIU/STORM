from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from storm_surgwmbench.data.validate_loader import validate_surgwmbench


def test_toy_dataset_generator_and_validator(toy_surgwmbench_root: Path) -> None:
    summary = validate_surgwmbench(
        toy_surgwmbench_root,
        "manifests/train.jsonl",
        interpolation_method="linear",
        check_files=True,
    )

    assert summary.ok
    assert summary.checked_entries == 2
    assert summary.checked_human_anchors == 40
    assert summary.checked_frames == 56
    assert summary.checked_interpolation_coordinates == 4 * 56


def test_loader_sanity_command_passes_on_toy_data(toy_surgwmbench_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.validate_surgwmbench_loader",
            "--dataset-root",
            str(toy_surgwmbench_root),
            "--manifest",
            "manifests/train.jsonl",
            "--interpolation-method",
            "linear",
            "--check-files",
            "--num-samples",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "SurgWMBench validation passed." in result.stdout
    assert "checked_entries: 2" in result.stdout
