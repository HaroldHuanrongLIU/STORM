"""Validate SurgWMBench manifests and loader-critical invariants."""

from __future__ import annotations

import argparse
from pathlib import Path

from storm_surgwmbench.data.validate_loader import validate_surgwmbench


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--interpolation-method", default=None)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    summary = validate_surgwmbench(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        interpolation_method=args.interpolation_method,
        check_files=args.check_files,
        num_samples=args.num_samples,
    )
    if summary.errors:
        print(f"SurgWMBench validation failed with {len(summary.errors)} error(s).")
        for error in summary.errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("SurgWMBench validation passed.")
    print(f"manifest: {summary.manifest}")
    print(f"checked_entries: {summary.checked_entries}")
    print(f"checked_frames: {summary.checked_frames}")
    print(f"checked_human_anchors: {summary.checked_human_anchors}")
    print(f"checked_interpolation_coordinates: {summary.checked_interpolation_coordinates}")


if __name__ == "__main__":
    main()
