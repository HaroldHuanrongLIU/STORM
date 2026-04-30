from __future__ import annotations

from pathlib import Path

import pytest

from tools.make_toy_surgwmbench import create_toy_surgwmbench


@pytest.fixture()
def toy_surgwmbench_root(tmp_path: Path) -> Path:
    return create_toy_surgwmbench(tmp_path / "SurgWMBench", num_clips=2)
