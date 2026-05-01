# Repository Guidelines

## Project Structure & Module Organization

This repository contains the original STORM Atari world-model implementation plus a SurgWMBench extension.

- `train.py`, `eval.py`, `agents.py`, `replay_buffer.py`, and `env_wrapper.py` are the original Atari-oriented training, evaluation, agent, replay, and environment code.
- `sub_models/` contains the STORM world model, Transformer blocks, attention utilities, and losses.
- `config_files/STORM.yaml` is the original STORM config.
- `storm_surgwmbench/` contains the SurgWMBench data-layer extension. Keep new surgical benchmark code here unless it must integrate with original STORM modules.
- `tools/` contains CLI utilities such as toy dataset generation and loader validation.
- `tests/` contains pytest coverage for the SurgWMBench data layer.
- `results/` stores example result artifacts.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run original STORM training/evaluation:

```bash
./train.sh
./eval.sh
```

Run SurgWMBench tests:

```bash
python -m pytest -q tests
```

Validate the local SurgWMBench dataset:

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 8
```

## Coding Style & Naming Conventions

Use Python 3.10+ style with type hints for new code. Follow existing PyTorch patterns, but avoid adding new hardcoded `.cuda()` calls in SurgWMBench code. Use `pathlib.Path` for paths and keep dataset paths manifest-relative. Prefer clear snake_case names for functions, variables, files, and test modules.

Do not modify original STORM files for SurgWMBench work unless integration requires it. Keep benchmark-specific additions under `storm_surgwmbench/`.

## Testing Guidelines

Tests use `pytest`; `pytest.ini` sets `pythonpath = .`. Name tests `test_*.py` and keep synthetic fixtures under `tmp_path`, not the repository tree. Data-layer changes should cover sparse anchors, dense variable-length clips, collators, metrics, and validation failures.

Run focused tests before committing:

```bash
python -m pytest -q tests/test_surgwmbench_dataset.py tests/test_collate.py tests/test_metrics.py
```

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, for example `Add SurgWMBench data layer` and `Update README.md`. Keep commits focused and avoid mixing generated cache files with source changes.

Pull requests should include a concise summary, test commands run, dataset assumptions, and any limitations. For visual or rollout changes, include saved examples or screenshots when relevant.

## Agent-Specific Instructions

Treat `/mnt/hdd1/neurips2026_dataset_track/SurgWMBench/README.md` as the canonical dataset contract when available. Do not create random train/val/test splits, infer frame extensions, clip coordinates silently, or report dense pseudo-coordinate metrics as human-ground-truth metrics.
