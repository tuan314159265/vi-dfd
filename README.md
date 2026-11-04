# Vi-DFD

Diacritic-Factorized Distillation for Lightweight Vietnamese Scene Text Recognition.
BTL Deep Learning, HCMUT. Research proposal: PARSeq-S → PARSeq-Ti on Vietnamese word crops.

## Status

This is an initialized research scaffold. Configuration validation, Unicode decomposition,
fixed-seed utilities, and run preparation with CSV/commit metadata work. PARSeq training,
DFD tensor operations, dataset adapters, evaluation, ONNX benchmarking, and model serving
are **not implemented yet**. Prepared runs contain no accuracy or latency measurements.

## Setup and smoke check

Python 3.12+; no GPU is required for scaffold checks.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_scaffold.py
python train.py --config configs/runs/E2/ti_dfd_s42.yaml --check-config
```

Prepare an artifact directory after committing your code/config changes:

```bash
python train.py --config configs/runs/E2/ti_dfd_s42.yaml --prepare-run
```

`--prepare-run` writes `resolved.yaml`, `metadata.json`, `run.csv`, an empty
`metrics.csv` with headers, and `environment.txt` under `results/runs/<run_id>/`.
The recorded status is `prepared`. It never trains a model or invents scores.
A clean Git working tree is required so the full commit hash identifies the source.

Model dependencies in `requirements-train.in` are an **unlocked draft**, not a
validated GPU environment. Resolve and lock them when integrating a pinned PARSeq
revision. Scaffold dependencies are pinned in `requirements.txt`.

## Layout

```text
vi-dfd/
├── configs/                 # standalone YAML per planned run; manifest + tuning policy
├── data/                    # scripts only; no VinText images/crops
│   ├── crop_vintext.py
│   ├── make_splits.py
│   └── subsets.py
├── dfd/
│   ├── vietchar.py           # NFC decomposition/composition; actual vocabulary still required
│   ├── heads.py              # factorized head interface
│   ├── distill.py            # marginalization/DFD loss interfaces
│   ├── metrics.py            # shared evaluation interface
│   ├── config.py
│   └── runtime.py
├── train.py
├── eval.py
├── bench/                   # ONNX export, quantization and benchmark entry points
├── serve/                   # FastAPI entry point + CPU Dockerfile skeleton
├── results/                 # documented CSV schema; generated artifacts ignored by Git
├── tests/
├── scripts/
├── docs/                    # artifact coverage checklist, implementation tasks, proposal notes
├── requirements.txt
└── README.md
```

## Experiments and reproducibility

- Main student seeds: **42, 43, 44**; one-seed diagnostics use **42**.
- `configs/run_manifest.csv` enumerates planned jobs; no job is marked completed.
- Epochs, batch size, learning rate, vocabulary and dataset paths remain `null` until E1/data preparation.
- KD values in templates are candidates, not validation-selected hyperparameters.
- Teacher/student must receive identical augmentation, permutation and ground-truth context.
- Use one frozen teacher checkpoint for the main KD comparison.
- Each member prioritizes Colab Free + Kaggle. AWS is the fallback if compute/time is insufficient.
- Record actual GPU/runtime per run; measure latency centrally on the same machine/session.
- Do not concurrently append to a shared CSV from independent notebooks; collect per-run files first.

Detailed experiment notes are in [docs/proposal/00 - Tổng quan.md](docs/proposal/00%20-%20Tổng%20quan.md).
[Artifact checklist](docs/ARTIFACT_CHECKLIST.md) maps every path in the supplied tree.
[Implementation tasks](docs/IMPLEMENTATION.md) distinguish working utilities from pending research code.

## Data

Store datasets outside this repo and configure their paths. Do not commit VinText
images, word crops, pretrained weights or checkpoints. Verify dataset terms before use;
README/setup instructions should point users to the original source.

The count “229 characters + EOS” is a proposal assumption, not a fabricated built-in
vocabulary. Build and verify the real table from the frozen data protocol. EOS policy,
alignment denominators, heavy-word definition and tuning-budget fairness need final decisions.

## Attribution

- PARSeq: https://github.com/baudm/parseq — intended backbone; no upstream implementation
  is vendored in this scaffold. Preserve applicable LICENSE/NOTICE when integrating code.
- VietTDR: inherited factorized-head concept per the supplied proposal; repository URL
  and bibliographic details still need verification.
- Structural Knowledge Distillation: inherited marginal-distillation principle.
- The DFD adaptation and compression analysis are proposed contributions, pending evidence.

No project redistribution license has been chosen yet. Upstream code and datasets retain
their respective terms. See [THIRD_PARTY.md](THIRD_PARTY.md).
