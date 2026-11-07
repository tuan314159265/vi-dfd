# Vi-DFD

**Diacritic-Factorized Distillation for Lightweight Vietnamese Scene Text Recognition**  
BTL Deep Learning, HCMUT. Research proposal: **PARSeq-S → PARSeq-Ti** on Vietnamese word crops.

## Status

This repository is an **initialized research scaffold**. Dataset exploration and preprocessing are conducted in **Jupyter notebooks**, separate from the research training scaffold.

- **Working scaffold:** configuration validation, Unicode decomposition helpers, fixed-seed utilities, tests, and run preparation with CSV/commit metadata.
- **Dataset overview — executed in a Jupyter notebook:** inspected VinText images, annotations and source splits; verified the counts recorded below.
- **Task 1 — Word cropping — implemented in a Jupyter notebook:** reads source image/annotation pairs, filters ignored labels, normalizes NFC, performs perspective cropping, and writes `crops/` and `metadata.csv` under `processed/vintext_notebook/`.
- **Task 2 — EDA — performed in a Jupyter notebook:** explores crop/annotation characteristics and Vietnamese text/diacritics. Specific plots/statistics should be cited from saved notebook outputs rather than assumed from this README.
- **Task 3 — Splitting and LMDB — implemented in a Jupyter notebook:** automatic SHA-256/pHash grouping, grouped train/val split, evaluation subsets, freeze manifests/checksums, and four-way LMDB export. **Completion of the LMDB export and read-back verification has not been confirmed from a run log here.**
- **Notebook implementation is separate from the `data/*.py` scaffold:** do not treat those Python files as the implementation of Tasks 1–3.
- **Not implemented yet:** PARSeq training, DFD tensor operations, complete dataset adapters, evaluation, ONNX benchmarking, and serving.

Prepared runs contain **no accuracy or latency measurements**. Dataset preprocessing and notebook code do not imply that training has been completed.

## Setup and smoke check

Python 3.12+; no GPU is required for scaffold checks.

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_scaffold.py
python train.py --config configs/runs/E2/ti_dfd_s42.yaml --check-config
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_scaffold.py
python train.py --config configs/runs/E2/ti_dfd_s42.yaml --check-config
```

For the dataset notebooks, install the additional dependencies in the **same Python/kernel environment** if they are not already installed:

```powershell
python -m pip install pandas numpy opencv-python Pillow lmdb ipykernel
```

Prepare an artifact directory **after committing** code/config changes:

```bash
python train.py --config configs/runs/E2/ti_dfd_s42.yaml --prepare-run
```

`--prepare-run` writes `resolved.yaml`, `metadata.json`, `run.csv`, an empty `metrics.csv` with headers, and `environment.txt` under `results/runs/<run_id>/`. The recorded status is `prepared`; it **never trains a model** or invents scores. A clean Git working tree is required so the full commit hash identifies the source.

Model dependencies in `requirements-train.in` are an **unlocked draft**, not a validated GPU environment. Resolve and lock them when integrating a pinned PARSeq revision. Scaffold dependencies are pinned in `requirements.txt`.

## Layout

```text
vi-dfd/
├── configs/                 # standalone YAML per planned run; manifest + tuning policy
├── data/                    # scaffold/module entry points; completed in notebook pipeline
│   ├── crop_vintext.py
│   ├── make_splits.py
│   └── subsets.py
├── dfd/
│   ├── vietchar.py          # NFC decomposition/composition; actual vocabulary still required
│   ├── heads.py             # factorized head interface
│   ├── distill.py           # marginalization/DFD loss interfaces
│   ├── metrics.py           # shared evaluation interface
│   ├── config.py
│   └── runtime.py
├── train.py
├── eval.py
├── bench/                   # ONNX export, quantization and benchmark entry points
├── serve/                   # FastAPI entry point + CPU Dockerfile skeleton
├── results/                 # documented CSV schema; generated artifacts ignored by Git
├── tests/
├── scripts/                 # scaffold checks and other utilities
├── docs/                    # artifact checklist, implementation tasks, proposal notes
├── notebooks/
    ├──01_overview_dataset.ipynb
    ├──02_crop_vintext.ipynb
    ├──03_EDA.ipynb
    └──04_dataset_splitting.ipynb
├── requirements.txt
└── README.md
```

The **overview, word-crop preprocessing, EDA, and dataset splitting are all carried out in Jupyter notebooks**. `notebooks/` above is the recommended location in the repository; move or commit the actual `.ipynb` files there and use their real filenames. The existing `data/crop_vintext.py`, `data/make_splits.py`, and `data/subsets.py` paths belong to the scaffold and must not be documented as the notebooks' execution commands.

## VinText data preparation

### Running the dataset notebooks

1. Open each actual `.ipynb` file in VS Code or Jupyter and select the same `.venv` kernel.
2. Run in order: **overview → word cropping → EDA → splitting/LMDB**.
3. Use the raw `dataset/` only as an input. Notebook-generated crops, metadata, split CSVs, reports, and LMDBs are saved under `processed/vintext_notebook/`.
4. Confirm the splitting notebook's final LMDB read-back outputs before marking the export complete.

### Local workspace (Windows)

Keep raw and processed datasets **outside** the Git repository:

```text
your_local
├── dataset/                         # original VinText; do not modify
│   ├── general_dict.txt
│   ├── vn_dictionary.txt
│   ├── labels/                     # gt_<image_id>.txt
│   ├── train_images/
│   ├── test_image/
│   └── unseen_test_images/
├── processed/
│   └── vintext_notebook/
│       ├── crops/                  # Task 1 word crops
│       ├── metadata.csv            # Task 1 word-level metadata
│       ├── crop_issues.csv         # Task 1, if generated
│       ├── splits/                 # Task 3 manifests and reports
│       │   ├── duplicates/
│       │   ├── manifests/
│       │   ├── subsets/
│       │   └── freeze_manifest.json
│       └── lmdb/                   # generated only after export
│           ├── train/
│           ├── val/
│           ├── test/
│           └── unseen_test/
└── vi-dfd/                          # Git repository
```

The folder tree under `splits/` describes the **organized notebook outputs**. A directory/file is present only after the corresponding notebook cell has run successfully.

### Dataset overview — completed in a notebook

Run the **overview notebook** with the project `.venv` selected as the Jupyter kernel. Results already observed:

| Raw split | Scene images |
|---|---:|
| Train (`train_images`) | 1,200 |
| Test (`test_image`) | 300 |
| Unseen test (`unseen_test_images`) | 500 |
| **Total** | **2,000** |

| Annotation statistic | Count |
|---|---:|
| Label files (`labels/*.txt`) | 2,000 |
| Rows with 9 comma-separated fields | 56,084 |
| Non-empty text transcripts, excluding `###` | 43,045 |
| Ignored transcripts (`###`) | 12,984 |
| Malformed rows reported by overview | 0 |
| Difference not counted as text or `###` | **55** |

The difference of **55 rows** is consistent with empty transcripts under the overview script's counting logic; confirm during preprocessing rather than labeling all 56,084 rows usable. These counts are **dataset overview statistics**, not successful crop or LMDB counts.

The annotation format is one word/text region per line:

```text
x1,y1,x2,y2,x3,y3,x4,y4,transcript
```

Example:

```text
94,10,117,10,117,41,93,41,CHẤT
```

Use `split(",", maxsplit=8)` because transcripts may contain commas. `gt_1.txt` corresponds to `im0001.jpg` by image ID; validate matches and coordinates as part of preprocessing.

### Task 1 — Word-crop preprocessing (notebook)

The **word-crop notebook** implements this pipeline:

1. Read image/annotation pairs by `image_id`.
2. Skip `###` and empty transcripts; record malformed/invalid instances.
3. Normalize text to Unicode **NFC**.
4. Apply quadrilateral perspective warp, retaining approximately **10% vertical margin**.
5. Save individual word crops and `metadata.csv` with `source_split`, `image_id`, crop path, transcript, and original word height (the notebook currently uses the column name `split` for the original split).

The current local output root for downstream work is:

```text
D:/BTL_DL/processed/vintext_notebook/
```

Do **not** resize all crops to a fixed PARSeq input size at this stage unless the training data-loader protocol calls for it. Verify crop quality and the recorded error log before exporting final LMDBs.

### Task 2 — EDA and data quality (notebook)

The **EDA notebook** covers analysis of:

- Character and Vietnamese diacritic-combination frequency analysis.
- Rare-combination frequency calculated **from train only**.
- Small-crop distribution and crop-quality checks.
- Original-scene exact and perceptual duplicate detection.

EDA results should be taken from the actual notebook cell outputs and figures; this README does not invent frequencies, charts, or quality metrics.

### Task 3 — Grouped splitting, frozen subsets, LMDB

The **splitting/LMDB notebook** follows this sequence:

1. **Input validation:** read `metadata.csv`, check crop paths and original source splits.
2. **Duplicate detection:** compute SHA-256 and 64-bit pHash on original **scene images**, not word crops.
3. **Automatic grouping:** group byte-identical images and pairs with pHash Hamming distance **≤ 4**; report distance **5–8** pairs without grouping. These are heuristic thresholds, not proof of identity.
4. **Train/validation split:** split the original 1,200 train scene images approximately **90/10**, seeded with **42**, keeping confirmed/automatically linked groups together; never split word crops independently.
5. **Official evaluations:** keep the 300 test and 500 unseen-test source assignments unchanged; report cross-official-split duplicates rather than silently reassigning them.
6. **Frozen evaluation subsets:** for val/test/unseen test, create `diacritic_heavy` (≥2 marked characters), `rare_combinations` (train frequency <20), and `small_crops` (original word height <16 px).
7. **Freeze manifest:** write split counts, configuration, source metadata hash, and CSV checksums to `splits/freeze_manifest.json`.
8. **LMDB export and verification:** write `train`, `val`, `test`, `unseen_test` LMDBs and read back sample count, first image, and label.

**Duplicate reports:**

```text
splits/duplicates/
├── source_image_hashes.csv
├── sha256_exact_pairs.csv
├── phash_auto_pairs.csv
├── phash_report_pairs.csv
├── duplicate_groups.csv
└── cross_official_split_duplicates.csv
```

**Split manifests and evaluation subsets:**

```text
splits/manifests/
├── train.csv
├── val.csv
├── test.csv
├── unseen_test.csv
└── image_split_assignments.csv

splits/subsets/
├── train_diacritic_combination_counts.csv
├── val_diacritic_heavy.csv
├── val_rare_combinations.csv
├── val_small_crops.csv
├── test_diacritic_heavy.csv
├── test_rare_combinations.csv
├── test_small_crops.csv
├── unseen_test_diacritic_heavy.csv
├── unseen_test_rare_combinations.csv
└── unseen_test_small_crops.csv
```

One pHash candidate pair was reported within train (`im0478`–`im0593`) and another crossed train/unseen test (`im0894`–`im1670`), both at distance 4. The cross-split pair is a **potential evaluation leakage risk** and must remain visible in experiment reporting. Source test/unseen-test images are not moved automatically.

#### LMDB key schema

Each split has an independent LMDB (typically `data.mdb` and `lock.mdb`):

| Key | Value |
|---|---|
| `num-samples` | Number of samples in this LMDB |
| `image-000000001` | Word-crop image bytes (PNG) |
| `label-000000001` | NFC transcript encoded as UTF-8 |
| `source-id-000000001` | `image_id:line_number` for tracing |

LMDB is a local key-value database; **no MySQL/PostgreSQL server is needed**. The PARSeq dataset adapter must use this exact schema or be updated together with the exporter. Existing LMDB directories are not overwritten automatically; move/backup them before a rebuild.

### Dataset reproducibility rules

- Keep the raw `dataset/` unchanged and source-image train/test/unseen assignments intact.
- Split by scene image and duplicate group, never by individual word crop.
- Fix seed and grouping thresholds; commit the small CSV/JSON manifests if redistribution terms permit, but not image data.
- Verify frozen split/subset checksums after any regeneration; checksum of manifest CSVs alone does **not** guarantee that image bytes are identical.
- Do not treat potential cross-split visual duplicates as a clean unseen benchmark without reporting the limitation.

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

## Data and dataset terms

Store datasets outside this repo and configure their paths. Do not commit VinText images, word crops, pretrained weights or checkpoints. Verify dataset terms before use; README/setup instructions should point users to the original source.

The count **“229 characters + EOS”** is a proposal assumption, not a fabricated built-in vocabulary. Build and verify the real table from the frozen data protocol. EOS policy, alignment denominators, heavy-word definition, and tuning-budget fairness need final decisions.

## Attribution

- PARSeq: <https://github.com/baudm/parseq> — intended backbone; no upstream implementation is vendored in this scaffold. Preserve applicable LICENSE/NOTICE when integrating code.
- VietTDR: inherited factorized-head concept per the supplied proposal; repository URL and bibliographic details still need verification.
- Structural Knowledge Distillation: inherited marginal-distillation principle.
- The DFD adaptation and compression analysis are proposed contributions, pending evidence.

No project redistribution license has been chosen yet. Upstream code and datasets retain their respective terms. See [THIRD_PARTY.md](THIRD_PARTY.md).
