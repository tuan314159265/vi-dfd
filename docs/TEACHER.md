# Task 9 — PARSeq-S teacher fine-tuning

## Inputs and exact model

Use the original VinText archive from the authors' repository:
https://github.com/VinAIResearch/dict-guided#dataset . The user downloaded
`vietnamese_original.zip`; no images or label manifests are committed here.

The teacher is official PARSeq `parseq`, not `parseq-tiny` and not the patch16/224
variant. Pin upstream to `1902db043c029a7e03a3818c616c06600af574be`.
It has width 384, encoder 12 layers/6 heads, decoder 1 layer/12 heads, patches 4x8,
and input 32x128. Load English `parseq-bb5792a6.pt` with its hash checked.

The Vietnamese vocabulary has **229 output classes = EOS + 228 characters**;
the tokenizer contains 231 tokens including BOS and PAD. Reset the output head.
Load all compatible encoder/decoder weights; remap 97 English input embeddings
by token identity, including EOS/BOS/PAD. Newly added Vietnamese embeddings retain
their seeded initialization. Initialization metadata lists loaded/reset keys and
the full checkpoint SHA-256.

## Rebuild the team's data locally

```bash
python scripts/prepare_teacher_data.py \
  --archive ~/Downloads/vietnamese_original.zip \
  --workspace ../datasets/vintext
```

This executes the team's existing notebook 02 and 04 code, overriding only paths
and suppressing image displays. It keeps margin 10%, split seed 42, pHash grouping,
training-only rare frequencies and original tone placement. Outputs remain outside
the public repository. A second invocation verifies and reuses frozen data.

Observed counts from this local rebuild on 2026-11-09:

| Split | LMDB samples before training-only filtering |
|---|---:|
| Train | 23,269 |
| Validation | 2,507 |
| Test | 7,201 |
| Unseen test | 10,068 |

The generated `preparation.json` records archive, notebook-source and LMDB hashes.
`splits/freeze_manifest.json` freezes image assignments and subsets. These are a
local reproduction of the notebook protocol; compare with Tiến's original frozen
artifacts before combining independently produced results.

The raw corpus has 228 distinct characters, including two degree-symbol occurrences
outside our fixed vocabulary. Whole unsupported/overlength training samples are
excluded and counted in `dataset_audit.json`. Validation retains every sample and
its original label/source ID. No transliteration, unsupported-character deletion,
label truncation, or ground-truth-conditioned decoding is performed on validation.
Only NFC and the shared whitespace-removal convention are applied.

## Environment

Python 3.12. Install the matching CUDA builds before the teacher dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements-teacher.txt -r requirements-dev.txt
git clone https://github.com/baudm/parseq ~/.cache/vi-dfd/parseq
git -C ~/.cache/vi-dfd/parseq checkout 1902db043c029a7e03a3818c616c06600af574be
```

Override `PARSEQ_ROOT` if upstream is located elsewhere. The CUDA device is hidden
inside this session's sandbox; GPU commands need execution outside that sandbox.

## Shared training budget

`configs/training_budget.yaml` fixes the same batch/epoch budget in all run YAMLs:

- Physical/effective batch **64**, one GPU, accumulation 1.
- **20 epochs**, matching the pinned upstream default epoch count.
- AdamW, one-cycle, base LR 7e-4, upstream scaling by effective batch/256.
- Actual peak LR on one GPU: **1.75e-4**; warm-up fraction 0.075; weight decay 0.
- Gradient clipping 20, mixed FP16, seed 42 for the teacher.
- Photometric RandAugment, magnitude 5, 3 operations; geometric operations excluded.
- Max label length 25; validation keeps longer labels in its denominator.
- Choose checkpoint by full validation word accuracy; also log heavy-word accuracy.

The measured RTX 3060 Laptop GPU probe used full 25-character targets, all six
permutations, backward and an AdamW step. Batch 64 peaked at **2248.7 MiB** allocated,
well within 6 GB. This validates memory fit; it does not establish teacher accuracy.
The 20-epoch budget must be assessed for teacher credibility on validation after E1.

## Run

Commit code/config changes first so each artifact records the exact source SHA.

```bash
python train.py --config configs/runs/E1/teacher_s42.yaml \
  --data-root ../datasets/vintext/processed/vintext_notebook --smoke-test

python train.py --config configs/runs/E1/teacher_s42.yaml \
  --data-root ../datasets/vintext/processed/vintext_notebook
```

Smoke mode runs one training and one validation batch and is recorded as a diagnostic,
not as an E1 result. Normal training uses all training/validation data. It does not
evaluate test or unseen test.

Artifacts under `results/runs/<run_id>/`:

- Effective YAML, source/config hash, fixed seed, GPU/runtime/package snapshot.
- Pretrained checkpoint hash, vocabulary/data identity and filtering audit.
- `validation.csv`: epoch, optimizer step, source/config hash, accuracy and 1-NED.
- Lightning loss logs, `checkpoints/last.ckpt`, best validation checkpoint.
- `training.json`: actual running/completed/failed status and best score/path.

After the full run has completed:

```bash
python scripts/finish_teacher.py results/runs/<run_id>
```

This verifies finite tensors, vocabulary/class shapes, the source commit and the
best checkpoint's CSV score. It writes `summary.json`, `summary.csv`, populated
`metrics.csv` and `teacher_parseq_s_229.pt`. The compact export is compatible with
`torch.load(..., weights_only=True)` and its tensors are checked for exact equality
with the best checkpoint. Recorded validation scores come from AMP inference;
perform a separate FP32 evaluation before the FP32/INT8 comparison in E4.

Resume with `--resume <trusted-local-last.ckpt>` and the identical configuration/data.
The checkpoint includes upstream permutation RNG state and Lightning optimizer/scheduler
state. Resume creates a new artifact directory; worker augmentation order is not claimed
to be bitwise identical to an uninterrupted run. Preserve the original and resumed
artifact paths together when reporting one logical run.

Because the artifact directory changes, Lightning's `ModelCheckpoint` does not reload
`best_model_score`: the resumed run's `best-*.ckpt` is the best of the epochs after the
resume only. Compare `validation_word_acc` in the `best-*.ckpt` names of both directories
and score the higher one. For student runs (20-40 min) prefer rerunning from scratch;
`scripts/run_queue.py` never resumes.

## Status boundary

Task 9 implements teacher training only. It does not claim that the teacher is already
credible, that DFD works, or that KD/student training or component metrics are finished.
