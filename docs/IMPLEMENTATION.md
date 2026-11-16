# Implementation state

## Working

- YAML template loading/validation and unresolved-field reporting.
- Fixed-seed utility for Python and installed NumPy/PyTorch.
- Run preparation with full Git SHA, config hash, seed, actual GPU detection,
  resolved config, environment snapshot and per-run CSV files.
- Pure-Python NFC base/modifier/tone decomposition and composition.
- One standalone YAML per listed run and a job manifest with model/seed identity.
- Artifact-layout coverage check and exported Obsidian proposal notes.
- Fixed 229-class vocabulary, mapping matrices and the factorized student head.
- Notebook crop/split/LMDB pipeline, plus a headless runner and frozen-data verification.
- PARSeq-S English initialization, new Vietnamese head, training, validation and checkpointing.
- Shared batch 64 / 20 epochs, verified with a CUDA memory probe; see `configs/training_budget.yaml`.
- Completed-run verification and a compact teacher weight export; see `docs/TEACHER.md`.

## Pending, in implementation order

1. Compare the locally reproduced frozen manifests with Tiến's originals before combining results.
2. Implement/run remaining E1 baselines and assess teacher credibility on validation.
3. ~~Implement shared metrics with Levenshtein alignment, denominators and test cases.~~ Metrics by Task 10 (`dfd/metrics.py`); evaluation pipeline in `eval.py`.
4. ~~Implement factorized torch head and valid-character log-probability composition.~~ Done (`dfd/heads.py`).
5. ~~Implement standard KD and DFD with shared permutations/contexts and masked loss.~~ Done
   (`dfd/distill.py`, `dfd/parseq_kd.py`); wiring `DistilledPARSeq` into the student training entry point is next.
6. ~~Finalize equal tuning budget before E2~~ (done: 12 trials each, `scripts/e2_tuning.py`); then run seeds 42/43/44.
7. Run E3 priority ablations, paired FP32/INT8 E4, and generalization E5.
8. Implement/export ONNX, benchmark on one machine (E6), analyze validation errors (E7).
9. Implement `/recognize`, final CPU dependencies and the demo Docker image.

Teacher, student training (`dfd/student.py`) and `eval.py` are implemented. `/health` is available in the API skeleton; `/recognize`
returns HTTP 501 rather than placeholder OCR predictions.

## Decisions made

- Vocabulary: 229 classes (228 characters + EOS at class 0), checked against VinText. Spaces are removed from labels; the two training crops containing `°` are excluded from training and such crops count as errors in evaluation.
- Heavy word: at least 2 characters carrying a tone, a modifier, or `đ/Đ`.
- Shared budget: batch 64, accumulation 1, 20 epochs, AdamW/one-cycle with upstream LR scaling.
- KD (`dfd/distill.py`, `dfd/parseq_kd.py`): teacher and student share images, permutations,
  context and masks; the teacher is frozen, in eval mode, outside the student's checkpoint and
  optimizer. CE and KD are means over valid target positions, pooled over permutations exactly
  like upstream PARSeq CE; EOS targets count only in the canonical and reverse permutations.
  Objective = CE + lambda * tau^2 * KD. Factorized students use per-component CE.
- alpha_i uses the teacher's tau-softened tone marginal, as written in proposal note 06.
- Joint KD on the factorized student composes the tempered component logits with the validity mask.
- Tuning budget: standard KD uses tau {1,2,4} x lambda {0.25,0.5,1,2} = 12 trials, equal to DFD.
- `joint_lambda = 1.0` for `marginal_and_joint`, so its loss is exactly full DFD + joint KD.
- Evaluation (`eval.py`, `dfd/evaluation.py`): every LMDB sample is scored, none filtered; labels and
  predictions are NFC, spaces removed, lowercased (case-insensitive protocol); label characters outside
  the table become the metrics invalid marker, so the crop counts as wrong. Split names in configs and
  `metrics.csv` are `validation`, `test`, `unseen_test` (LMDB folders `val`, `test`, `unseen_test`).
  Non-PARSeq models and INT8 exports are scored from a `source_id,prediction` CSV with the same code.
- The validity mask does not enter training for marginal DFD (component CE + marginal KL), so the
  `no_validity_mask` ablation differs from full DFD only at decoding and scoring.

## Decisions finalized on 2026-11-16

See `docs/EXPERIMENTS.md`, section 7: reference data build, teacher check, the E2 paired-difference
rule, component ER denominators and alignment, unmasked decoding for E3, E5 image-level subsampling
(`dfd/teacher_data.py`) and a single teacher seed for E4.

`configs/tuning.yaml` is a study policy, not a model-run config accepted by train.py.
The run templates contain candidate KD values only. The shared batch/epoch budget
is fixed; model-specific data paths and KD choices still need to be resolved.
