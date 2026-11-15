# Implementation state

## Working in this initial repository

- YAML template loading/validation and unresolved-field reporting.
- Fixed-seed utility for Python and installed NumPy/PyTorch.
- Run preparation with full Git SHA, config hash, seed, actual GPU detection,
  resolved config, environment snapshot and per-run CSV files.
- Pure-Python NFC base/modifier/tone decomposition and composition.
- One standalone YAML per listed run and a job manifest with model/seed identity.
- Artifact-layout coverage check and exported Obsidian proposal notes.

## Pending, in implementation order

1. Verify VinText terms and actual annotation format; crop with perspective warp/margin.
2. Freeze image-level split, near-duplicate groups, vocabulary and subsets.
3. Finalize EOS/token handling; hand-verify vocabulary and fixed mapping matrices.
4. Pin/integrate PARSeq and a GPU-compatible dependency lock; run E1.
5. Implement shared metrics with Levenshtein alignment, denominators and test cases.
6. ~~Implement factorized torch head and valid-character log-probability composition.~~ Done (`dfd/heads.py`).
7. ~~Implement standard KD and DFD with shared permutations/contexts and masked loss.~~ Done
   (`dfd/distill.py`, `dfd/parseq_kd.py`); wiring into the training script waits for Task 9's code.
8. Finalize equal tuning budget before E2; then run seeds 42/43/44.
9. Run E3 priority ablations, paired FP32/INT8 E4, and generalization E5.
10. Implement/export ONNX, benchmark on one machine (E6), analyze validation errors (E7).
11. Implement `/recognize`, final CPU dependencies and the demo Docker image.

The stage entry points intentionally fail with an actionable message until their
backend is implemented. `/health` is available in the API skeleton; `/recognize`
returns HTTP 501 rather than placeholder OCR predictions.

## Decisions made

- Vocabulary: 229 classes (228 characters + EOS at class 0), checked against VinText. Spaces are removed from labels; crops containing `°` are dropped from training and counted as errors in evaluation.
- Heavy word: at least 2 characters carrying a tone, a modifier, or `đ/Đ`.
- KD (`dfd/distill.py`, `dfd/parseq_kd.py`): teacher and student share images, permutations,
  context and masks; the teacher is frozen, in eval mode, outside the student's checkpoint and
  optimizer. CE and KD are means over valid target positions, pooled over permutations exactly
  like upstream PARSeq CE; EOS targets count only in the canonical and reverse permutations.
  Objective = CE + lambda * tau^2 * KD. Factorized students use per-component CE.
- alpha_i uses the teacher's tau-softened tone marginal, as written in proposal note 06.
- Joint KD on the factorized student composes the tempered component logits with the validity mask.
- The validity mask does not enter training for marginal DFD (component CE + marginal KL), so the
  `no_validity_mask` ablation differs from full DFD only at decoding and scoring.

## Decisions to finalize

- Component ER denominators, insertion/deletion handling, alignment tie-breaking.
- Standard-KD tuning allocation: DFD's listed grid has 12 trials vs KD's 6.
- E4 teacher count: one teacher seed vs table wording of three seeds.
- Compare quantization within the same checkpoint; separate it from downsizing.
- Meaning of “one standard deviation” in E2's decision rule.
- Valid decoding/scoring for the unmasked Cartesian-product ablation.
- `joint_lambda` for the `marginal_and_joint` ablation (the loss refuses to run while it is null).

`configs/tuning.yaml` is a study policy, not a model-run config accepted by train.py.
The run templates contain candidate KD values only. Training settings stay unset until E1.
