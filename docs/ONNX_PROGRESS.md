# ONNX / E4 & E6 Progress — 28/11/2026

Implemented PARSeq fixed-horizon AR + refinement ONNX export (batch 1, opset 17), dynamic INT8 constant-weight `MatMul`/`Gemm` quantization, source-ID prediction CSV export, and CPU latency measurement.

All **26/26 ONNX tasks** (`../onnx_artifacts/completion.json`, `status: completed`) are finished: 13 checkpoints (`teacher_s42` plus seeds 42, 43, 44 for `ti_ce`, `ti_factorized`, `ti_kd`, and `ti_dfd`) × 2 precisions (`fp32` and `int8_dynamic`). Checkpoint mapping from the group archive is recorded in `../onnx_artifacts/group_models.json`. All ONNX models, predictions, and latency JSONs are stored outside Git in `../onnx_artifacts/` (~549 MB); all 26 scored run directories are stored outside Git in `results/runs/` (each evaluated on all 7,201 `test` and 10,068 `unseen_test` samples across all 4 subsets via `eval.py --predictions-dir` under Decision 4 metrics).

## Summary by variant (mean ± SD across 3 seeds; teacher is 1 seed)

| Variant | Precision | Seeds | ONNX size (MB) | Test word acc (%) | Unseen word acc (%) | Test tone ER (%) | Median CPU latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `teacher` | FP32 | 1 | 93.24 | 84.60 | 87.79 | 2.20 | 35.279 |
| `teacher` | INT8 dynamic | 1 | 34.07 | 84.59 | 87.82 | 2.20 | 20.605 |
| `ti_ce` | FP32 | 3 | 24.85 | 82.66 ± 0.48 | 85.63 ± 0.26 | 2.40 ± 0.07 | 13.022 ± 0.378 |
| `ti_ce` | INT8 dynamic | 3 | 10.25 | 82.64 ± 0.56 | 85.53 ± 0.25 | 2.39 ± 0.10 | 10.331 ± 1.679 |
| `ti_factorized` | FP32 | 3 | 24.82 | 82.63 ± 0.10 | 85.72 ± 0.13 | 2.29 ± 0.04 | 12.524 ± 0.477 |
| `ti_factorized` | INT8 dynamic | 3 | 10.20 | 82.62 ± 0.08 | 85.76 ± 0.16 | 2.31 ± 0.05 | 9.687 ± 0.845 |
| `ti_kd` | FP32 | 3 | 24.85 | 82.74 ± 0.48 | 85.47 ± 0.34 | 2.43 ± 0.21 | 14.817 ± 6.430* |
| `ti_kd` | INT8 dynamic | 3 | 10.25 | 82.64 ± 0.43 | 85.45 ± 0.43 | 2.46 ± 0.21 | 11.835 ± 4.490* |
| `ti_dfd` | FP32 | 3 | 24.82 | 82.75 ± 0.12 | 86.06 ± 0.14 | 2.26 ± 0.05 | 11.680 ± 0.334 |
| `ti_dfd` | INT8 dynamic | 3 | 10.20 | 82.71 ± 0.18 | 86.06 ± 0.09 | 2.26 ± 0.02 | 9.165 ± 0.539 |

\* `ti_kd` latency SD is inflated by `ti_kd_s42` being timed in an earlier session (see caveats below). Excluding `ti_kd_s42`, `ti_kd_s43`/`s44` match the other PARSeq-Ti runs (`11.604`/`10.627 ms` FP32; `9.348`/`9.139 ms` INT8 dynamic).

## Full 26-task results (E4 accuracy & E6 CPU latency)

| Checkpoint | Precision | Test all (%) | Test heavy (%) | Unseen all (%) | Base ER (%) | Mod ER (%) | Tone ER (%) | Size (MB) | Median (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `teacher_s42` | FP32 | 84.5994 | 84.6678 | 87.7930 | 4.0069 | 1.3269 | 2.1991 | 93.24 | 35.279 |
| `teacher_s42` | INT8 dynamic | 84.5855 | 84.4974 | 87.8228 | 3.9770 | 1.3083 | 2.2028 | 34.07 | 20.605 |
| `ti_ce_s42` | FP32 | 82.1275 | 82.6235 | 85.5483 | 4.4653 | 1.6139 | 2.4749 | 24.85 | 13.024 |
| `ti_ce_s42` | INT8 dynamic | 82.0303 | 82.6235 | 85.5085 | 4.4318 | 1.5617 | 2.4898 | 10.25 | 11.895 |
| `ti_ce_s43` | FP32 | 83.0718 | 84.4974 | 85.9257 | 4.4430 | 1.5058 | 2.3333 | 24.85 | 13.399 |
| `ti_ce_s43` | INT8 dynamic | 83.1412 | 84.3271 | 85.7966 | 4.3684 | 1.5207 | 2.3035 | 10.25 | 10.541 |
| `ti_ce_s44` | FP32 | 82.7663 | 83.3049 | 85.4291 | 4.2491 | 1.4462 | 2.3780 | 24.85 | 12.643 |
| `ti_ce_s44` | INT8 dynamic | 82.7524 | 83.3049 | 85.2900 | 4.2566 | 1.5096 | 2.3631 | 10.25 | 8.557 |
| `ti_factorized_s42` | FP32 | 82.6135 | 83.4753 | 85.7767 | 4.4541 | 1.2859 | 2.2998 | 24.82 | 12.508 |
| `ti_factorized_s42` | INT8 dynamic | 82.5441 | 82.7939 | 85.7966 | 4.4653 | 1.3195 | 2.3035 | 10.20 | 10.662 |
| `ti_factorized_s43` | FP32 | 82.5302 | 84.3271 | 85.5781 | 4.4914 | 1.4387 | 2.2513 | 24.82 | 13.009 |
| `ti_factorized_s43` | INT8 dynamic | 82.5996 | 84.3271 | 85.5781 | 4.5100 | 1.4462 | 2.2737 | 10.20 | 9.236 |
| `ti_factorized_s44` | FP32 | 82.7385 | 82.7939 | 85.8164 | 4.3200 | 1.3717 | 2.3258 | 24.82 | 12.054 |
| `ti_factorized_s44` | INT8 dynamic | 82.7107 | 82.6235 | 85.8959 | 4.2417 | 1.3381 | 2.3668 | 10.20 | 9.163 |
| `ti_kd_s42` | FP32 | 82.2802 | 82.9642 | 85.5085 | 4.4541 | 1.6102 | 2.5085 | 24.85 | 22.221* |
| `ti_kd_s42` | INT8 dynamic | 82.2525 | 83.1346 | 85.5185 | 4.4355 | 1.6102 | 2.5197 | 10.25 | 17.019* |
| `ti_kd_s43` | FP32 | 83.2384 | 85.0085 | 85.7966 | 4.3386 | 1.5990 | 2.1917 | 24.85 | 11.604 |
| `ti_kd_s43` | INT8 dynamic | 83.0996 | 85.1789 | 85.8462 | 4.3460 | 1.6177 | 2.2252 | 10.25 | 9.348 |
| `ti_kd_s44` | FP32 | 82.6968 | 84.3271 | 85.1112 | 4.4020 | 1.5841 | 2.5942 | 24.85 | 10.627 |
| `ti_kd_s44` | INT8 dynamic | 82.5580 | 84.1567 | 84.9921 | 4.4280 | 1.5729 | 2.6352 | 10.25 | 9.139 |
| `ti_dfd_s42` | FP32 | 82.8913 | 83.4753 | 86.0648 | 4.3870 | 1.2486 | 2.3072 | 24.82 | 11.436 |
| `ti_dfd_s42` | INT8 dynamic | 82.9190 | 83.4753 | 86.0946 | 4.3833 | 1.2486 | 2.2886 | 10.20 | 9.395 |
| `ti_dfd_s43` | FP32 | 82.6830 | 84.1567 | 86.2038 | 4.3721 | 1.4276 | 2.2401 | 24.82 | 11.543 |
| `ti_dfd_s43` | INT8 dynamic | 82.6413 | 83.3049 | 86.1343 | 4.3945 | 1.4052 | 2.2476 | 10.20 | 8.549 |
| `ti_dfd_s44` | FP32 | 82.6691 | 83.6457 | 85.9158 | 4.4169 | 1.4909 | 2.2215 | 24.82 | 12.060 |
| `ti_dfd_s44` | INT8 dynamic | 82.5719 | 83.8160 | 85.9654 | 4.4467 | 1.5021 | 2.2587 | 10.20 | 9.550 |

## Verification and measurement caveats

- **FP32 parity:** Decoded text matches native PyTorch on real validation crops (32 samples for 12 checkpoints; 16 samples for the earlier `ti_kd_s42` pilot), with max absolute logit difference between `2.46e-5` and `8.01e-5`. This is bounded sample verification, not a proof over all images.
- **Dynamic INT8 scope:** Constant-weight `MatMul`/`Gemm` nodes are quantized; `Conv` and unsupported operators remain FP32 (mixed precision). INT8 accuracy is measured independently on full `test` and `unseen_test` splits rather than inferred from FP32 parity.
- **Latency protocol:** ONNX Runtime 1.20.1 on CPU, 4 intra-op threads, 1 inter-op thread, sequential execution, batch 1, 50 warmup calls, median of 500 timed calls on a fixed preprocessed input. Measurements cover the **fixed maximum horizon** (25 characters + EOS, encoder + AR + refinement, no early stop) and exclude preprocessing and string decoding. Do not compare directly with early-stop native PyTorch timings.
- **Session caveat for `ti_kd_s42`:** `ti_kd_s42` (`22.221 ms` FP32, `17.019 ms` INT8 dynamic) was timed in an earlier session than the remaining 24 tasks (`ti_kd_s43`/`s44` are `11.604`/`10.627 ms` FP32 and `9.348`/`9.139 ms` INT8). An older pilot timing file (`ti_kd_s42_int8_latency.json`, `25.783 ms`) is excluded; tables above use `ti_kd_s42_int8_dynamic_latency.json`. For strict single-session E6 reporting (`docs/EXPERIMENTS.md`), re-timing the ONNX models in one uninterrupted CPU session is recommended before finalizing report tables.

## Remaining follow-up items

- **E4 paired bootstrap:** Compute paired word-level bootstrap 95% CIs (Decision 4: 1,000 resamples, seed 42, relative ER change between FP32 and dynamic INT8 per checkpoint) from the saved `test` prediction CSVs.
- **Out of scope for PARSeq ONNX tools:** Baseline CRNN/VietOCR ONNX exports and GPU throughput benchmarks are not implemented here.
- **Artifact handoff:** Local artifacts (`../onnx_artifacts/`, ~549 MB, and 26 scored run folders under `results/runs/`) still need upload confirmation to the private group Drive according to group storage conventions.

See [bench/README.md](../bench/README.md) for commands and protocol details.
