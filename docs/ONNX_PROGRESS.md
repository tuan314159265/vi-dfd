# ONNX / E6 progress — 24/11/2026

Implemented PARSeq full fixed-horizon AR + refinement export (batch 1, opset 17), dynamic INT8 constant-weight MatMul/Gemm quantization, source-ID prediction CSV export, and CPU latency measurement. Models are outside Git in `../onnx_artifacts/`.

## Verified pilot: KD seed 42

- FP32 decoded text matches native PyTorch on 16 real validation crops; max absolute logits error 0.000032425. Bounded verification, not proof on all samples.
- INT8 graph loads on CPU; 109 MatMulInteger nodes. Conv remains FP32. Full accuracy evaluation still required.
- CPU four intra-op threads, batch 1, 50 warmup, median 500 timed calls, sequential FP32 then INT8 in one session, fixed seeded input.

| Precision | Median graph latency (ms) |
|---|---|
| FP32 | 32.752 |
| Dynamic INT8 | 25.783 |

These are **fixed maximum horizon**, not early-stop decoding timings. Includes encoder, AR and refinement; excludes preprocessing/string decoding. Do not present as a complete final E6 comparison or compare with early-stop native timing.

## Remaining dependencies

- Full ti_ce and full ti_dfd checkpoints are absent locally. Need artifacts from the group for E4/E6 comparisons; joint_kd is not a substitute for full DFD.
- Teacher and KD seed 43/44 export and accuracy need completion; baseline CRNN/VietOCR ONNX export and GPU throughput remain unsupported by these tools.
- Drive upload blocked: connector returns 404 for the supplied private folder, browser host unavailable. Artifacts remain local.

See bench/README.md for commands and measurement limitations.
