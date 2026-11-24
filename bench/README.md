# ONNX export and CPU benchmark

PARSeq export includes encoder, autoregressive decoding, composed-character head (when present) and refinement. Batch 1, input 1×3×32×128, opset 17. AR is unrolled to the model maximum of 25 characters plus EOS, with **fixed maximum horizon** (no early termination). Decoded equivalence against native PyTorch is verified on real validation crops; this is bounded validation, not proof over all images.

Dynamic INT8 quantizes supported constant-weight MatMul/Gemm; Conv and unsupported operators stay FP32. It is mixed precision. INT8 accuracy must be measured separately, never inferred from FP32 parity.

## Commands

```bash
python bench/export_onnx.py --checkpoint <best.ckpt> --output <model_fp32.onnx> --verify-data <data>/lmdb/val --verify-samples 32
python bench/quantize.py --input <model_fp32.onnx> --output <model_int8.onnx>
python bench/latency.py --model <model_fp32.onnx> --output <fp32_latency.json>
python bench/latency.py --model <model_int8.onnx> --output <int8_latency.json>
python bench/predict_onnx.py --model <model.onnx> --data-root <data> --output <prediction_dir> --splits test unseen_test
# Score with the appropriate E4/E6 precision config and eval.py --predictions-dir.
```

CPU: ONNX Runtime, batch 1, four intra-op CPU threads, one inter-op thread, sequential execution, warmup 50, median 500 measurements. Fixed seeded input is already preprocessed. Timing includes full fixed-horizon graph, excludes preprocessing and string decoding. Record fixed horizon in result tables; do not compare directly to early-stop native PyTorch timings or claim interactive end-to-end latency. All models need the same convention and hardware/session.

Current export supports PARSeq teacher/student. Baseline CRNN/VietOCR ONNX exports and GPU throughput are not implemented. Do not treat the local prototype measurements as a complete E4/E6 comparison.
