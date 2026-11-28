# ONNX export and CPU benchmark

PARSeq export includes the encoder, autoregressive decoding, composed-character head (when present), and refinement. Batch 1, input `1×3×32×128`, opset 17. Autoregressive decoding is unrolled to the model maximum of 25 characters plus EOS, with a **fixed maximum horizon** (no early termination). Decoded equivalence against native PyTorch is verified on real validation crops during export; this is bounded validation, not proof over all images.

Dynamic INT8 quantizes supported constant-weight `MatMul`/`Gemm` operators; `Conv` and unsupported operators remain FP32 (mixed precision). INT8 accuracy must always be scored separately on the full evaluation splits and never inferred from FP32 parity.

## Commands

```bash
python bench/export_onnx.py --checkpoint <best.ckpt> --output <model_fp32.onnx> --verify-data <data>/lmdb/val --verify-samples 32
python bench/quantize.py --input <model_fp32.onnx> --output <model_int8.onnx>
python bench/latency.py --model <model_fp32.onnx> --output <fp32_latency.json>
python bench/latency.py --model <model_int8.onnx> --output <int8_latency.json>
python bench/predict_onnx.py --model <model.onnx> --data-root <data> --output <prediction_dir> --splits test unseen_test
# Score with the appropriate E4/E6 precision config and eval.py --predictions-dir.
```

## CPU latency protocol and limitations

- Runtime: ONNX Runtime CPU, batch 1, 4 intra-op threads, 1 inter-op thread, sequential execution, 50 warmup calls, median of 500 timed calls on a fixed preprocessed input tensor.
- Scope: timing measures the full fixed-horizon ONNX graph and excludes image preprocessing and string decoding. Record fixed-horizon timing in result tables; do not compare directly to early-stop native PyTorch timings or claim interactive end-to-end latency.
- Single-session requirement: all models compared in E6 latency tables must be measured under the same convention, hardware, and uninterrupted session.
- Coverage: this pipeline supports PARSeq teacher and student checkpoints (26/26 E4/E6 tasks completed; see [docs/ONNX_PROGRESS.md](../docs/ONNX_PROGRESS.md)). Baseline CRNN/VietOCR ONNX exports and GPU throughput benchmarks are not implemented here.
