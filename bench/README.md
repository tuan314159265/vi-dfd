# Benchmark protocol

Entry points are scaffolds until PARSeq/ONNX integration.

CPU: ONNX Runtime, input 1x3x32x128, batch 1, four CPU threads, 50 warm-up runs,
median of 500 runs, every model on the same machine/session. Compare FP32 and
dynamic INT8 for each checkpoint. Record CPU/runtime and decoding settings.
GPU throughput: batch 64; record actual device and peak memory.
