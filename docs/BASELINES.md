# E1 baseline training

`train.py` supports `crnn` and `vietocr` on the same frozen VinText LMDBs.

- CRNN is a local RGB CNN + two bidirectional LSTM256 layers + CTC baseline, randomly initialized. It is not a claim to reproduce a particular pretrained CRNN. Its width axis has 64 CTC steps. Checkpoint selection is case-insensitive validation word accuracy.
- VietOCR uses official `vgg_seq2seq` Vietnamese weights and its original vocabulary, pinned upstream revision `fe8c3a7fc714aec57ab81cec844eb3adf0c1636c`. Training labels are restricted to the shared project charset. All evaluation samples remain included. Native VietOCR inputs use [0,1]; the shared photometric transform is converted back from [-1,1].
- Budget follows YAML: batch 64, 20 epochs, AdamW/OneCycleLR, AMP on CUDA, gradient clip 20. VietOCR uses cross-entropy teacher forcing and greedy autoregressive decoding; CRNN uses CTC and greedy collapse. Upstream Vietnamese pretraining differs from PARSeq pretraining and must be disclosed.
- Outputs: best `.pt` checkpoint, `validation.csv` with 1-based epochs, initialization provenance, data hashes, raw prediction CSVs and shared `eval.py` metrics. No teacher is required. Baseline checkpoints are not PARSeq Lightning checkpoints: do not pass them to `eval.py --checkpoint` or the PARSeq queue.
- Smoke runs carry `diagnostic: fast_dev_run` and `smoke_completed` status; they are never research results. Baseline resume is currently refused.

## Setup (fish)

```fish
uv pip install --python .venv/bin/python -r requirements-baselines.txt
git clone https://github.com/pbcquoc/vietocr ~/.cache/vi-dfd/vietocr
git -C ~/.cache/vi-dfd/vietocr checkout fe8c3a7fc714aec57ab81cec844eb3adf0c1636c
source .venv/bin/activate.fish
set DATA "$HOME/Documents/School/261/DL/datasets/vintext/tien_20261116"
```

Run from the committed baseline branch, not `exp-v1` (which predates this code). Record this implementation commit for E1. Do not move the group's existing tags.

```fish
python train.py --config configs/runs/E1/crnn_s42.yaml --data-root "$DATA" --smoke-test
python train.py --config configs/runs/E1/vietocr_s42.yaml --data-root "$DATA" --smoke-test
# Then full runs, sequentially:
python train.py --config configs/runs/E1/crnn_s42.yaml --data-root "$DATA"
python train.py --config configs/runs/E1/vietocr_s42.yaml --data-root "$DATA"
```

Each full run exports/scorers validation automatically. To produce test/unseen predictions later, use the saved checkpoint with a baseline-specific export command (not yet provided); do not tune against test. No ONNX export is included in this implementation.


## Kết quả local đã kiểm tra

CRNN và VietOCR đã hoàn tất 20 epoch và chấm validation; xem [RESULTS_E1_BASELINES.md](RESULTS_E1_BASELINES.md). Chưa chấm test/unseen_test hoặc đo latency.
