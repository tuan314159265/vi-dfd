# Kết quả E1 — CRNN và VietOCR

Cập nhật ngày 28/11/2026 (train hoàn tất ngày 23/11, chấm đủ split ngày 24/11). Cả hai baseline đã hoàn tất 20 epoch, seed 42, trên dữ liệu VinText của Tiến. Checkpoint tốt nhất được chọn hoàn toàn theo `validation` word accuracy (không train lại hoặc chọn lại theo test), sau đó chấm đầy đủ cả ba split `validation` (2.507 mẫu), `test` (7.201 mẫu), và `unseen_test` (10.068 mẫu) bằng `scripts/eval_baseline.py` với `eval.py` chung theo quyết định 4.

## Tổng hợp word accuracy trên tập `all`

| Model | Validation (%) | Test (%) | Unseen test (%) |
|---|---:|---:|---:|
| CRNN | 63.3028 | 58.1447 | 61.2932 |
| VietOCR | 84.6031 | 79.6001 | 83.1049 |

## Chi tiết theo split và subset

Mỗi model có đủ 12 dòng trong `metrics.csv` (3 split × 4 subset).

| Model | Split | All (%) | Diacritic-heavy (%) | Small crops (%) | Rare combinations (%) | Base ER (%) | Mod ER (%) | Tone ER (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CRNN | validation | 63.3028 | 68.3060 | 46.6258 | 16.6667 | 8.2999 | 2.4303 | 3.2332 |
| CRNN | test | 58.1447 | 63.2027 | 46.2136 | 12.5000 | 8.7890 | 2.7321 | 3.8056 |
| CRNN | unseen_test | 61.2932 | 66.8814 | 45.0418 | 27.2727 | 7.9035 | 2.5574 | 3.6438 |
| VietOCR | validation | 84.6031 | 90.7104 | 66.4622 | 50.0000 | 3.9926 | 0.9222 | 1.4755 |
| VietOCR | test | 79.6001 | 85.1789 | 68.0906 | 62.5000 | 5.9637 | 1.4350 | 2.4526 |
| VietOCR | unseen_test | 83.1049 | 86.3402 | 66.2839 | 69.6970 | 4.3457 | 1.0542 | 1.7345 |

Số lượng mẫu theo split (`all` / `diacritic_heavy` / `small_crops` / `rare_combinations`): validation 2.507 / 183 / 489 / 6; test 7.201 / 587 / 1.545 / 24; unseen_test 10.068 / 776 / 1.916 / 33. Nhóm `rare_combinations` rất ít mẫu nên không dùng để rút ra kết luận mạnh. Tập train giữ 23.267/23.269 mẫu sau lọc ký tự ngoài vocabulary.

## CRNN

- Run ID: `20261122T204604Z_crnn_s42_3daf107c`.
- Source commit: `4e05fbc0cec4c39765dcc5b5c54b7e46056589da`.
- Best checkpoint: `best-epoch=18-validation_word_acc=0.6330.pt`.
- Training elapsed: 5.72 phút.
- Initialization: `{"pretrained": false, "architecture": "local CNN-2xBiLSTM256-CTC"}`.

## VietOCR

- Run ID: `20261122T205156Z_vietocr_s42_b0682572`.
- Source commit: `4e05fbc0cec4c39765dcc5b5c54b7e46056589da`.
- Best checkpoint: `best-epoch=19-validation_word_acc=0.8460.pt`.
- Training elapsed: 22.48 phút.
- Initialization: `{"upstream_commit": "fe8c3a7fc714aec57ab81cec844eb3adf0c1636c", "pretrained": true, "weights_sha256": "0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db", "weights_source": "/home/tuan/.cache/torch/hub/checkpoints/vgg_seq2seq.pth"}`.

## Protocol và giới hạn

- Recipe: 20 epoch, batch 64, AdamW + OneCycleLR, photometric augmentation, AMP trên CUDA; xem YAML resolved trong từng run.
- CRNN là baseline local CNN + 2-layer BiLSTM256 + CTC, khởi tạo ngẫu nhiên; không nhận là tái lập một checkpoint CRNN pretrained cụ thể. Greedy CTC collapse.
- VietOCR `vgg_seq2seq` dùng pretrained tiếng Việt upstream và vocabulary gốc; fine-tune bằng CE teacher forcing, greedy decoding. Cần công bố rõ khác biệt dữ liệu pretraining so với PARSeq khi viết báo cáo.
- Cả hai dùng cùng pipeline ảnh crop 32×128; VietOCR chuyển tensor về [0,1]. Validation/test/unseen_test giữ toàn bộ mẫu, chấm case-insensitive bằng `eval.py` chung với metrics theo quyết định 4 (`scripts/eval_baseline.py --splits validation test unseen_test`).
- Đây là kết quả trên nhánh triển khai E1 (`feat/e1-baselines`), không phải run tại `exp-v1`/`exp-v2`; không thay đổi các tag chung.
- Các run smoke và run CRNN bị dừng khi sửa scheduler không tính vào kết quả.

## Artifact và việc tiếp theo

- Mỗi thư mục run chứa checkpoint best `.pt`, `resolved.yaml`, `metadata.json`, initialization provenance, `data_identity.json`, `validation.csv`, `training.json`, `baseline_predictions/`, `predictions/`, `metrics.csv` (đủ 12 dòng) và `eval_info.json`.
- Bản bàn giao local đầy đủ nằm ở `../e1_completed/<run_id>/`. Không commit checkpoint, dự đoán hoặc dữ liệu vào Git.
- Upload Drive private nhóm chưa được xác nhận (trước đó connector Drive trả 404 và browser host không khả dụng); cần bàn giao hai thư mục trên lên Drive nhóm theo quy ước `runs/<run_id>/`.
- Chưa có số liệu ONNX/latency cho CRNN và VietOCR. Xem thêm [hướng dẫn baseline](BASELINES.md).
