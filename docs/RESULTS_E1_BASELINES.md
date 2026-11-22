# Kết quả E1 — CRNN và VietOCR

Kiểm tra ngày 23/11/2026. Cả hai baseline đã hoàn tất 20 epoch, seed 42, trên dữ liệu VinText của Tiến. Chọn checkpoint theo validation word accuracy; chưa chấm test hoặc unseen_test.

| Model | Validation all (%) | Diacritic-heavy (%) | Small crops (%) | Rare combinations (%) |
|---|---|---|---|---|
| CRNN | 63.3028 | 68.3060 | 46.6258 | 16.6667 |
| VietOCR | 84.6031 | 90.7104 | 66.4622 | 50.0000 |

Validation có 2.507 mẫu; nhóm nhiều dấu 183, ảnh nhỏ 489, tổ hợp hiếm chỉ 6. Không kết luận mạnh từ nhóm hiếm. Train giữ 23.267/23.269 mẫu sau lọc ký tự ngoài vocabulary.

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
- VietOCR vgg_seq2seq dùng pretrained tiếng Việt upstream và vocabulary gốc; fine-tune bằng CE teacher forcing, greedy decoding. Cần công bố khác biệt dữ liệu pretraining so với PARSeq.
- Cả hai dùng cùng pipeline ảnh crop 32×128; VietOCR chuyển tensor về [0,1]. Validation giữ toàn bộ mẫu, chấm case-insensitive bằng eval.py chung với metrics theo quyết định 4.
- Đây là kết quả trên nhánh triển khai E1, không phải run tại exp-v1/exp-v2; không thay đổi các tag chung.
- Các run smoke và run CRNN bị dừng khi sửa scheduler không tính vào kết quả.

## Artifact và việc tiếp theo

Các thư mục `results/runs/<run_id>/` chứa checkpoint best `.pt`, resolved.yaml, metadata, initialization provenance, data_identity.json, validation.csv, training.json, baseline_predictions/, predictions/, metrics.csv và eval_info.json. Artifact ở local; chưa xác nhận upload Drive. Không commit checkpoint, dự đoán hoặc dữ liệu vào Git.

Cần bàn giao hai thư mục run qua Drive private và chấm test/unseen_test bằng checkpoint tốt nhất khi hoàn thiện lệnh export baseline. Chưa có số liệu latency/ONNX. Xem [hướng dẫn baseline](BASELINES.md).
