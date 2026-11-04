---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Baselines và cấu hình huấn luyện

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Đối chứng chính — cùng PARSeq-Ti backbone và ngân sách

| Model | Head | Objective | Mục đích |
|---|---|---|---|
| Ti vanilla | Single 230-way | CE ký tự | Mốc student thường |
| Ti + standard KD | Single 230-way | CE + full-vocabulary KL từ teacher | So với KD thông thường [5] |
| Ti + factorized head, no KD | Ba head | CE components | Tách hiệu ứng của head |
| Ti + DFD — ours | Ba head | CE components + marginal KL | Phương pháp đề xuất |

## Baseline bổ sung

- **PARSeq-S teacher:** tham chiếu accuracy/efficiency, không bảo đảm là upper bound tuyệt đối trước khi đo.
- **VietOCR `vgg_seq2seq` [9]:** khởi tạo từ Vietnamese pretrained weights, fine-tune cùng split.
- **CRNN:** baseline nhẹ cho efficiency comparison.

VietOCR đã được pretrain trên dữ liệu tiếng Việt; cần công bố khác biệt dữ liệu pretraining khi diễn giải so sánh.

## Training configuration theo proposal

| Mục | Teacher | Student variants |
|---|---|---|
| Initialization | Official English pretrained weights + head mới | Official `parseq_tiny` weights, head theo biến thể |
| Trainable | Toàn bộ model | Toàn bộ student; teacher frozen khi KD |
| Optimizer/schedule | AdamW, one-cycle theo PARSeq config | Cùng recipe |
| Batch/epochs | Chốt sau E1 | Giữ cố định cho các run so sánh |
| Checkpoint | Best validation word accuracy | Cùng tiêu chí |
| Precision | AMP | AMP |
| Seeds | Teacher 1 seed | E2: 3 seed; các mục khác theo ma trận |
| Hardware ban đầu | Một GPU khoảng 16 GB, ghi model thực tế | Theo tài nguyên từng người |

Phương án tài nguyên đã trao đổi: **mỗi người ưu tiên Colab Free + Kaggle; AWS dự phòng khi không đủ compute/tiến độ**. Xem [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).

## Tuning trong E2

- DFD: $\tau\in\{1,2,4\}$; $\lambda\in\{0.5,1\}$; $w_t\in\{1,2\}$.
- Standard KD được cấp ngân sách tuning tương đương trên validation.
- Tuning một seed; sau đó giữ hyperparameters và train ba seed.
- Cùng seed dùng cùng data order; giữ đồng nhất augmentation và các yếu tố kiểm soát được.

> [!todo] Chốt ngân sách tuning
> Grid DFD trên có 12 tổ hợp; grid $\tau\times\lambda$ của standard KD chỉ có 6. Bản gốc yêu cầu “same-size grid” nhưng chưa chỉ rõ 12 trial cho standard KD. Cần định nghĩa trước khi chạy.

## Quy ước bổ sung theo yêu cầu người dùng

- Cấu trúc repo rõ ràng, có `requirements.txt` và YAML.
- Kết quả CSV kèm commit hash.
- Seed cố định; quy ước dùng **42, 43, 44** cho run ba seed, **42** cho run một seed.
- Ghi phiên bản phần mềm, model weights và dữ liệu; lưu đủ config/checkpoint/run metadata.

Schema và ví dụ cấu hình ở [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).
