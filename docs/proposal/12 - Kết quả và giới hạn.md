---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Kết quả và giới hạn

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

> [!info] Chưa có kết quả
> Các bảng dưới đây là template từ proposal. Dấu “—” nghĩa là chưa đo, không phải bằng 0. Chưa có cơ sở kết luận DFD tốt hơn KD thường hoặc tone bị ảnh hưởng mạnh hơn base.

## Main results — VinText test

Mean ± std trên ba seed cho các student E2. Teacher và additional baselines ghi đúng số seed thực tế.

| Model | Params | CPU latency ms | Word acc. all | Word acc. diacritic-heavy | Tone ER | Base ER |
|---|---|---|---|---|---|---|
| PARSeq-S teacher | — | — | — | — | — | — |
| Ti vanilla | — | — | — | — | — | — |
| Ti + standard KD | — | — | — | — | — | — |
| Ti + factorized, no KD | — | — | — | — | — | — |
| Ti + DFD | — | — | — | — | — | — |
| VietOCR vgg_seq2seq | — | — | — | — | — | — |

- [ ] Bảng unseen test tương ứng.
- [ ] Modifier ER và insertion/deletion trong bảng component metrics.
- [ ] Per-seed paired differences.
- [ ] Bảng FP32/INT8 và bootstrap CI của E4.
- [ ] Tone confusion matrices.
- [ ] Bảng generalization của E5.
- [ ] Accuracy-vs-latency plot của E6.
- [ ] Error categories và ảnh ví dụ E7.
- [ ] CSV và liên kết run/checkpoint/config/commit cho mỗi hàng kết quả.

Ablation results có mẫu riêng trong [11 - Ablation E3](11%20-%20Ablation%20E3.md).

## Conclusion — chờ dữ liệu

- **RQ1:** Chưa trả lời.
- **RQ2:** Chưa trả lời.
- **Chi phí triển khai:** Chưa đo.

Nếu không có improvement, báo cáo trung thực và phân tích tác động của compression, uncertainty và trade-off accuracy/efficiency.

## Limitations theo proposal

1. Chỉ nhận dạng word crops; chưa bao gồm detector có thể crop mất dấu trước recognition.
2. Khoảng 25K training words theo bản gốc; tổ hợp hiếm ít dữ liệu, uncertainty lớn.
3. Một teacher–student pair và một quantization scheme dynamic INT8; chưa bao quát pruning hoặc kiến trúc khác.
4. Tone-focused weighting là heuristic; learning component weights là hướng mở rộng.
5. Khả năng áp dụng cho các script giàu dấu khác là future work, chưa được kiểm chứng.

## Khi tổng hợp kết quả

Chỉ gộp các run đúng protocol. Ghi khác biệt pretrained data, split, GPU/runtime và training budget. Khi phần cứng training khác nhau, không trộn training time thành so sánh tốc độ mô hình; đo deployment latency theo protocol chung.

Xem [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md), [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md), [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).
