---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# PARSeq và cấu hình mô hình

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## PARSeq xử lý crop như thế nào?

Theo cấu hình mô tả trong proposal [2]:

1. **Patch embedding:** ảnh 32 × 128, patch 4 × 8 → 8 × 16 = 128 visual tokens.
2. **ViT encoder:** 12 lớp Transformer tạo 128 visual feature vectors.
3. **Visio-lingual decoder:** một lớp decoder, mỗi vị trí đầu ra có learned position query; attention tới context characters được permutation mask cho phép và visual features.
4. **Classifier:** ánh xạ mỗi decoder output sang phân phối ký tự. Bản gốc mô tả 94 ký tự tiếng Anh + EOS; cấu hình tiếng Việt dự kiến 229 + EOS.

Permutation language modeling lấy nhiều thứ tự output trong một batch. Inference có thể decode trái sang phải hoặc song song rồi refinement.

## Teacher và student

| Thành phần | PARSeq-S teacher | PARSeq-Ti student |
|---|---|---|
| Model width | 384 | 192 |
| Encoder heads / MLP / layers | 6 / 1536 / 12 | 3 / 768 / 12 |
| Decoder heads / MLP / layers | 12 / 1536 / 1 | 6 / 768 / 1 |
| Tham số theo proposal | Khoảng 23.8M | Đo sau khi thay head |
| Output head | Linear 229 + EOS | Baseline: linear 229 + EOS; DFD: ba head |
| Khởi tạo | Official pretrained weights tiếng Anh | Official `parseq_tiny` weights |

- [ ] Đối chiếu cấu hình và tên checkpoint với repo PARSeq trước khi triển khai.
- [ ] Đo tham số thực tế của từng model thay vì chỉ dùng số ước lượng.

## Phần giữ nguyên trong student

Patch embedding, encoder, decoder, input character embeddings, permutation training và decoding procedure. Context vẫn là **ký tự đã ghép** trong vocabulary, không đổi thành ba input embeddings.

Chỉ thay output head, objective và loss weighting. DFD hướng tới kiểm soát tác động của từng thay đổi qua các baseline và ablation.

## Chuẩn bị model

1. Fine-tune PARSeq-S trên training split VinText bằng head mới theo vocabulary tiếng Việt.
2. Đánh giá teacher và baseline Ti ở E1.
3. Đóng băng teacher khi chạy KD; student được cập nhật toàn bộ.
4. Mọi run KD phải dùng cùng image augmentation, permutation và ground-truth context cho teacher/student.

Xem [05 - Factorized output head](05%20-%20Factorized%20output%20head.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md), [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md).
