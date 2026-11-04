---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Ablation E3

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Tách tác động của kiến trúc, marginalization và weighting. Mọi biến thể dùng recipe và hyperparameters đã chọn trong E2; khởi tạo nhất quán theo seed.

| Variant | Thay đổi | Câu hỏi | Seeds |
|---|---|---|---|
| Full DFD | Giữ đủ head, marginal KL, weighting | Reference | 3 |
| − tone weighting | $w_t=1$ và $\alpha_i=1$ | Tone-focused weighting có ích không? | 3 |
| − validity mask | Ghép toàn bộ base × modifier × tone | Mask có giảm tổ hợp vô nghĩa không? | 1 |
| Joint KD on factorized student | KL trên composed full-vocabulary distribution thay marginal KL | Gain đến từ marginalization hay distillation vào head này? | 3 |
| Marginal + joint KD | Dùng cả hai KL terms | Hai tín hiệu có bổ sung nhau không? | 1 |
| Tone-only KD | Chỉ distill tone marginal | Bao nhiêu gain đến riêng từ tone? | 1 |
| Temperature sweep | $\tau\in\{1,2,4\}$ | Nhạy với temperature không? | 1 |

## Protocol

1. Giữ data pipeline, epoch budget và quy tắc checkpoint như E2.
2. Chọn checkpoint bằng validation.
3. Đánh giá test sau khi chọn xong cấu hình.
4. Ba-seed variants hỗ trợ claim; một-seed variants ghi rõ **indicative only**.
5. Khi giảm compute, cắt các biến thể phụ trước; giữ main baselines và thời gian viết.

## Điểm cần định nghĩa trước khi chạy

- [ ] Bỏ validity mask có thể sinh triple không có Unicode character trong vocabulary: chốt cách decode và tính accuracy/invalid-output rate.
- [ ] Chốt EOS cho cả joint distribution và unmasked ablation.
- [ ] Chốt relative weights trong marginal + joint KD.
- [ ] Phân biệt temperature sweep phục vụ ablation với tuning đã thực hiện ở E2.

## Mẫu bảng kết quả

| Variant | Seeds | Word acc. all | Word acc. heavy | Tone ER | Base ER | Ghi chú |
|---|---|---|---|---|---|---|
| Full DFD | 3 | — | — | — | — | Chưa chạy |
| − tone weighting | 3 | — | — | — | — | Chưa chạy |
| − validity mask | 1 | — | — | — | — | Indicative |
| Joint KD | 3 | — | — | — | — | Chưa chạy |
| Marginal + joint KD | 1 | — | — | — | — | Indicative |
| Tone-only KD | 1 | — | — | — | — | Indicative |
| Temperature sweep | 1/config | — | — | — | — | Indicative |

Xem [05 - Factorized output head](05%20-%20Factorized%20output%20head.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md), [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md).
