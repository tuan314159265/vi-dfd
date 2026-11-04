---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Nghiên cứu liên quan

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md) · Nguồn [1]–[13]: [16 - Tài liệu tham khảo](16%20-%20T%C3%A0i%20li%E1%BB%87u%20tham%20kh%E1%BA%A3o.md).

## Tiếng Việt như một bài toán OCR

Một ký tự có thể đồng thời mang dấu biến âm và dấu thanh: `ệ = e + circumflex + dot below`. Dấu nhỏ hoặc nằm ngoài bounding box khiến detector và recognizer dễ mất thông tin. Survey [10] và nghiên cứu diacritic-focused detection [8] là các nguồn động lực được bản gốc viện dẫn.

VinText [1] là benchmark chính được đề xuất dùng; công trình đi kèm sử dụng dictionary-guided recognition để xếp hạng lại ứng viên theo tương thích hình ảnh.

## STR và hiệu quả tính toán

- **PARSeq [2]:** visual encoder + Transformer decoder, permutation language modeling; hỗ trợ autoregressive, non-autoregressive và iterative refinement. Bản gốc chọn PARSeq-Ti cho student.
- **VietOCR [9]:** thư viện OCR tiếng Việt với VGG và Seq2Seq/Transformer, có pretrained weights tiếng Việt; dùng làm baseline thực tế.
- **CRNN:** baseline recognizer nhẹ cho so sánh hiệu quả.

## Diacritic-aware recognition

Theo mô tả trong proposal, **VietTDR [3]** phân rã ký tự thành base, modifier và tone, dùng các head song song, tone-position attention prior và phạt tổ hợp không hợp lệ. Mục tiêu là tăng độ chính xác với mô hình khoảng 21M tham số, không nghiên cứu nén hoặc distillation.

DFD chỉ kế thừa ý tưởng head phân rã; không dùng tone-position attention prior hoặc triple input embedding. Chi tiết ở [05 - Factorized output head](05%20-%20Factorized%20output%20head.md).

## Knowledge distillation

| Công trình | Vai trò trong đề tài |
|---|---|
| Hinton et al. [5] | KD bằng softened full-vocabulary distribution; baseline chính |
| Text is Text [6] | Character-level distillation cho text recognition |
| Symmetrical Linguistic Feature Distillation [7] | Chuyển kiến thức ngôn ngữ từ CLIP sang STR |
| Structural Knowledge Distillation [4] | Matching marginal distributions giữa structured predictors có cách phân rã khác nhau; nền tảng nguyên lý của DFD |

## Research gap theo proposal

1. Diacritic-aware Vietnamese STR tập trung tăng accuracy ở kích thước cố định; thiếu phân tích component-wise khi nén.
2. Các phương pháp STR KD được khảo sát chủ yếu distill trên vocabulary ký tự, chưa khai thác cấu trúc bên trong ký tự có dấu.
3. Marginal-based KD trong NLP chưa được bản khảo sát tìm thấy ứng dụng tương ứng cho orthographic components của recognizer cần nén.

> [!todo] Trước khi khẳng định novelty
> Kiểm chứng lại các nguồn và cập nhật related work. Các nhận định “chưa có” phía trên là lập luận của proposal, chưa phải kết luận sau một systematic literature review.

Liên quan: [01 - Bài toán và đóng góp](01%20-%20B%C3%A0i%20to%C3%A1n%20v%C3%A0%20%C4%91%C3%B3ng%20g%C3%B3p.md), [16 - Tài liệu tham khảo](16%20-%20T%C3%A0i%20li%E1%BB%87u%20tham%20kh%E1%BA%A3o.md).
