---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Bài toán và đóng góp

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Abstract gốc

Vietnamese text relies on diacritics: tone marks and vowel modifiers turn a small set of Latin letters into more than two hundred distinct characters, many of which differ only by a tiny mark. Optical character recognition (OCR) systems deployed on phones and CPUs need small models, yet it is not known what model compression costs for such a script. This proposal studies (i) whether shrinking or quantizing a scene text recognizer degrades tone recognition faster than base-letter recognition, and (ii) how to recover what is lost. We propose Diacritic-Factorized Distillation (DFD): the 229-way output distribution of a full-size PARSeq teacher is marginalized into base-letter, vowel-modifier and tone distributions, which supervise a lightweight PARSeq-Ti student equipped with factorized output heads. Experiments on the VinText benchmark will compare DFD with standard training and standard logit distillation under an identical budget, reporting component-wise error rates, variance over three seeds and CPU latency.

## Động lực

Nhận dạng chữ trong ảnh phục vụ đọc biển hiệu, hóa đơn, tài liệu định danh, eKYC và số hóa tài liệu. Điện thoại và máy chủ CPU cần mô hình nhỏ, độ trễ thấp.

Tiếng Việt dùng dấu biến âm và dấu thanh. Các chữ ă, â, ê, ô, ơ, ư và đ mở rộng bảng Latin; năm dấu thanh tạo sáu thanh kể cả ngang. Ví dụ `ma, má, mà, mả, mã, mạ` khác nghĩa chỉ bởi dấu thanh. Dấu rất nhỏ, dễ mờ, bị crop mất hoặc nhầm lẫn.

KD thường truyền toàn bộ phân phối ký tự. Thông tin về một thanh nằm rải rác trên nhiều ký tự có cùng thanh; student phải tự tổng hợp tín hiệu đó. Giả thuyết là năng lực phân biệt thanh dễ suy giảm khi dung lượng mô hình giảm.

## Câu hỏi nghiên cứu

- **RQ1:** Khi giảm PARSeq-S → PARSeq-Ti hoặc FP32 → INT8, lỗi dấu thanh có tăng bất cân xứng so với lỗi chữ gốc không?
- **RQ2:** Với cùng kích thước student và ngân sách huấn luyện, chưng cất trong không gian phân rã dấu có phục hồi tốt hơn KD thông thường không?

Primary metric dự kiến cho RQ2: word accuracy trên từ có ít nhất hai ký tự mang dấu. Định nghĩa chính xác ở [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md).

## Đóng góp dự kiến, phụ thuộc bằng chứng

1. Phân tích tác động của thu nhỏ mô hình và INT8 lên lỗi base, modifier và tone.
2. DFD: áp dụng marginal-based distillation [4] để chuyển teacher dùng vocabulary đầy đủ sang student dùng head phân rã mà không phải huấn luyện lại teacher theo kiến trúc phân rã.
3. Protocol đánh giá component ER, tone confusion và CPU latency, kèm code tái lập.

Không nhận factorized head là mới: thiết kế kế thừa VietTDR [3]. Không nhận nguyên lý chưng cất marginal là mới: kế thừa Structural KD [4]. Điểm mới dự kiến nằm ở câu hỏi nghiên cứu, bằng chứng và cách thích nghi cho nén recognizer tiếng Việt.

## Phạm vi

Nhận dạng **ảnh crop từ**, chưa đánh giá hệ thống detection + recognition hoàn chỉnh. Teacher PARSeq-S, student PARSeq-Ti; FP32 và dynamic INT8; VinText là benchmark chính.

Xem [02 - Nghiên cứu liên quan](02%20-%20Nghi%C3%AAn%20c%E1%BB%A9u%20li%C3%AAn%20quan.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md) và [12 - Kết quả và giới hạn](12%20-%20K%E1%BA%BFt%20qu%E1%BA%A3%20v%C3%A0%20gi%E1%BB%9Bi%20h%E1%BA%A1n.md).
