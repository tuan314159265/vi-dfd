---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Chỉ số và benchmark

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Accuracy và lỗi ký tự

| Chỉ số | Đo điều gì | Chiều tốt | Giới hạn |
|---|---|---|---|
| Word accuracy — diacritic-heavy **(primary)** | Tỷ lệ từ hoàn toàn đúng trên subset có ≥ 2 ký tự mang dấu | Cao | Không phân biệt sai một dấu với sai cả từ |
| Word accuracy — full test | Tỷ lệ từ hoàn toàn đúng toàn bộ tập | Cao | Có thể bị chi phối bởi từ ngắn/không dấu |
| 1 − NED | Đúng ở mức ký tự theo normalized edit distance | Cao | Sai dấu được tính như sai ký tự; không phân loại nguyên nhân |
| Base / Modifier / Tone ER | Lỗi từng component trên vị trí đã alignment | Thấp | Insert/delete được báo riêng |
| Tone confusion matrix | Các cặp thanh hay bị nhầm | Diagnostic | Không phải một scalar score |
| Parameters, FLOPs | Quy mô mô hình và phép tính | Thấp | Không tương đương real latency |
| CPU latency, peak memory | Chi phí inference | Thấp | Phụ thuộc phần cứng/runtime |
| GPU throughput | Số mẫu xử lý theo thời gian | Cao | Phụ thuộc batch và phần cứng |

## Matching và alignment

1. Normalize prediction và ground truth bằng NFC.
2. Main evaluation **diacritic-sensitive, case-insensitive**; phiên bản case-sensitive đưa vào phụ lục báo cáo.
3. Dùng Levenshtein alignment giữa prediction và label.
4. Trên các cặp ký tự aligned, phân rã thành base/modifier/tone và so sánh.
5. Báo insertion/deletion riêng.

- [ ] Định nghĩa rõ denominator cho từng ER và cách xử lý EOS.
- [ ] Chốt tie-breaking khi có nhiều Levenshtein alignments tối ưu.
- [ ] Ghi rõ NED normalization, quy ước empty strings và cách average.
- [ ] Xem thêm tone ER trên vị trí nguyên âm để tránh che mất lỗi do nhiều ký tự mặc định thanh ngang; đây là mục cần chốt, chưa thay primary protocol.

## Thống kê theo proposal

- E2: mean ± standard deviation trên ba seed, kèm per-seed paired differences.
- E4: relative increase của từng ER, 95% bootstrap confidence intervals bằng resampling test words.
- Tone confusion cho teacher, vanilla Ti và DFD Ti.

Decision rules đầy đủ ở [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md).

## CPU latency protocol

- ONNX Runtime; FP32 và dynamic INT8.
- Input `1 × 3 × 32 × 128`, batch 1.
- 4 CPU threads.
- 50 warm-up runs; median của 500 runs.
- Tất cả model trên cùng máy, cùng phiên đo.
- Ghi CPU, runtime/version, precision, decoding mode và cấu hình liên quan.

## GPU và memory

Throughput ở batch 64; peak memory; accuracy-vs-latency plot. Cần ghi phần cứng thực tế của từng lần đo và thống nhất scope đo của các recognizer.

Xem [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md), [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md), [12 - Kết quả và giới hạn](12%20-%20K%E1%BA%BFt%20qu%E1%BA%A3%20v%C3%A0%20gi%E1%BB%9Bi%20h%E1%BA%A1n.md).
