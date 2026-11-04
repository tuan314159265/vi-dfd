---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Tiến độ và rủi ro

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Tương ứng Phụ lục B, nội bộ; không đưa nguyên phần kế hoạch vào bài nộp.

## Các mốc trong proposal

| Mốc | Điều kiện cần đạt |
|---|---|
| Tuần 1 | Chốt novelty với giảng viên; GPU, thành viên, phân công; freeze split/subsets |
| **13/11/2026** | Baseline chạy được |
| **27/11/2026** | Đóng băng thí nghiệm để dành gần hai tuần viết/review |
| **10/12/2026** | Hoàn tất nội bộ |
| **13/12/2026** | Hạn nộp chính thức; có ba ngày buffer |

Nếu mốc 13/11 trượt, cắt bớt cấu hình E3 trước, giữ thời gian viết.

## Phương án compute đã trao đổi

- Mỗi người ưu tiên tài khoản **Colab Free và Kaggle** để chạy các run được giao.
- **AWS là phương án dự phòng** nếu tài nguyên miễn phí không đủ hoặc không kịp tiến độ.
- Ghi GPU thực tế cho từng phiên; không giả định mọi tài khoản được cùng model GPU hoặc cùng thời lượng.
- Chia job theo experiment/model/config/seed; thống nhất checkpoint và đường dẫn artifact để có thể resume.
- Benchmark CPU latency thực hiện tập trung trên cùng máy theo protocol.

Đây là phương án sử dụng tài nguyên, chưa có thông tin tài khoản cụ thể, quota đã kiểm tra hay AWS budget. Không có tài nguyên AWS nào được tạo trong đợt tổ chức note này.

## Risk register

| Rủi ro | Dấu hiệu sớm | Phương án dự phòng |
|---|---|---|
| DFD không hơn KD hoặc head không KD | E2 gain nhỏ hơn std | Báo cáo kết quả âm; nhấn phân tích compression và trade-off |
| Teacher kém VietOCR | E1 validation | Synthetic pretraining hoặc teacher lớn hơn; ghi scope/budget thay đổi |
| Tổ hợp hiếm thiếu dữ liệu | EDA tuần 1 | SynthTIGER 0.5–1M crops, ưu tiên tổ hợp hiếm |
| Thiếu GPU | Run > 6 giờ hoặc mất thời gian chờ | Chia qua Colab/Kaggle; giảm E3 phụ; AWS dự phòng; xin hỗ trợ giảng viên nếu cần |
| Novelty bị xem là trùng VietTDR | Phản hồi giảng viên | Trao đổi sớm; nhấn distillation và compression analysis |
| Vi phạm license VinText | Chuẩn bị data/repo | Chỉ script tải/xử lý; không redistribute ảnh/crop; dẫn nguồn |
| Phiên notebook bị ngắt | Run dài, checkpoint chưa lưu | Checkpoint định kỳ, lưu optimizer/scheduler/RNG state để resume |

## Cần chốt

- [ ] Tên thành viên thực tế; proposal đang dự kiến ba người.
- [ ] GPU/tài khoản mỗi người đã có, dung lượng lưu trữ chung và cách bàn giao artifacts.
- [ ] Mức chi AWS dự phòng và người phụ trách khi cần dùng.
- [ ] Giảng viên chấp nhận ranh giới đóng góp.
- [ ] Người phụ trách data, mô hình/loss, evaluation/report và reviewer chéo.
- [ ] Resolve vocabulary/EOS và tuning-budget ambiguities trước training.

## Khung phân công — chưa gán tên

| Luồng | Công việc | Người phụ trách |
|---|---|---|
| Data/protocol | Crop, split, EDA, subset, mapping | Chưa chốt |
| Model/training | Teacher, Ti, KD, DFD, ablation | Chưa chốt |
| Evaluation/report | Metrics, quantization, benchmark, results, viết | Chưa chốt |

Training jobs có thể chia cho cả nhóm độc lập với người sở hữu module code. Xem [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).
