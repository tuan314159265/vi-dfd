---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Dữ liệu và protocol

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Dataset theo proposal

| Dataset | Vai trò | Quy mô ghi trong bản gốc | Trạng thái/điều khoản ghi trong bản gốc |
|---|---|---|---|
| VinText [1,11] | Train, validation, test chính | 2,000 ảnh, khoảng 56K text instances; 1,200 train / 300 test / 500 unseen test | Research/education; không phân phối lại; phải trích dẫn |
| VnSceneText [12] | Cross-dataset test | Khoảng 3,000 ảnh scene tiếng Việt | Cần xin tác giả |
| Synthetic Vietnamese words | Optional pretraining, bổ sung tổ hợp hiếm | 0.5–1M crop tự sinh | Dữ liệu nhóm tạo |
| MC-OCR 2021 [13] | Optional out-of-domain test hóa đơn | 2,436 ảnh | Cần kiểm tra điều khoản trên challenge site |

- [ ] Xác nhận số lượng khả dụng, split và license từ nguồn gốc trước khi tải/chia sẻ.
- [ ] Kiểm tra license font và tài nguyên dùng sinh dữ liệu.

## Tiền xử lý

1. Chuẩn hóa nhãn về Unicode NFC và một quy ước đặt dấu thanh.
2. Perspective warp từ polygon bốn điểm; giữ khoảng 10% margin trên/dưới để tránh mất dấu.
3. Bỏ nhãn illegible `###`.
4. Resize 32 × 128 và normalize theo PARSeq.
5. Dùng RandAugment của PARSeq; tránh vertical flip và vertical crop mạnh có thể làm mất/thay dấu nhưng giữ nhãn cũ.

## Split và chống leakage

- Split theo **ảnh**, không theo word crop; mọi crop của một ảnh ở cùng split.
- Validation = 10% ảnh thuộc official training set, fixed seed.
- Gom nhóm ảnh gần trùng của cùng biển hiệu bằng perceptual hash; nhóm nằm trong một split.
- Thống kê tần suất tổ hợp chỉ trên training split.
- Chọn checkpoint và hyperparameters chỉ bằng validation.
- Test và unseen test dùng cho đánh giá cuối, không dùng tuning.
- Đóng băng định nghĩa các subset ở tuần 1, trước training.

Nếu phát hiện duplicate group cắt qua official split, cần ghi rõ cách xử lý và phần nào khác official protocol; không âm thầm đổi split.

**Đã chốt:** cặp `im0894` (train) và `im1670` (unseen test), pHash distance 4, giữ nguyên split chính thức và ghi vào phần giới hạn của báo cáo.

## Subset dự kiến

| Subset | Định nghĩa trong proposal |
|---|---|
| Diacritic-heavy | Từ có ít nhất 2 ký tự mang dấu |
| Small crop | Chiều cao crop gốc < 16 px |
| Rare combination | Tổ hợp xuất hiện < 20 lần trên training split |
| Unseen test | Split riêng của VinText |

- [x] “Mang dấu” gồm modifier, tone hoặc cả hai, và tính cả `đ/Đ`.
- [x] Rare combination theo (base không phân biệt hoa/thường, modifier, tone), khớp cách chấm không phân biệt hoa/thường. Từ thuộc subset nếu chứa ít nhất một tổ hợp hiếm.
- [x] Height là `original_word_height`, đo trên tứ giác gốc, trước margin và trước resize.
- [x] Giữ nguyên vị trí dấu thanh như nhãn gốc (chỉ NFC), không viết lại `hoà/hòa`, vì nhãn phải khớp chữ trong ảnh.

## Synthetic pretraining

Có thể dùng SynthTIGER tạo 0.5–1M crop tiếng Việt, tập trung tổ hợp hiếm. Nếu phải bổ sung pretraining sau E1, áp dụng và ghi rõ ngân sách tương ứng cho các baseline để giữ so sánh công bằng.

Xem [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md), [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md), [14 - Tiến độ và rủi ro](14%20-%20Ti%E1%BA%BFn%20%C4%91%E1%BB%99%20v%C3%A0%20r%E1%BB%A7i%20ro.md).
