---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Kế hoạch thí nghiệm E1–E7

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Ma trận tổng thể

| ID | Câu hỏi | Models | Seeds theo proposal | Đầu ra |
|---|---|---|---|---|
| E1 | Baseline và teacher đủ tốt chưa? | Vanilla Ti, teacher, VietOCR, CRNN | 1 | Bảng baseline |
| E2 | DFD hơn KD thường không? RQ2 | DFD + ba baseline Ti chính | 3 sau tuning | Main results |
| E3 | Thành phần nào tạo ra cải thiện? | Các biến thể DFD | 3 hoặc 1 | [Bảng ablation](11%20-%20Ablation%20E3.md) |
| E4 | Nén làm mất gì? RQ1 | Teacher, vanilla Ti, DFD × FP32/INT8 | 3 trong bảng gốc | Component ER, CI, confusion |
| E5 | Generalization | E2 models | 1–3 | Kết quả subset/data fraction |
| E6 | Student có rẻ hơn không? | Tất cả models | Benchmark | Accuracy-vs-latency |
| E7 | Thất bại ở đâu, vì sao? | E2 models | Không huấn luyện riêng | Error categories/examples |

Tất cả dùng chung data pipeline, split và evaluation script. Các student chính giữ cùng epoch budget. Với E4 cần chốt dùng một teacher hay ba teacher: bảng gốc ghi 3 seed nhưng phần training chỉ huấn luyện teacher một seed.

## E1 — Baseline credibility

**Vì sao:** Cải thiện trên baseline yếu không đủ thuyết phục; teacher phải đủ tốt để KD có ý nghĩa.

1. Fine-tune PARSeq-S từ official weights, dùng head tiếng Việt.
2. Train vanilla Ti cùng recipe; chốt epochs, batch size và learning rate sau giai đoạn này.
3. Fine-tune VietOCR; train CRNN trên cùng split.
4. Đánh giá validation bằng script chung.

**Báo cáo:** Word accuracy full/diacritic-heavy, 1-NED, params.

**Decision rule của proposal:** Nếu teacher không tốt hơn rõ ràng cả Ti và VietOCR trên validation, bổ sung synthetic pretraining trước E2. Cần định nghĩa “rõ ràng” và công bố lại ngân sách pretraining.

## E2 — Main comparison / RQ2

1. Tuning temperature, lambda và tone weight trên validation với một seed; bảo đảm ngân sách tuning ngang nhau cho standard KD.
2. Chạy bốn student models bằng ba seed, cùng data order theo seed.
3. Đánh giá test và unseen test sau khi chọn cấu hình.

**Báo cáo:** Mean ± std và per-seed paired differences với từng baseline.

**Decision rule gốc:** Chỉ claim improvement khi mean gain lớn hơn một standard deviation và cùng dấu ở cả ba seed; nếu không, ghi inconclusive. Cần chốt std của paired differences hay std của model scores trước khi chạy. Quy tắc này là heuristic của proposal, không tự động tương đương significance test.

## E4 — Compression analysis / RQ1

1. Export teacher, vanilla Ti, DFD sang ONNX FP32 và dynamic INT8.
2. Tính base/modifier/tone ER cho mọi model/precision.
3. Theo bản gốc, tính relative increase so với teacher:

$$\Delta ER_x=\frac{ER_{model,x}-ER_{teacher,x}}{ER_{teacher,x}}.$$

4. Bootstrap test words để ước lượng 95% CI; lập tone confusion matrices.

**Báo cáo:** Component ER table, relative-increase chart, teacher/vanilla/DFD confusion.

**Decision rule gốc:** RQ1 “yes” nếu tone relative increase lớn hơn base với CI không chồng nhau, ở cả downsizing và quantization.

**Cần chốt để tách hai tác động:** Downsizing so teacher FP32 → Ti FP32; quantization so FP32 → INT8 của **cùng kiến trúc/checkpoint**. Không dùng riêng teacher FP32 làm reference cho mọi phép so sánh rồi gọi tất cả là tác động lượng tử hóa. Xử lý trường hợp denominator ER bằng 0 và lưu cả absolute change.

## E5 — Generalization

1. Đánh giá frozen subsets: unseen, crop height < 16 px, rare combination < 20 training occurrences.
2. Train vanilla Ti và DFD với 25% và 50% training images, một seed mỗi cấu hình; validation/test giữ nguyên.
3. Nếu được cấp quyền, test VnSceneText không fine-tuning.
4. MC-OCR là optional out-of-domain test, không phải hạng mục bắt buộc đã cam kết.

**Báo cáo:** Word accuracy theo subset và data fraction.

## E6 — Efficiency

Theo [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md): ONNX Runtime, batch 1, bốn CPU threads, 50 warm-up, median 500 runs, cùng máy/cùng phiên. Đo FP32/INT8; thêm GPU throughput batch 64 và peak memory.

**Báo cáo:** Params, latency, memory, accuracy và accuracy-vs-latency plot.

Training có thể chia qua nhiều tài khoản GPU; **latency comparison vẫn đo tập trung** trên phần cứng thống nhất.

## E7 — Error analysis

Mỗi E2 model lấy ngẫu nhiên 20 lỗi trên **validation**, gán nhãn thủ công: mark lost, wrong tone, wrong base, crop problem, label error. Chọn ví dụ thành công/thất bại tiêu biểu.

**Báo cáo:** Bảng nhóm lỗi và figure minh họa; ghi rõ mẫu 20 lỗi/model là phân tích định tính.

## Thứ tự triển khai

Dữ liệu/metrics → E1 → tuning E2 → E2 ba seed → E3 ưu tiên → E4/E6 → E5/E7 → tổng hợp báo cáo. Theo dõi lịch ở [14 - Tiến độ và rủi ro](14%20-%20Ti%E1%BA%BFn%20%C4%91%E1%BB%99%20v%C3%A0%20r%E1%BB%A7i%20ro.md) và giao run ở [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).
