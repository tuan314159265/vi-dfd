---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Vi-DFD — BTL Deep Learning

**Diacritic-Factorized Distillation for Lightweight Vietnamese Scene Text Recognition**

- **Trạng thái:** proposal; chưa có kết quả thực nghiệm.
- **Bản gốc:** 30/10/2026; chuyển thành bộ note ngày 04/11/2026 từ nội dung người dùng cung cấp.
- **Tác giả trong bản gốc:** Vinh Cong Doan — HCMUT, `vinh.doan2005@hcmut.edu.vn`; thành viên 2 và 3 chưa điền.
- **Nguồn:** [Claude artifact](https://claude.ai/artifact/BsU3AYCwZ74quXDJWAA5gV#26c8be8f-0247). Nội dung được lấy từ bản dán trong hội thoại.
- **Hạn nội bộ:** 10/12/2026. **Hạn nộp chính thức:** 13/12/2026.

> [!info] Cách sử dụng
> Đây là bản tổ chức lại proposal để học và triển khai trong Obsidian. Các con số kiến trúc, thống kê dữ liệu, mô tả công trình và điều khoản sử dụng được giữ theo bản gốc, chưa kiểm chứng độc lập. Kết quả, kết luận và thông tin còn thiếu đều để trạng thái chờ.

> Repository snapshot: exported from the reorganized Obsidian notes; not a verbatim export of the Claude artifact.

## Ý tưởng trong một đoạn

Nghiên cứu xem việc thu nhỏ hoặc lượng tử hóa mô hình nhận dạng chữ trong ảnh tiếng Việt có làm lỗi dấu thanh tăng mạnh hơn lỗi chữ gốc hay không. DFD cộng xác suất của teacher PARSeq-S thành ba phân phối **chữ gốc, dấu biến âm, dấu thanh**, rồi dùng chúng để giám sát student PARSeq-Ti có ba head tương ứng. So sánh với CE, KD thường và head phân rã không KD trên cùng ngân sách; báo cáo độ chính xác, lỗi từng thành phần và độ trễ CPU.

## Bản đồ note

| Note | Nội dung |
|---|---|
| [01 - Bài toán và đóng góp](01%20-%20B%C3%A0i%20to%C3%A1n%20v%C3%A0%20%C4%91%C3%B3ng%20g%C3%B3p.md) | Abstract, động lực, RQ1/RQ2 và ranh giới novelty |
| [02 - Nghiên cứu liên quan](02%20-%20Nghi%C3%AAn%20c%E1%BB%A9u%20li%C3%AAn%20quan.md) | Vietnamese STR, PARSeq, VietTDR, KD và research gap |
| [03 - PARSeq và cấu hình mô hình](03%20-%20PARSeq%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20m%C3%B4%20h%C3%ACnh.md) | Kiến trúc teacher/student, phần giữ nguyên và huấn luyện |
| [04 - Phân rã ký tự tiếng Việt](04%20-%20Ph%C3%A2n%20r%C3%A3%20k%C3%BD%20t%E1%BB%B1%20ti%E1%BA%BFng%20Vi%E1%BB%87t.md) | Bảng base–modifier–tone, vocabulary, ma trận ánh xạ |
| [05 - Factorized output head](05%20-%20Factorized%20output%20head.md) | Ba head, ghép xác suất, valid-character mask |
| [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md) | Marginal distillation, temperature và tone weighting |
| [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md) | Dataset, crop, augmentation, split, chống leakage |
| [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md) | Word accuracy, component ER, alignment, latency |
| [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md) | Các đối chứng, tuning và reproducibility |
| [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md) | Mục tiêu, cách chạy, đầu ra và decision rules |
| [11 - Ablation E3](11%20-%20Ablation%20E3.md) | Các biến thể tách hiệu ứng của head, KD và weighting |
| [12 - Kết quả và giới hạn](12%20-%20K%E1%BA%BFt%20qu%E1%BA%A3%20v%C3%A0%20gi%E1%BB%9Bi%20h%E1%BA%A1n.md) | Mẫu bảng kết quả, conclusion và limitations |
| [13 - Codebase và triển khai](13%20-%20Codebase%20v%C3%A0%20tri%E1%BB%83n%20khai.md) | Cấu trúc repo, ONNX, API, Docker, attribution |
| [14 - Tiến độ và rủi ro](14%20-%20Ti%E1%BA%BFn%20%C4%91%E1%BB%99%20v%C3%A0%20r%E1%BB%A7i%20ro.md) | Mốc chặn, việc cần chốt, phương án dự phòng |
| [15 - Rubric và checklist nộp bài](15%20-%20Rubric%20v%C3%A0%20checklist%20n%E1%BB%99p%20b%C3%A0i.md) | Đối chiếu 40 điểm, báo cáo và code |
| [16 - Tài liệu tham khảo](16%20-%20T%C3%A0i%20li%E1%BB%87u%20tham%20kh%E1%BA%A3o.md) | Danh mục [1]–[13], trạng thái liên kết |
| [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md) | Requirements, YAML, CSV/commit hash, seed và Colab/Kaggle/AWS |

## Luồng đọc

1. Nắm bài toán ở [01 - Bài toán và đóng góp](01%20-%20B%C3%A0i%20to%C3%A1n%20v%C3%A0%20%C4%91%C3%B3ng%20g%C3%B3p.md).
2. Đọc [03 - PARSeq và cấu hình mô hình](03%20-%20PARSeq%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20m%C3%B4%20h%C3%ACnh.md) → [04 - Phân rã ký tự tiếng Việt](04%20-%20Ph%C3%A2n%20r%C3%A3%20k%C3%BD%20t%E1%BB%B1%20ti%E1%BA%BFng%20Vi%E1%BB%87t.md) → [05 - Factorized output head](05%20-%20Factorized%20output%20head.md) → [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md).
3. Chốt [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md), [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md) và [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md) trước khi chạy.
4. Theo dõi [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md), [11 - Ablation E3](11%20-%20Ablation%20E3.md) và điền [12 - Kết quả và giới hạn](12%20-%20K%E1%BA%BFt%20qu%E1%BA%A3%20v%C3%A0%20gi%E1%BB%9Bi%20h%E1%BA%A1n.md).

## Việc cần làm trước

- [x] Chọn phương án compute: mỗi người ưu tiên Colab Free + Kaggle, AWS dự phòng.
- [ ] Chốt thành viên, tài khoản/tài nguyên thực tế và phân công run.
- [ ] Trao đổi với giảng viên về ranh giới đóng góp.
- [ ] Kiểm tra nguồn, license và vocabulary chính xác.
- [ ] Đóng băng split và các subset trước khi huấn luyện.
- [ ] Làm cho baseline chạy được trước 13/11.
