---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Rubric và checklist nộp bài

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Tương ứng Phụ lục C. Bảng rubric 40 điểm dưới đây được giữ theo nội dung người dùng cung cấp; chưa có file đề bài chính thức để đối chiếu độc lập.

| Hạng mục | Điểm | Note đáp ứng | Cần chú ý |
|---|---|---|---|
| Problem & motivation | 4 | [01 - Bài toán và đóng góp](01%20-%20B%C3%A0i%20to%C3%A1n%20v%C3%A0%20%C4%91%C3%B3ng%20g%C3%B3p.md), [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md) | Giải thích độ khó do dấu và deployment constraints |
| Contribution | 6 | [01 - Bài toán và đóng góp](01%20-%20B%C3%A0i%20to%C3%A1n%20v%C3%A0%20%C4%91%C3%B3ng%20g%C3%B3p.md), [05 - Factorized output head](05%20-%20Factorized%20output%20head.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md) | Tách DFD khỏi lợi ích head; không nhận lại novelty của VietTDR/Structural KD |
| Technical soundness | 6 | [03 - PARSeq và cấu hình mô hình](03%20-%20PARSeq%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20m%C3%B4%20h%C3%ACnh.md) đến [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md) | Teacher/student cùng permutation, augmentation và context |
| Protocol & baselines | 5 | [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md), [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md) | Cùng evaluation script, split và tuning budget |
| Evidence & generalization | 7 | [10 - Kế hoạch thí nghiệm E1-E7](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md), [11 - Ablation E3](11%20-%20Ablation%20E3.md) | Ba seed, frozen subsets, không chọn lọc kết quả |
| Analysis | 4 | E4/E7, [12 - Kết quả và giới hạn](12%20-%20K%E1%BA%BFt%20qu%E1%BA%A3%20v%C3%A0%20gi%E1%BB%9Bi%20h%E1%BA%A1n.md) | CI, confusion, lỗi và cả thí nghiệm thất bại |
| Report quality | 5 | Proposal → bản báo cáo | Tiếng Anh, định dạng LNCS, ghi chú dùng AI |
| Code presentation | 3 | [13 - Codebase và triển khai](13%20-%20Codebase%20v%C3%A0%20tri%E1%BB%83n%20khai.md), [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md) | Reproducibility, attribution, không kèm data bị cấm phân phối |
| **Tổng** | **40** | | |

Điểm yếu tiềm năng nhất của proposal là Contribution nếu E2 không tách được hiệu ứng marginal KD khỏi factorized head.

## Nội dung đưa vào bài nộp

- [ ] Abstract, introduction và RQ.
- [ ] Related work với nguồn đã kiểm tra và bibliographic metadata đầy đủ.
- [ ] Phương pháp, công thức, decomposition và EOS rõ ràng.
- [ ] Dataset/protocol, baseline, configuration và hardware.
- [ ] Kết quả thật của E2/E4, ablation và generalization trong scope.
- [ ] Error analysis, limitations và conclusion bám evidence.
- [ ] LNCS, English proofreading, figure/table captions và AI-use note.

## Phần nội bộ tách khỏi bài nộp

Cấu trúc vận hành repo, deployment phục vụ CV, lịch/rủi ro và bảng đối chiếu rubric tương ứng Phụ lục A–C không đưa nguyên văn vào proposal nộp. Vẫn dẫn code/reproducibility cần thiết trong báo cáo.

## Trước khi bàn giao

- [ ] Các số trong paper khớp CSV và commit hash.
- [ ] Từng seed có config, checkpoint và metadata.
- [ ] Không có placeholder thành viên, reference hoặc `[TO BE UPDATED]` bị bỏ quên.
- [ ] Không trình bày indicative one-seed results như bằng chứng ba seed.
- [ ] Kiểm tra license/attribution.
- [ ] Giữ mốc nội bộ 10/12/2026.
