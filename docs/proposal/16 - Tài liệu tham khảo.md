---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Tài liệu tham khảo

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Danh mục dưới đây chép lại thông tin từ proposal. Các nhãn “paper”, “repo”, “PDF” trong bản dán không chứa URL thực nên chưa thể chuyển thành link chính xác. Chưa kiểm chứng độc lập nội dung hoặc bibliographic metadata; cần bổ sung trước khi nộp.

| ID | Reference theo bản gốc | Vai trò | Việc cần bổ sung |
|---|---|---|---|
| [1] | N. Nguyen et al. **Dictionary-Guided Scene Text Recognition**. CVPR 2021. VinText dataset | Benchmark chính, recognizer đi kèm | Paper, repo, data terms |
| [2] | D. Bautista, R. Atienza. **Scene Text Recognition with Permuted Autoregressive Sequence Models**. ECCV 2022. PARSeq | Teacher/student architecture | Paper/arXiv, repo, cấu hình, LICENSE/NOTICE |
| [3] | **VietTDR — Vietnamese Triple-Decomposition Recognizer**. Master's thesis code repository, GUET | Factorized head và ranh giới novelty | Tác giả, năm, thesis/repo URL, license |
| [4] | X. Wang et al. **Structural Knowledge Distillation: Tractably Distilling Information for Structured Predictor**. ACL 2021 | Nguyên lý marginal-based KD | PDF, tác giả/tiêu đề chính xác |
| [5] | G. Hinton, O. Vinyals, J. Dean. **Distilling the Knowledge in a Neural Network**. 2015 | Standard KD | Link, metadata/BibTeX |
| [6] | A. K. Bhunia et al. **Text is Text, No Matter What: Unifying Text Recognition using Knowledge Distillation**. ICCV 2021 | KD cho text recognition | Paper/arXiv |
| [7] | Z. Wang et al. **Symmetrical Linguistic Feature Distillation with CLIP for Scene Text Recognition**. ACM MM 2023 | Linguistic distillation cho STR | Paper/arXiv |
| [8] | W. Huang, S. Shi, Y. Wen. **Focusing on Diacritics to Improve Vietnamese Scene Text Detection**. IJCNN 2024. IEEE | Động lực diacritic-aware detection | DOI/paper |
| [9] | **VietOCR**, open-source Vietnamese OCR library, pbcquoc | Practical Vietnamese baseline | Repo gốc, pretrained data/weights, license; bản gốc nhắc README mirror |
| [10] | **A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions**. 2025. arXiv | Tổng quan OCR tiếng Việt | Tác giả, arXiv URL, nội dung claim |
| [11] | **Ensemble Learning for Vietnamese Scene Text Spotting in Urban Environments**. 2024. arXiv | Nguồn bổ sung thống kê split VinText | Tác giả, arXiv URL, đối chiếu [1] |
| [12] | V. H. Huynh, T. T. T. Nguyen, Q. T. Ngo. **Vietnamese Scene Text Detection Method based on Deep Learning**. Journal on Information Technologies & Communications, 2024. VnSceneText | Optional cross-dataset test | PDF, dataset access và terms |
| [13] | X.-S. Vu et al. **MC-OCR Challenge: Mobile-Captured Image Document Recognition for Vietnamese Receipts**. RIVF 2021 | Optional receipt test | Preprint, challenge site, license |

## Checklist tra cứu

- [ ] Ưu tiên paper, official repo, dataset release và challenge site gốc.
- [ ] Kiểm chứng mô tả VietTDR trước khi xác lập phần kế thừa và novelty.
- [ ] Đối chiếu PARSeq checkpoint/config và số tham số.
- [ ] Xác nhận vocabulary, split statistics và license VinText.
- [ ] Kiểm tra số ảnh và quyền truy cập VnSceneText/MC-OCR.
- [ ] Tạo BibTeX thống nhất, thêm tác giả và URL còn thiếu.

Xem [02 - Nghiên cứu liên quan](02%20-%20Nghi%C3%AAn%20c%E1%BB%A9u%20li%C3%AAn%20quan.md), [07 - Dữ liệu và protocol](07%20-%20D%E1%BB%AF%20li%E1%BB%87u%20v%C3%A0%20protocol.md).
