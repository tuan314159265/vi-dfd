---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Codebase và triển khai

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Nội dung tương ứng Phụ lục A. Repo phục vụ tái lập nghiên cứu và trình bày pipeline triển khai. Hiện đây là **thiết kế repo**, chưa phải thông báo đã triển khai code.

## Cấu trúc dự kiến

```text
vi-dfd/
├── requirements.txt       # dependencies có version cụ thể
├── configs/
│   ├── teacher.yaml
│   ├── ti_ce.yaml
│   ├── ti_kd.yaml
│   ├── ti_factorized.yaml
│   ├── ti_dfd.yaml
│   └── ablations/          # YAML cho từng biến thể E3
├── data/                  # script, không chứa ảnh/crop VinText
│   ├── crop_vintext.py
│   ├── make_splits.py     # image-level + duplicate groups, fixed seed
│   └── subsets.py         # định nghĩa subset đóng băng trước train
├── dfd/
│   ├── vietchar.py        # bảng decompose/compose
│   ├── heads.py           # factorized head + valid composition
│   ├── distill.py         # marginal matrices và DFD loss
│   └── metrics.py         # word acc, NED, component ER, confusion
├── train.py
├── eval.py
├── bench/                 # ONNX export, INT8 và benchmark metadata
├── serve/
│   ├── app.py             # FastAPI POST /recognize
│   └── Dockerfile         # CPU-only demo
├── tests/                 # mapping round-trip và metric/loss cases cần thiết
├── results/
│   ├── runs.csv           # một dòng/một run
│   ├── metrics.csv        # một dòng/run/split/subset/precision
│   └── <run_id>/          # config, metadata, logs, predictions
└── README.md
```

## Requirements và YAML

- `requirements.txt` ghi version đã kiểm tra trong môi trường chạy; không để dependency “latest” không kiểm soát.
- Chốt cách cài PyTorch/CUDA theo GPU runtime; ghi exact environment của mỗi run.
- YAML lưu model, data, optimizer, training budget, seed và KD settings.
- Lưu resolved YAML sau các CLI overrides cùng run artifacts.
- Kết quả CSV bắt buộc có `run_id`, `seed`, `commit_hash` và config identity.

Schema chi tiết và YAML minh họa: [17 - Tái lập và phân chia GPU](17%20-%20T%C3%A1i%20l%E1%BA%ADp%20v%C3%A0%20ph%C3%A2n%20chia%20GPU.md).

## Attribution và dữ liệu

- Kiểm tra LICENSE/NOTICE PARSeq trước khi dùng.
- Ghi rõ code kế thừa PARSeq và phần tham khảo VietTDR trong source/README.
- README chỉ hướng dẫn tải VinText từ nguồn gốc; không đưa dataset/crop vào repo.

## Deployment demo — ngoài rubric

1. Export student ONNX.
2. Dynamic INT8, báo cáo accuracy loss và latency thực tế.
3. FastAPI `POST /recognize` nhận word crop, trả text và confidence từng component.
4. Docker CPU-only; README có một lệnh `docker run` để dùng demo.
5. Optional: ghép detector có sẵn làm end-to-end demo, tách khỏi benchmark cropped-word recognition.

## Tiêu chí bàn giao

- [ ] Có lệnh reproduce data preparation, train, eval và benchmark.
- [ ] Mapping round-trip và metric/loss sanity checks chạy được.
- [ ] Checkpoint/config/seed/commit tương ứng với các hàng kết quả.
- [ ] Run metadata ghi GPU và phiên bản runtime.
- [ ] Có sample input/output được phép phân phối.
- [ ] README mô tả limitations và attribution.

Xem [04 - Phân rã ký tự tiếng Việt](04%20-%20Ph%C3%A2n%20r%C3%A3%20k%C3%BD%20t%E1%BB%B1%20ti%E1%BA%BFng%20Vi%E1%BB%87t.md), [08 - Chỉ số và benchmark](08%20-%20Ch%E1%BB%89%20s%E1%BB%91%20v%C3%A0%20benchmark.md), [15 - Rubric và checklist nộp bài](15%20-%20Rubric%20v%C3%A0%20checklist%20n%E1%BB%99p%20b%C3%A0i.md).
