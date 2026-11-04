---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Tái lập và phân chia GPU

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Yêu cầu đã trao đổi

Có cấu trúc repo, `requirements.txt`, cấu hình YAML, log CSV kèm commit hash và seed cố định. Mỗi người ưu tiên **Colab Free + Kaggle** để chạy các job được giao; **AWS dự phòng** nếu compute miễn phí không đủ hoặc không kịp tiến độ.

Các ví dụ bên dưới là quy ước thiết kế cho codebase, **chưa phải các script/config đã được triển khai và chạy kiểm chứng**.

## Seed cố định

Quy ước dùng `42, 43, 44` cho các cấu hình ba seed; run một seed và tuning dùng `42`. Lưu seed trong YAML và CSV, không chọn lại seed theo kết quả.

Khi code training, seed Python `random`, NumPy, PyTorch CPU/CUDA, DataLoader generator và workers. Resume cần khôi phục RNG, sampler, optimizer/scheduler và AMP scaler state, ngoài model weights.

Cùng seed không bảo đảm bitwise-identical giữa các GPU/runtime khác nhau; cần ghi environment để giải thích khác biệt.

## Requirements

- Có `requirements.txt` với version đã kiểm chứng; chỉ chốt số version sau khi chọn môi trường chạy.
- Nhóm dependency dự kiến: PyTorch/torchvision, NumPy, Pillow/OpenCV, YAML/config loader, ONNX/ONNX Runtime, thư viện edit distance; FastAPI/Uvicorn cho demo.
- Ghi Python, Torch, CUDA, driver, GPU và platform của từng run.
- Lưu environment snapshot cùng artifacts; ghi rõ source commit PARSeq và checkpoint pretrained.
- Với Colab/Kaggle, giữ cell setup từ repo để mỗi người chạy cùng quy trình.

## YAML minh họa

Đây là schema đề xuất để thống nhất cách triển khai, chưa gắn vào parser cụ thể.

```yaml
experiment: E2
variant: ti_dfd
seed: 42

model:
  student: parseq_tiny
  teacher: parseq_small
  head: factorized
  vocab_path: artifacts/vocab.json
  teacher_checkpoint: artifacts/teacher/best.ckpt

data:
  split_manifest: artifacts/splits.json
  subset_manifest: artifacts/subsets.json
  image_size: [32, 128]
  normalization: NFC

training:
  epochs: null       # chốt sau E1; phải điền trước khi chạy
  batch_size: null   # chốt sau E1
  optimizer: AdamW
  lr: null           # chốt sau E1
  schedule: one_cycle
  amp: true

distillation:
  enabled: true
  temperature: 2.0   # giá trị minh họa; chọn bằng validation
  lambda: 1.0
  component_weights:
    base: 1.0
    modifier: 1.0
    tone: 2.0
  tone_confidence_weight: true

output:
  root: results
  save_predictions: true
  save_resolved_config: true
```

Mỗi run cần resolved config đầy đủ sau override; từ chối chạy nếu trường bắt buộc vẫn `null`. Hash config không chứa các thông tin chỉ đổi vì thư mục output hoặc timestamp nếu dùng để nhận diện cấu hình khoa học.

## CSV và run artifacts

`runs.csv` — một dòng/một run:

```csv
run_id,experiment,variant,seed,commit_hash,git_dirty,config_hash,data_hash,teacher_checkpoint_hash,platform,gpu,python_version,torch_version,cuda_version,start_utc,end_utc,status,checkpoint_path
```

`metrics.csv` — một dòng/run/split/subset/precision:

```csv
run_id,commit_hash,seed,split,subset,precision,num_words,word_acc,ned_score,base_er,modifier_er,tone_er,insertions,deletions,params,cpu_latency_ms,gpu_throughput,peak_memory_mb
```

- `commit_hash`: full Git SHA của code thực sự dùng cho run.
- `git_dirty`: nếu working tree có thay đổi, lưu patch tương ứng; chỉ commit hash không đủ mô tả code chưa commit.
- `data_hash`: fingerprint của vocabulary, split/subset manifests và dữ liệu dùng cho run.
- Thống nhất metric fractions hay percentages và đơn vị memory; định nghĩa trong README.
- Lưu predictions để tính lại metrics/CI mà không phải train lại.
- Lưu config, metadata, logs, checkpoints và predictions dưới `results/<run_id>/` hoặc artifact storage có liên kết rõ ràng.
- Các notebook export CSV riêng; một script tổng hợp kiểm tra schema/trùng run trước khi tạo bảng chung, tránh nhiều người ghi đồng thời cùng CSV.

## Chia GPU/job trong nhóm

“Riêng tài khoản” là mỗi người dùng tài khoản dịch vụ của mình để chạy phần việc được giao. Mỗi người có thể chuyển giữa Colab Free và Kaggle theo tài nguyên đang dùng được. Chưa có thông tin để khẳng định quota hoặc loại GPU được cấp.

**Đơn vị giao việc:** `experiment + variant + config_hash + seed`, không chỉ “chạy DFD”. Ví dụ:

| Job | Nơi chạy | Người chạy | Trạng thái |
|---|---|---|---|
| E1 / teacher / seed 42 | Colab hoặc Kaggle | Chưa gán | Chưa chạy |
| E2 / ti_ce / seed 42 | Colab hoặc Kaggle | Chưa gán | Chưa chạy |
| E2 / ti_kd / seed 42 | Colab hoặc Kaggle | Chưa gán | Chưa chạy |
| E2 / ti_factorized / seed 42 | Colab hoặc Kaggle | Chưa gán | Chưa chạy |
| E2 / ti_dfd / seed 42 | Colab hoặc Kaggle | Chưa gán | Chưa chạy |

Tạo thêm job cho seed 43/44 sau tuning. Một người có thể chạy nhiều model; ownership module code không quyết định bắt buộc nơi chạy model.

## Thứ tự chạy và bàn giao

1. Chốt code commit, dependencies, data manifests và run config chung.
2. E1 chạy teacher và baseline; lưu teacher checkpoint để mọi job KD dùng cùng teacher.
3. Gán job trước khi mở notebook, đánh dấu running để tránh trùng.
4. Checkpoint định kỳ vào nơi lưu bền qua phiên notebook.
5. Khi xong, bàn giao checkpoint, resolved YAML, metadata, CSV và predictions.
6. Đối chiếu completeness và hash trước khi aggregate results.
7. Đo CPU latency tập trung trên cùng máy; training trên nhiều GPU không thay protocol latency.

## Khi cần AWS

Ưu tiên chuyển các run còn thiếu hoặc dài sang AWS khi Colab/Kaggle không đủ đáp ứng mốc tiến độ. Trước lúc triển khai, chốt người vận hành, số giờ/mức chi và cơ chế dừng sau job. Hiện note chỉ ghi phương án dự phòng, chưa tạo AWS instance.

Liên quan: [09 - Baselines và cấu hình huấn luyện](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md), [13 - Codebase và triển khai](13%20-%20Codebase%20v%C3%A0%20tri%E1%BB%83n%20khai.md), [14 - Tiến độ và rủi ro](14%20-%20Ti%E1%BA%BFn%20%C4%91%E1%BB%99%20v%C3%A0%20r%E1%BB%A7i%20ro.md).
