# Kết quả đợt 2 — Tuấn

Đã hoàn tất 6/6 run train và eval trên tag `exp-v2`. Các số dưới đây là word accuracy (%), lấy từ metrics.csv, subset all. Chưa tổng hợp cùng kết quả DFD của các thành viên nên chưa kết luận phương pháp nào tốt hơn.

| Config | Test (%) | Unseen test (%) | Run ID |
|---|---|---|---|
| `configs/runs/E2/ti_kd_s43.yaml` | 83.2384 | 85.7966 | `20261123T093321Z_ti_kd_s43_3971b310` |
| `configs/runs/E2/ti_kd_s44.yaml` | 82.6968 | 85.1112 | `20261123T100549Z_ti_kd_s44_71c76171` |
| `configs/runs/E3/joint_kd_s42.yaml` | 82.9190 | 86.1343 | `20261123T103930Z_joint_kd_s42_ac974450` |
| `configs/runs/E3/joint_kd_s43.yaml` | 83.4329 | 86.2733 | `20261123T111632Z_joint_kd_s43_1d02f360` |
| `configs/runs/E3/joint_kd_s44.yaml` | 82.9607 | 86.3627 | `20261123T115518Z_joint_kd_s44_032c8759` |
| `configs/runs/E3/marginal_joint_kd_s42.yaml` | 82.7663 | 85.9356 | `20261123T122254Z_marginal_joint_kd_s42_185f0a9c` |

## Artifact và việc còn lại

- Local: `../phase2_completed/<run_id>/`; 6 thư mục chứa checkpoint, config, metadata, validation.csv, metrics, dự đoán và thông tin nguồn dữ liệu.
- Chưa xác nhận upload Drive hoặc cập nhật sheet nhóm. Cần bàn giao nguyên 6 thư mục vào `runs/<run_id>/` trên Drive private.
- KD seed 42 là trial thắng tuning, cần chấm test/unseen_test để đủ bộ ba seed cho E2; không train lại. Các bảng cuối do Vinh tổng hợp theo protocol.
- E1 CRNN/VietOCR mới có validation; phần export checkpoint để chấm test/unseen_test còn thiếu. ONNX và E6 latency chưa hoàn tất.
- Không commit artifact, checkpoint, nhãn hoặc dự đoán vào Git.

## KD seed 42 đã chấm lại (24/11)

Checkpoint thắng tuning, không train lại: test **82.2802%**, unseen **85.5085%** (xem metrics.csv của run 84e95e78 để lấy số chính xác). Đã chạy eval.py --replace trên validation/test/unseen với metrics quyết định 4; cần upload bản artifact cập nhật.
