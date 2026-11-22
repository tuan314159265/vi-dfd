# Kết quả E2 tuning KD thường — Tuấn

Đã kiểm tra ngày 20/11/2026: **12/12 trial hoàn tất train và eval validation**, seed 42, cùng commit `97604d777c509381eb4118564368b18a8504d4d6` (`exp-v1`). Đây là kết quả tuning, chưa phải kết luận so sánh DFD với KD.

- Dữ liệu: bản VinText của Tiến; checksum freeze manifest, LMDB train/val và LMDB dùng chấm validation khớp `configs/data_reference.json`.
- Mỗi trial có metadata `completed`, queue `done`, một checkpoint best và một dòng validation/all cho đủ 2.507 mẫu.
- Không dùng test để chọn siêu tham số.

| τ | λ | Validation word acc (%) | 1-NED | Run ID |
|---|---|---|---|---|
| 1 | 0.25 | 87.5150 | 0.940634 | `20261120T093402Z_ti_kd_s42_65165ca8` |
| 1 | 0.5 | 87.5548 | 0.941618 | `20261120T100917Z_ti_kd_s42_2d6d295b` |
| 1 | 1 | 87.7543 | 0.942282 | `20261120T103614Z_ti_kd_s42_fce6603e` |
| 1 | 2 | 87.7942 | 0.942941 | `20261120T110004Z_ti_kd_s42_60c10092` |
| 2 | 0.25 | 87.6745 | 0.941790 | `20261120T112406Z_ti_kd_s42_6d981171` |
| 2 | 0.5 | 87.6745 | 0.941733 | `20261120T135447Z_ti_kd_s42_4f8bd0a4` |
| 2 | 1 | 87.8341 | 0.943053 | `20261117T194943Z_ti_kd_s42_84e95e78` |
| 2 | 2 | 87.7543 | 0.942128 | `20261120T120100Z_ti_kd_s42_10d88ef9` |
| 4 | 0.25 | 87.5150 | 0.940398 | `20261120T122302Z_ti_kd_s42_99f03520` |
| 4 | 0.5 | 87.6745 | 0.941219 | `20261120T124126Z_ti_kd_s42_627a9d74` |
| 4 | 1 | 87.7144 | 0.942917 | `20261120T125912Z_ti_kd_s42_6e805b1d` |
| 4 | 2 | 87.5947 | 0.941843 | `20261120T131650Z_ti_kd_s42_98cb7d00` |

## Lựa chọn sơ bộ

Theo word accuracy validation rồi 1-NED, tốt nhất trong 12 trial KD là **τ=2, λ=1**, word accuracy **87.8341%**. Vinh đã tổng hợp 24 trial và khóa tại PR #4 ngày 21/11. Tag `exp-v2` đã có: KD τ=2, λ=1; DFD τ=1, λ=1, w_t=2. Các lựa chọn KD trong bảng này vẫn giữ nguyên.

## Bàn giao artifact

- Bản hoàn chỉnh local: `../kd_completed/<run_id>/` (ngoài repo).
- Cần chuyển các thư mục run này vào `runs/<run_id>/` trên Drive private nhóm: gồm checkpoint best, `resolved.yaml`, `metadata.json`, `metrics.csv`, `data_identity.json`, `eval_info.json`, `queue.json`, dự đoán và log đi kèm.
- Chưa xác nhận upload Drive; push note này không thay thế bàn giao artifact.
- Không commit checkpoint, ảnh, LMDB, manifest chứa nhãn hoặc dự đoán vào Git.
- Run đầu được rescore bằng runner mới; số dùng ở bảng là `eval.py` chính thức (87,8341%), không lấy 87,87395% trong log validation lúc train.

## Tiếp theo

1. Upload artifact, cập nhật sheet theo dõi.
2. Đã xong: Vinh gom KD và DFD, khóa siêu tham số tại PR #4 và đặt `exp-v2`.
3. Tuấn chạy KD seed 43/44 và E3 joint KD seed 42/43/44, marginal + joint seed 42 từ `exp-v2`. E1 VietOCR/CRNN và phần xuất ONNX/benchmark vẫn là công việc riêng.

## Cập nhật sau PR #5

Main đã cập nhật metrics theo quyết định 4: invalid marker U+FFFD, mẫu số component ER theo ký tự nhãn, insert/delete báo riêng. Word accuracy và cấu hình thắng không đổi. Các giá trị 1-NED/ER ở artifact exp-v1 chưa được xác nhận chấm lại bằng metrics mới, nên không dùng chúng làm bảng cuối trước khi re-evaluate. Upload Drive vẫn chưa được xác minh từ máy này.
