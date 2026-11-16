# Kế hoạch chạy thí nghiệm E1–E7

Cập nhật 16/11/2026. Đóng băng thí nghiệm ngày **27/11/2026**. File này là nơi chốt chung: chạy thí nghiệm nào, chạy bằng lệnh gì, ai code, ai chạy. Mọi số liệu đưa vào báo cáo phải đến từ các run làm đúng theo file này.

## 1. Tóm tắt

| | Số run train mới | Ghi chú |
|---|---|---|
| E1 baseline | 2 | Teacher đã xong (Tuấn, 09/11). `ti_ce` dùng lại run `ti_ce` seed 42 của E2. Còn VietOCR và CRNN |
| E2 tuning | 24 | 12 trial DFD + 12 trial KD thường, seed 42, chọn bằng validation |
| E2 chạy chính | 10 | 4 student × 3 seed = 12. `ti_kd` và `ti_dfd` seed 42 dùng lại trial tốt nhất của tuning |
| E3 ablation | 8 | `no_validity_mask` và temperature sweep không cần train (xem mục 4) |
| E5 ít dữ liệu | 4 | `ti_ce`, `ti_dfd` với 25% và 50% dữ liệu train |
| E4, E6, E7 | 0 | Chỉ chấm điểm, xuất ONNX, đo độ trễ, phân tích lỗi trên checkpoint đã có |
| **Tổng** | **48** | |

**Không có thí nghiệm nào phải chạy 54 lần.** Con số 54 trước đây là **tổng số run train của cả bảy thí nghiệm cộng lại** (4 + 24 + 12 + 10 + 4). Tính kỹ hơn thì còn 48 run mới, vì teacher đã train xong và có 6 run trùng cấu hình với run khác nên dùng lại được.

**Thời gian:** teacher PARSeq-S train 20 epoch mất 29,6 phút trên RTX 3060 Laptop. Student PARSeq-Ti nhỏ hơn nhiều, nhưng run có KD phải chạy thêm teacher ở mỗi bước. Ước tính mỗi run student mất khoảng 20–40 phút, tức cả 48 run khoảng 20–30 giờ GPU, chia ba người còn khoảng 7–10 giờ mỗi người. Số chính xác chốt ở mốc 13 (mục 3, bước 0).

## 2. Hệ quy chiếu chung (phải giống nhau ở mọi run)

Hai run chỉ so sánh được với nhau khi các mục dưới đây trùng nhau. `train.py` và `eval.py` tự ghi các mục này vào thư mục run (`metadata.json`, `data_identity.json`, `eval_info.json`), nên lúc tổng hợp ai cũng kiểm tra lại được.

| Mục | Giá trị chốt | Kiểm tra bằng | Trạng thái |
|---|---|---|---|
| Dữ liệu | Bản VinText của Tiến gửi ngày 16/11 (`lmdb/` + `splits/`) | `scripts/check_data.py` so SHA-256 với `configs/data_reference.json` | Đề xuất, cần nhóm đồng ý |
| Teacher | `best-epoch=17-validation_word_acc=0.8867.ckpt` của Tuấn, SHA-256 `e185cec4…bd609` | `check_data.py --teacher-checkpoint` | Chờ Tuấn xác nhận split (mục 7) |
| Code | Một tag Git chung, đặt trên `main`: `exp-v1` cho đợt đầu, `exp-v2` sau khi khóa siêu tham số | `metadata.json` ghi commit, `train.py` từ chối chạy nếu working tree bẩn | Đặt tag sau khi merge file này |
| PARSeq upstream | Commit `1902db043c029a7e03a3818c616c06600af574be` | `dfd/parseq.py` tự kiểm tra | Chốt |
| Thư viện | `requirements-teacher.txt` (torch 2.5.1, Lightning 2.2.0.post0, …) | `environment.txt` trong mỗi run | Chốt |
| Recipe train | Batch 64, 20 epoch, AdamW + OneCycleLR, base LR 7e-4, warmup 0,075, clip 20, AMP | `configs/training_budget.yaml`, `resolved.yaml` | Chốt (09/11) |
| Seed | Tuning và run một seed: 42. Run ba seed: 42, 43, 44 | YAML | Chốt |
| Chọn checkpoint | `best` theo `validation_word_acc`, không bao giờ nhìn test | Callback trong `fit_parseq` | Chốt |
| Chọn siêu tham số | Word acc trên validation, hòa thì so 1-NED, còn hòa nữa thì lấy giá trị nhỏ hơn trong grid | `scripts/e2_tuning.py select` | Chốt |
| Chấm điểm | Chỉ dùng `eval.py`: NFC, bỏ dấu cách, không phân biệt hoa thường, chấm mọi mẫu (không lọc) | `eval_info.json` | Chốt |
| Model không phải PARSeq | Ghi file `source_id,prediction` rồi chấm bằng `eval.py --predictions-dir` | Như trên | Chốt |

Dữ liệu được giải nén **ngoài repo** để ảnh VinText không bao giờ lọt vào Git: license không cho phát tán lại, và chỉ cần một lần `git add -A` nhầm là ảnh lên GitHub. Mọi người vẫn dùng **cùng một bản** dữ liệu. Mỗi người tải bản của Tiến từ Drive, giải nén vào đâu cũng được, rồi truyền đường dẫn qua `--data-root`. Dữ liệu có giống nhau hay không được kiểm tra bằng SHA-256, không dựa vào việc để chung một thư mục. Trên Colab hay Kaggle đường dẫn đằng nào cũng khác máy cá nhân.

## 3. Cài đặt một lần (máy cá nhân, Colab, Kaggle)

```bash
git clone https://github.com/tuan314159265/vi-dfd.git && cd vi-dfd
git checkout exp-v1                      # hoặc tag đang dùng ở đợt đó
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-teacher.txt -r requirements-dev.txt
git clone https://github.com/baudm/parseq ~/.cache/vi-dfd/parseq
git -C ~/.cache/vi-dfd/parseq checkout 1902db043c029a7e03a3818c616c06600af574be

# Dữ liệu và teacher: lấy từ thư mục Drive private của nhóm, giải nén NGOÀI repo
DATA=/content/vintext_notebook           # chứa lmdb/ và splits/
TEACHER=/content/teacher_s42_e1/checkpoints/best-epoch=17-validation_word_acc=0.8867.ckpt
python scripts/check_data.py --data-root $DATA --teacher-checkpoint "$TEACHER"   # phải báo 27/27
python -m pytest -q                      # phải pass hết
```

Nếu `check_data.py` báo `DIFFERENT` hay `MISSING` thì **không chạy**, báo nhóm trước.

### Bước 0: mốc 13, đo thời gian (người có GPU, làm một lần)

```bash
python scripts/time_kd_step.py --batch-size 64 --train-samples 23267
```

Gửi kết quả (ms/step và thời gian một epoch của bốn biến thể) vào nhóm để chốt lịch.

## 4. Lệnh chạy từng thí nghiệm

Mọi lệnh chạy từ thư mục repo, trên đúng tag, working tree sạch. Mỗi lệnh `train.py` tự tạo thư mục `results/runs/<run_id>/`. Thư mục này nằm trong `.gitignore` nên không gây xung đột Git.

**Một run gồm 3 bước:**

```bash
# 1. Train. Thêm --teacher-checkpoint với run có KD (ti_kd, ti_dfd, E3, E5 ti_dfd)
python train.py --config <config.yaml> --data-root $DATA --teacher-checkpoint "$TEACHER"
#    In ra "Run directory: results/runs/<run_id>"

# 2. Chấm điểm checkpoint tốt nhất (split và subset lấy từ config)
RUN=results/runs/<run_id>
python eval.py --run-dir $RUN --checkpoint $RUN/checkpoints/best-*.ckpt --device cuda

# 3. Upload cả thư mục $RUN lên Drive nhóm: Vi-DFD/runs/<run_id>/
```

Run bị ngắt giữa chừng thì resume bằng `--resume $RUN/checkpoints/last.ckpt` với **đúng** config và dữ liệu cũ. Run resume không được tính là seed mới.

### E1: baseline và teacher

| Run | Lệnh / config | Ghi chú |
|---|---|---|
| Teacher | Đã xong. Chấm lại chính thức: `eval.py --config configs/runs/E1/teacher_s42.yaml --checkpoint "$TEACHER" --data-root $DATA/lmdb --subset-dir $DATA/splits/subsets --splits validation test unseen_test` | Số đối chiếu đã có trong `docs/PROGRESS_VINH.md` |
| `ti_ce` | Dùng lại run E2 `ti_ce_s42`, chấm thêm `--splits validation` | Không train riêng |
| VietOCR, CRNN | Chưa có code train. Xuất dự đoán ra `<dir>/<split>.csv`, rồi `eval.py --config configs/runs/E1/vietocr_s42.yaml --predictions-dir <dir> ...` | Task 12 |

### E2 giai đoạn 1: tuning (24 run, seed 42)

```bash
python scripts/e2_tuning.py make          # Vinh chạy MỘT lần, commit configs/tuning_runs/E2/, đặt tag exp-v1
python train.py --config configs/tuning_runs/E2/ti_dfd_tau2_lam1_wt2.yaml --data-root $DATA --teacher-checkpoint "$TEACHER"
python eval.py --run-dir $RUN --checkpoint $RUN/checkpoints/best-*.ckpt --device cuda   # chấm validation
```

Grid ở `configs/tuning.yaml`: DFD gồm τ ∈ {1, 2, 4} × λ ∈ {0,5; 1} × w_t ∈ {1, 2} (12 trial), KD thường gồm τ ∈ {1, 2, 4} × λ ∈ {0,25; 0,5; 1; 2} (12 trial). `ti_ce` và `ti_factorized` không có siêu tham số KD nên không tuning.

Xong 24 trial: mọi người upload thư mục run. Vinh tải `resolved.yaml` và `metrics.csv` của 24 run về `results/runs/`, rồi chạy `python scripts/e2_tuning.py select`. Trial diverge vẫn giữ trong bảng, không chạy lại để thay.

**Khóa cấu hình (Vinh, một commit duy nhất, rồi đặt tag `exp-v2`):** ghi τ, λ, w_t đã chọn vào các config dưới đây. Sau commit này không ai sửa siêu tham số nữa.

| Config | Lấy giá trị nào |
|---|---|
| `E2/ti_dfd_s4{2,3,4}`, `E4`, `E5`, `E6`, `E7` của `ti_dfd` | τ, λ, w_t của DFD |
| `E2/ti_kd_s4{2,3,4}` và các config `ti_kd` của E5, E6, E7 | τ, λ của KD thường |
| `E3/no_tone_weight_*` | τ, λ của DFD. Giữ w_t = 1 và `tone_confidence_weight: false` |
| `E3/joint_kd_*`, `E3/marginal_joint_kd_*` | τ, λ của DFD. `joint_lambda` giữ 1,0 |
| `E3/tone_only_kd_*` | τ, λ, w_t của DFD. Giữ trọng số base = modifier = 0 |
| `E3/temperature_sweep_*`, `E3/no_validity_mask_*` | λ, w_t của DFD (τ của sweep giữ nguyên 1, 2, 4) |

### E2 giai đoạn 2: chạy chính (10 run mới, tag `exp-v2`)

| Biến thể | Seed 42 | Seed 43, 44 |
|---|---|---|
| `ti_ce` | `configs/runs/E2/ti_ce_s42.yaml` | `ti_ce_s43`, `ti_ce_s44` |
| `ti_factorized` | `ti_factorized_s42` | `ti_factorized_s43`, `ti_factorized_s44` |
| `ti_kd` | **Dùng lại** trial KD tốt nhất của tuning, chấm thêm `--splits test unseen_test --subsets all diacritic_heavy small_crops rare_combinations` | `ti_kd_s43`, `ti_kd_s44` |
| `ti_dfd` | **Dùng lại** trial DFD tốt nhất, chấm như trên | `ti_dfd_s43`, `ti_dfd_s44` |

`ti_ce` và `ti_factorized` không cần teacher hay tuning, nên **chạy được ngay từ tag `exp-v1`**, song song với tuning. Riêng 6 run này được miễn quy tắc tag `exp-v2`, vì commit khóa cấu hình không đụng tới config của chúng.

Quy tắc kết luận (proposal 10): DFD chỉ được coi là tốt hơn khi mức tăng trung bình lớn hơn một độ lệch chuẩn **và** cùng chiều ở cả ba seed. Nếu không thì ghi "inconclusive".

### E3: ablation (8 run mới, tag `exp-v2`)

| Biến thể | Config | Run mới | Ghi chú |
|---|---|---|---|
| Full DFD | (E2 `ti_dfd`) | 0 | Tham chiếu |
| − tone weighting | `no_tone_weight_s4{2,3,4}` | 3 | |
| Joint KD | `joint_kd_s4{2,3,4}` | 3 | |
| Marginal + joint | `marginal_joint_kd_s42` | 1 | Chỉ mang tính gợi ý (1 seed) |
| Tone-only KD | `tone_only_kd_s42` | 1 | Chỉ mang tính gợi ý |
| Temperature sweep | (3 trial tuning DFD có λ, w_t đã chọn) | 0 | Grid tuning đã có đủ τ ∈ {1, 2, 4} |
| − validity mask | (checkpoint `ti_dfd` seed 42) | 0 | Chỉ khác lúc decode. `train.py` cố ý từ chối train config này. Còn chờ chốt cách decode (mục 7) |

Thiếu GPU thì cắt theo thứ tự: tone-only, marginal + joint, seed 43/44 của joint KD.

### E4: nén model (không train)

Dùng checkpoint E2 `ti_ce`, `ti_dfd` (3 seed) và teacher. Xuất ONNX FP32 và INT8 dynamic từ **cùng một checkpoint** (tách tác động thu nhỏ với tác động lượng tử hóa), ghi dự đoán ra CSV, chấm bằng `eval.py --config configs/runs/E4/<...>.yaml --predictions-dir <dir>`. Code xuất ONNX thuộc Task 19.

### E5: tổng quát hóa

- **Không train:** các config `E5/<variant>_s4x.yaml` chấm lại checkpoint E2 trên `unseen_test` và các subset: `eval.py --config configs/runs/E5/ti_dfd_s42.yaml --checkpoint <checkpoint E2> ...`.
- **4 run train:** `E5/ti_ce_s42_fraction{25,50}`, `E5/ti_dfd_s42_fraction{25,50}`. Hiện `train.py` cố ý từ chối, vì chưa có cách lấy mẫu con được chốt (mục 7). Phải có code này trước khi chạy.

### E6, E7

- **E6:** đo độ trễ CPU của file ONNX trên **một máy duy nhất** và trong một phiên, theo mục `benchmark` của config (4 luồng, batch 1, 50 lần warmup, 500 lần đo, lấy median). Không so độ trễ đo trên các GPU khác nhau.
- **E7:** phân tích lỗi trên `predictions/validation.csv` của 4 student seed 42 (20 lỗi mỗi model). Không nhìn lỗi trên test để chọn mẫu.

## 5. Quy tắc để không xung đột

1. **Chỉ chạy từ tag chung, với working tree sạch.** `train.py` từ chối chạy khi có file sửa đổi hoặc file chưa track. Muốn thử code thì làm trên nhánh riêng và không báo cáo số liệu từ nhánh đó.
2. **Sửa code thì tạo nhánh mới từ `main`, mở PR, có một người review.** Không commit thẳng lên `main`. Không push hay merge các nhánh cũ đã xóa ngày 16/11 (chúng chứa ảnh VinText).
3. **Config chỉ do một người sửa vào hai thời điểm:** Vinh commit grid tuning (tag `exp-v1`) và commit khóa siêu tham số (tag `exp-v2`). Người chạy không sửa YAML. Nếu cần sửa thì báo nhóm, sửa qua PR và đặt tag mới.
4. **Code sửa giữa đợt** (ví dụ phát hiện bug) thì đặt tag mới, rồi ghi vào bảng theo dõi những run nào đã chạy trên tag cũ. Run có bug phải chạy lại. Không trộn kết quả của hai tag nếu thay đổi có ảnh hưởng tới train.
5. **Nhận việc trước khi chạy:** ghi tên và đánh dấu `running` trong sheet theo dõi của nhóm (cột như `configs/run_manifest.csv`: experiment, variant, seed, config, người chạy, nền tảng, run_id, trạng thái). Không ghi chung một file CSV trong repo.
6. **Kết quả không vào Git:** checkpoint, `results/runs/`, dữ liệu và dự đoán để trên Drive private của nhóm, mỗi run một thư mục `Vi-DFD/runs/<run_id>/`. `run_id` có thời gian và mã ngẫu nhiên nên hai người không bao giờ trùng tên.
7. **Tổng hợp tập trung:** chỉ một người (Vinh) gom `metrics.csv` của mọi run, kiểm tra commit, tag và data identity, rồi mới làm bảng kết quả.
8. **Ghi nền tảng:** đặt `DFD_PLATFORM=colab`, `kaggle` hoặc `local` trước khi chạy. GPU thật được ghi tự động.

## 6. Phân công

### Code còn thiếu

| Việc | Cần cho | Người code | Người review | Hạn |
|---|---|---|---|---|
| Cách decode khi bỏ validity mask (E3) | E3 `no_validity_mask` | Tuấn chốt quy tắc, Vinh code trong `dfd/heads.py` + `eval.py` | Tuấn | 22/11 |
| Lấy mẫu con 25% / 50% cho `training_fraction` | E5 (4 run) | Tiến (quy tắc lấy mẫu theo ảnh, cố định seed, ghi manifest) | Vinh | 21/11 |
| Train VietOCR, CRNN và xuất CSV dự đoán | E1 | Tuấn (Task 12) | Tiến | 20/11 |
| Xuất ONNX FP32 / INT8, dự đoán ra CSV, đo độ trễ | E4, E6 | Tuấn (Task 19) | Vinh | 24/11 |
| Script tổng hợp bảng E2 (mean ± std, hiệu theo cặp seed), CI bootstrap cho E4 | E2, E4 | Vinh (E2), Tiến (E4) | Chéo | 25/11 |
| Sửa ký hiệu không hợp lệ `|` trong `dfd/metrics.py` | Mọi bảng | Tiến (Task 10) | Vinh | Trước khi chấm điểm chính thức |

### Chạy run (48 run train)

Chia theo người có GPU và theo task. Người chạy không nhất thiết là người code.

| Người | Đợt 1 (tag `exp-v1`, 17–20/11) | Đợt 2 (tag `exp-v2`, 21–25/11) | Số run |
|---|---|---|---|
| **Vinh** | Tuning DFD: 12 trial | `ti_dfd` s43, s44. E3: `no_tone_weight` × 3, `tone_only_kd` × 1 | 18 |
| **Tuấn** | Tuning KD thường: 12 trial. Đo mốc 13. VietOCR, CRNN (E1) | `ti_kd` s43, s44. E3: `joint_kd` × 3, `marginal_joint_kd` × 1 | 20 |
| **Tiến** | `ti_ce` × 3, `ti_factorized` × 3 (chạy ngay, không cần teacher) | E5: 4 run `fraction` | 10 |

Ngoài phần train: Tiến chấm E4, E5 và làm E7. Tuấn làm E6 (đo độ trễ trên một máy). Vinh chạy `select`, khóa cấu hình, tổng hợp E2/E3. Ai xong sớm thì nhận thêm run của người khác qua sheet theo dõi.

### Lịch

| Ngày | Việc |
|---|---|
| 16–17/11 | Chốt mục 2 và 7. Tuấn xác nhận split. Đặt tag `exp-v1`. Đo mốc 13 |
| 17–20/11 | Đợt 1: 24 trial tuning, 6 run `ti_ce` / `ti_factorized`, E1 baseline |
| 21/11 | Vinh chạy `select`, khóa cấu hình, đặt tag `exp-v2` |
| 21–25/11 | Đợt 2: E2 seed 43/44, E3, E5 |
| 24–27/11 | Chấm điểm E4, E5, E6, E7. Tổng hợp bảng |
| **27/11** | Đóng băng thí nghiệm |

## 7. Còn phải chốt trước khi chạy

| Việc | Ai | Chặn |
|---|---|---|
| Tuấn gửi `splits/freeze_manifest.json` và `splits/manifests/image_split_assignments.csv` của bản dữ liệu Tuấn đã dùng train teacher. Hai file này không có nhãn | Tuấn | Mọi run KD. Nếu gán ảnh vào train/val khác với bản của Tiến thì phải train lại teacher trên bản của Tiến (khoảng 30 phút GPU) |
| Cả nhóm đồng ý dùng bản dữ liệu của Tiến (mục 2) | Cả nhóm | Mọi run |
| Nghĩa của "một độ lệch chuẩn" trong quy tắc kết luận E2 (std của run nào, mẫu hay tổng thể) | Cả nhóm | Kết luận E2 |
| Mẫu số của Component ER, cách tính insert/delete | Tiến | Bảng E4 |
| E4 dùng một teacher seed hay ba (proposal đang ghi ba) | Cả nhóm | E4 |
| Cách decode bỏ validity mask và cách lấy mẫu con cho E5 | Xem mục 6 | E3, E5 |
| Thư mục Drive private chung và sheet theo dõi run | Vinh tạo | Bàn giao kết quả |
