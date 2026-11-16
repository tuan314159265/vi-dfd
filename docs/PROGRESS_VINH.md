# Tiến độ các task của Vinh

Cập nhật: 16/11/2026 (chiều). `main` có Task 7, 11, 14, 26 và phần chuẩn bị Task 15. Nhánh `integrate/clean-history` nối tiếp `main` với notebook dữ liệu của Tiến (bản đã xóa ảnh, `921bb6c`), 3 commit Task 9 của Tuấn (cherry-pick, giữ tác giả) và phần train student của Vinh. Nhánh này **không** chứa commit `9b3ab7e` (ảnh VinText trong output notebook). Xem mục "Các nhánh và ảnh VinText".

## Trạng thái

| # | Task | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Proposal chi tiết | Done (từ trước) | |
| 7 | Bảng phân rã ký tự | **Done** | Đã đối chiếu với charset VinText, bảng giữ nguyên 229 lớp |
| 11 | Head phân rã cho student | **Done** | Đã kiểm chứng trên PARSeq thật |
| 14 | Cài DFD và KD thường | **Done** | Đã ghép vào `train.py` (`dfd/student.py`). Smoke test `ti_ce`, `ti_kd`, `ti_dfd` chạy thông trên dữ liệu thật với teacher thật của Tuấn. Xem mục "Train student" |
| 15 | E2 sơ bộ, chọn siêu tham số | **Đang chờ** | Đã có script sinh grid (12 + 12 trial) và chọn cấu hình, đã có `eval.py`, đã có dữ liệu thật (16/11). Đã có teacher và recipe (16/11). Còn chờ Tuấn xác nhận split, mốc 13 và GPU. Xem "Cần nhắc" |
| 16 | E2 đủ 3 seed + E3 | Chờ Task 15 | |
| 25 | Đóng gói và hoàn tất nội bộ | Chờ thí nghiệm và báo cáo | |
| 26 | Pipeline đánh giá `eval.py` (task mới) | **Done** | Đã kiểm chứng trên LMDB giả, checkpoint PARSeq thật và LMDB VinText thật (16/11). Xem mục "Dữ liệu thật" |

Các task "Cả nhóm" (13, 20, 23, 24) phụ thuộc kết quả chưa có. Cho mốc 13 em đã chuẩn bị script đo thời gian step có KD.

## Cần nhắc (việc đang chặn task của Vinh)

| Ai | Việc | Chặn task | Ghi chú |
|---|---|---|---|
| **Tuấn** | Gửi `splits/freeze_manifest.json` và `splits/manifests/image_split_assignments.csv` của bản dữ liệu Tuấn tự tạo (chỉ có checksum và image id → split, không có nhãn) | 15 | Teacher được train trên bản dữ liệu Tuấn tạo lại ngày 09/11, không phải bản Tiến gửi: SHA của `freeze_manifest.json` và của cả 4 file LMDB đều khác. Phải xác nhận gán ảnh vào train/val trùng với bản của Tiến thì mới dùng teacher cho E2 được. Xem mục "Teacher của Tuấn" |
| **Cả nhóm** | Chốt dùng **một** bản dữ liệu (đề xuất: bản của Tiến) cho mọi run student và mọi lần chấm điểm | 15, 16 | Kể cả khi split trùng, ảnh crop vẫn khác byte, nên mọi student và mọi bảng kết quả phải dùng chung một bản |
| **Cả nhóm** | Đồng ý thay `main` bằng nhánh `integrate/clean-history` và xóa ba nhánh remote còn chứa ảnh | Merge | Xem mục "Các nhánh và ảnh VinText" |
| **Tuấn** | Kết quả E1 (Task 12), nhất là Ti vanilla | 15, mốc 13 | Ti vanilla là mốc so sánh của E2 |
| **Tiến** | Hoàn tất Task 10 và mở PR | 15 | Nhánh `origin/feat/metrics-evaluation` (commit `03109a4`, 15/11) đã có `dfd/metrics.py`, bảng vẫn ghi In progress. Xem mục "Góp ý cho module metric" bên dưới |
| **Cả nhóm (mốc 13)** | Chạy `scripts/time_kd_step.py --batch-size 64 --train-samples 23267` trên GPU thật | 13, 15 | Ra ms/step cho 4 biến thể và thời gian 1 epoch dự kiến, để chốt ngân sách GPU và số seed |
| **Tuấn, Tiến** | Dùng `eval.py` cho E1, E4, E5 thay vì tự chấm điểm | 12, 17, 18 | VietOCR, CRNN và bản INT8/ONNX ghi file `source_id,prediction` rồi chấm bằng `--predictions-dir`, để mọi model chung mẫu số |
| **Người chạy E3 (Tuấn)** | Chốt cách decode khi bỏ validity mask | 16 | `joint_lambda` đã chốt là 1.0 |

## Đã chốt ngày 15/11

- **Ngân sách tuning:** KD thường dùng $	au \in \{1, 2, 4\}$, $\lambda \in \{0.25, 0.5, 1, 2\}$, tức 12 trial, bằng DFD. Lý do ghi ở `docs/proposal/09`, đã ghi vào `configs/tuning.yaml`.
- **`joint_lambda = 1.0`** cho `marginal_joint_kd`: loss bằng đúng DFD đầy đủ cộng Joint KD. Lý do ghi ở `docs/proposal/11`, đã ghi vào YAML.
- **`eval.py` do Vinh phụ trách** (Task 26).

## Task 7: bảng phân rã ký tự

File: `dfd/vietchar.py`, `tests/test_vietchar.py`, `scripts/check_vocab_coverage.py`.

- `DecompositionTable.from_characters()` tạo bảng cố định **229 lớp** gồm EOS, 94 ký tự ASCII in được và 134 chữ cái tiếng Việt.
- Thứ tự lớp trùng đầu ra của PARSeq: EOS là lớp 0, ký tự từ lớp 1. ASCII giữ thứ tự của `94_full.yaml`.
- Ba bộ thành phần có kích thước 95 base (gồm EOS), 5 modifier, 6 tone. EOS là triple `([EOS], none, level)`.
- `table.matrix("base" | "modifier" | "tone")` trả ma trận marginalize $M_x$ kích thước `229 × K`.
- `table.fingerprint()` dùng để ghi vào metadata của run: `5958403f4ca3a84c80d007a9af14c1f9cfd9ffa7dd0fd3cd812c65140911360d`. Test ghim giá trị này.
- `scripts/check_vocab_coverage.py` so ký tự của dataset với bảng.

## Task 11: head phân rã

File: `dfd/heads.py`, `tests/test_heads.py`.

- Thay head bằng `model.head = FactorizedHead(model.head.in_features, table)`, mã gốc PARSeq không cần sửa.
- `forward` trả log-xác suất chuẩn hóa trên đúng 229 lớp hợp lệ, cùng shape `[N, L, 229]` với head cũ.
- `component_logits(hidden)` trả logit ba thành phần. `component_loss(logits, targets, pad_id)` trả CE từng thành phần và tổng.
- Head kế thừa `nn.Linear` với ba khối đầu ra, vì `PARSeq._device` đọc `head.parameters(recurse=False)`.
- Đã kiểm chứng trên PARSeq commit `1902db043c029a7e03a3818c616c06600af574be`: `forward` có decode AR và refinement, `tokenizer.decode`, một bước train có backward tới encoder.
- Đã nạp `parseq_tiny-e7a21b54.pt` vào student. Chỉ `head.*` và `text_embed.embedding.weight` lệch shape và phải bỏ qua khi nạp, 172 key còn lại nạp đúng.
- Chưa cài ablation bỏ validity mask (E3), vì cách decode và chấm điểm vẫn chưa chốt. Chưa nối vào vòng train, phần này thuộc Task 9 và 14.

## Task 14: DFD và KD thường

File: `dfd/distill.py`, `dfd/parseq_kd.py`, `tests/test_distill.py`, `tests/test_parseq_kd.py`.

### Loss (`dfd/distill.py`)

- `DistillationObjective.from_config(config["distillation"])` đọc thẳng mục `distillation` của YAML. Nó kiểm tra mode, $\tau$, $\lambda$, trọng số thành phần và `joint_lambda`.
- Có ba mode:

  | Mode | Dùng cho | Số hạng KD tại vị trí $i$ |
  |---|---|---|
  | `marginal` | `ti_dfd`, `no_tone_weight`, `tone_only_kd`, `no_validity_mask`, temperature sweep | $\alpha_i\sum_x w_x\,\mathrm{KL}(p^x_T\Vert p^x_S(\tau))$ |
  | `joint` | `ti_kd` (KD logit 229 lớp, head thường), `joint_kd` (head phân rã) | $\alpha_i\,\mathrm{KL}(p_T\Vert p_S(\tau))$ trên 229 lớp |
  | `marginal_and_joint` | `marginal_joint_kd` | tổng hai số hạng trên, số hạng joint nhân `joint_lambda` |

- Loss tổng là `CE + λ·τ²·KD`. Với head phân rã, CE là tổng CE của ba thành phần. Với head thường, CE là CE 229 lớp.
- $\alpha_i = 1 - H(p^t_T)/\log 6$ khi `tone_confidence_weight: true`, ngược lại $\alpha_i = 1$.
- KD tính bằng float32 kể cả khi bật AMP. Gradient không đi vào teacher.
- Quy ước mask, EOS và cách lấy trung bình đã ghi vào `docs/proposal/06` và `docs/IMPLEMENTATION.md`.

### Bước train có teacher (`dfd/parseq_kd.py`)

- `shared_permutation_loss` chép vòng lặp `training_step` của PARSeq gốc. Hoán vị được sinh một lần rồi dùng chung. Mỗi lượt, teacher và student nhận cùng `tgt_in`, cùng `tgt_mask`, `query_mask`, `tgt_padding_mask`. Teacher chạy trong `no_grad` và ở eval mode.
- `DistilledPARSeq` kế thừa PARSeq của `strhub`, nhận đúng tham số như PARSeq. Nó có thêm `use_factorized_head(table)` và `configure_distillation(objective, teacher)`.
- Teacher nằm ngoài cây module của student, nên không bị lưu vào checkpoint student, không vào optimizer, và Lightning không bật train mode cho nó. Teacher được tự chuyển sang device của student.
- Teacher và student được kiểm tra trùng tokenizer. Teacher phải có head 229 lớp thường.
- `load_frozen_teacher(path, table)` nạp checkpoint và kiểm tra `charset_train` của teacher.

### Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Không bật KD, head thường: loss khớp `PARSeq.training_step` gốc khi dùng cùng hoán vị | Khớp tới `rel=1e-5` |
| Teacher là bản sao của student: KD joint qua mọi hoán vị | < 1e-6. Đây là bằng chứng hai model thấy cùng hoán vị và context. Với một teacher khác, KD > 1e-3 |
| Teacher rộng hơn (giống S so với Ti): DFD backward | Gradient tới encoder và head student; teacher không có gradient |
| `Trainer.fit` 2 step với `DistilledPARSeq` + head phân rã + DFD | Chạy được; teacher không đổi; log có `train/kd`, `train/kl_tone`, `train/alpha`, `train/ce_tone` |
| Công thức: marginal, $\alpha$, KL, mask padding, $\tau^2$, tone-only, joint trên head phân rã | 24 unit test |
| Mọi YAML có KD trong `configs/runs` đều đọc được | Pass. Chỉ `marginal_joint_kd` báo lỗi vì `joint_lambda: null` |

### Ghép vào script train (khi Tuấn push Task 9)

```python
from dfd.config import load_config
from dfd.distill import DistillationObjective
from dfd.parseq_kd import DistilledPARSeq, load_frozen_teacher
from dfd.vietchar import DecompositionTable

config = load_config("configs/runs/E2/ti_dfd_s42.yaml")
table = DecompositionTable.from_characters()
student = DistilledPARSeq(**parseq_tiny_hparams)    # thay cho strhub PARSeq, cùng tham số
if config["model"]["head"] == "factorized":
    student.use_factorized_head(table)              # sau khi nạp pretrained parseq_tiny (bỏ head.*, text_embed)
objective = DistillationObjective.from_config(config["distillation"])
teacher = load_frozen_teacher(config["model"]["teacher_checkpoint"], table) if objective.enabled else None
student.configure_distillation(objective, teacher, table)
trainer.fit(student, datamodule)
```

`ti_ce` có thể giữ PARSeq gốc. `ti_factorized` dùng `DistilledPARSeq` với KD tắt, để CE tính theo từng thành phần.

### Giới hạn và ghi chú

- Chưa chạy trên dữ liệu thật hay teacher thật. Các test dùng model nhỏ khởi tạo ngẫu nhiên.
- $\alpha_i$ dùng tone marginal đã làm mềm bằng $\tau$, đúng công thức proposal. Vì vậy $\tau$ lớn làm giảm $\alpha_i$, và trong E3 hai yếu tố $\tau$ và tone weighting không độc lập.
- Ablation `no_validity_mask` khi train giống hệt DFD đầy đủ, vì CE thành phần và KL marginal không dùng mask. Hai biến thể chỉ khác nhau lúc decode. Có thể dùng chung checkpoint DFD để tiết kiệm một run, nếu nhóm đồng ý.
- Trên CPU (batch 8, chỉ để thử script), một step có teacher tốn khoảng 1,9 lần step không KD. Con số trên GPU phải đo lại ở mốc 13.

## Task 15: phần đã chuẩn bị

- `scripts/e2_tuning.py make` sinh 1 YAML cho mỗi trial, seed 42, chỉ đánh giá trên validation, vào `configs/tuning_runs/E2/`. Grid lấy từ `configs/tuning.yaml`: 12 trial DFD và 12 trial KD thường. Script từ chối chạy khi `equal_budget_policy` còn `null`.
- `scripts/e2_tuning.py select` đọc `results/runs/*/resolved.yaml` và `metrics.csv`, xếp hạng theo word acc validation, hòa thì xét 1-NED rồi thứ tự grid. Nó in cả các trial chưa có kết quả.
- `scripts/time_kd_step.py` đo ms/step và bộ nhớ đỉnh cho `ti_ce`, `ti_factorized`, `ti_kd`, `ti_dfd` với kích thước Ti/S chính thức. Lệnh cho mốc 13, thay số crop train thật vào:

  ```bash
  python scripts/time_kd_step.py --device cuda --batch-size <batch của Task 9> --train-samples 23267
  ```

- Khi đủ điều kiện: điền `training.*`, `data.*`, `model.teacher_checkpoint` vào hai YAML gốc E2 (`ti_dfd_s42`, `ti_kd_s42`), chạy `make`, train từng trial, chạy `select`, rồi ghi cấu hình đã chọn vào 12 YAML E2 ba seed. Nếu xin compute, gửi bảng kết quả `select` cho giảng viên.

## Task 26: pipeline đánh giá `eval.py`

File: `eval.py`, `dfd/evaluation.py`, `tests/test_evaluation.py`.

```bash
# PARSeq (teacher, ti_ce, ti_kd, ti_factorized, ti_dfd, E3): tự nhận head thường hay head phân rã
python eval.py --run-dir results/runs/<run_id> --checkpoint best.ckpt     --data-root <processed>/lmdb --subset-dir <processed>/splits/subsets

# Model khác (VietOCR, CRNN, ONNX/INT8): <dir>/<split>.csv gồm cột source_id,prediction
python eval.py --run-dir results/runs/<run_id> --predictions-dir <dir> --data-root ... --subset-dir ...
```

- Không có `--run-dir` thì dùng `--config`, và script tự tạo run mới bằng `prepare_run` (cần Git sạch). Split và subset lấy từ mục `evaluation` của YAML, có thể ghi đè bằng `--splits`, `--subsets`.
- Tự đọc LMDB, **không lọc mẫu nào**, khác `LmdbDataset` của PARSeq. Nhãn dài hơn 25 ký tự hay chứa ký tự ngoài bảng (`°`) vẫn được giữ và tính là sai, nên mọi model chung mẫu số.
- Chuẩn hóa nhãn và dự đoán: NFC, bỏ dấu cách, chuyển chữ thường (protocol không phân biệt hoa/thường). Ký tự nhãn ngoài bảng được thay bằng ký hiệu không hợp lệ của module metric, nên crop đó tính là sai. Dấu thanh giữ nguyên vị trí (`hoà` ≠ `hòa`).
- Subset nối bằng `source-id` (`image_id:line_number`). Script báo lỗi nếu file subset có mẫu không có trong LMDB.
- Tên split trong YAML và `metrics.csv` là `validation`, `test`, `unseen_test`, ứng với thư mục LMDB `val`, `test`, `unseen_test`.
- Ghi vào thư mục run:
  - `metrics.csv`: một dòng cho mỗi split × subset, kèm commit hash, config hash, seed, số tham số.
  - `predictions/<split>.csv`: dự đoán từng mẫu, dùng cho phân tích lỗi (Task 21).
  - `tone_confusion/<split>_<subset>.json`: ma trận nhầm thanh (Task 17).
  - `eval_info.json`: SHA-256 của checkpoint và của từng LMDB, fingerprint bảng ký tự, số nhãn chứa ký tự ngoài bảng.
- Không ghi đè dòng đã có trong `metrics.csv` trừ khi thêm `--replace`. Subset rỗng thì bỏ qua và báo.
- Checkpoint chỉ chấm ở FP32. INT8 chấm qua `--predictions-dir`, với `precision` lấy từ YAML (E4).

Kiểm chứng (6 test): LMDB và subset giả gồm nhãn có `°`, nhãn 30 ký tự, khác hoa/thường, sai thanh, khác vị trí dấu; chạy đủ hai chế độ. Checkpoint PARSeq có head phân rã được lưu rồi nạp lại, cho cùng dự đoán với model gốc.

## Teacher của Tuấn (nhận 16/11)

Nhánh `origin/feat/teacher-parseq-s` (`30b9ba4`) và gói `Vi-DFD_Task9_Teacher_E1_20261109.zip`. Gói được giải nén ra `Downloads/processed/teacher_s42_e1/`. SHA-256 khớp 18/18 file.

- Recipe đã chốt: batch 64, 20 epoch, AdamW + OneCycleLR, base LR 7e-4 (peak thực 1,75e-4), warmup 0,075, weight decay 0, clip 20, AMP. Đã nằm sẵn trong `training.*` của mọi YAML.
- `load_frozen_teacher` nạp được checkpoint, charset đúng 228 ký tự theo thứ tự bảng, 23.935.717 tham số.

Chấm lại bằng `eval.py` (CPU, FP32) trên bản dữ liệu của Tiến:

| Split | n | Word acc | 1-NED | Base ER | Modifier ER | Tone ER |
|---|---|---|---|---|---|---|
| validation | 2.507 | 0,8871 | 0,9461 | | | 0,0232 |
| test | 7.201 | 0,8460 | 0,9195 | 0,0659 | 0,0395 | 0,0481 |
| unseen_test | 10.068 | 0,8779 | 0,9479 | 0,0282 | 0,0170 | 0,0240 |

- Validation 88,71% so với 88,67% Tuấn báo (2.224 so với 2.223 từ đúng), diacritic-heavy 91,80% so với 91,26% (168 so với 167 từ). Lệch đúng 1 từ, hợp với khác biệt CPU FP32 và GPU AMP.
- Đây là bằng chứng gián tiếp rằng split val trùng nhau. Nếu ảnh val của Tiến nằm trong train của Tuấn, độ chính xác trên val sẽ cao bất thường. Vẫn cần Tuấn gửi file để xác nhận trực tiếp (xem "Cần nhắc").
- Số ở bảng là để đối chiếu, chạy ngoài thư mục `results/runs`. Số chính thức của E1 phải chạy `eval.py --config configs/runs/E1/teacher_s42.yaml` từ commit sạch.

## Train student (Task 14 ghép xong)

File: `dfd/student.py`, `tests/test_student.py`, `train.py`. Sửa nhỏ trên code của Tuấn: vòng `Trainer.fit` của `run_teacher` tách thành `fit_parseq` dùng chung. `initialize_teacher` nhận thêm `url`, nên khởi tạo được cả trọng số tiếng Anh `parseq_tiny`. `--smoke-test` chạy được trên CPU (FP32).

```bash
python train.py --config configs/runs/E2/ti_dfd_s42.yaml \
    --data-root <processed>/vintext_notebook \
    --teacher-checkpoint <processed>/teacher_s42_e1/checkpoints/best-epoch=17-validation_word_acc=0.8867.ckpt
# thêm --smoke-test để chạy thử 1 batch; ti_ce, ti_factorized không cần --teacher-checkpoint
```

- Student dùng y hệt pipeline của teacher: cùng cách đọc LMDB (train bỏ 2 crop `°`, val giữ đủ 2.507), cùng augment, budget 64 × 20 epoch, metric validation, cách chọn checkpoint tốt nhất. Chỉ khác head (thường hoặc phân rã) và loss (CE, KD, DFD, các biến thể E3), lấy từ YAML.
- `StudentPARSeq` kế thừa `VietnamesePARSeq` của Tuấn rồi đến `DistilledPARSeq`, nên validation và checkpoint theo kiểu teacher, còn bước train là bước có KD.
- Khởi tạo từ `parseq_tiny-e7a21b54.pt` theo đúng cách teacher: chép 97 context embedding theo token, reset head. Student có head phân rã thì thay head sau bước này.
- `--data-root` tự điền `vocab_path`, `split_manifest`, `subset_manifest` theo cấu trúc chuẩn nếu YAML để `null`. Không cần `preparation.json`, nên dùng được bản dữ liệu của Tiến.
- `data_identity.json` của run ghi SHA của freeze manifest và LMDB, kèm data identity của teacher. Nếu teacher train trên freeze manifest khác, script in `WARNING` (hiện đang in, vì lý do ở mục trên).
- Từ chối chạy, kèm lý do: `training_fraction < 1` (E5 chưa chốt cách lấy mẫu) và `no_validity_mask` (train giống hệt DFD, nên dùng lại checkpoint `ti_dfd` cùng seed).

| Kiểm tra | Kết quả |
|---|---|
| `pytest` toàn repo | 145 passed (thêm 10 test student, 3 test của Tuấn) |
| Smoke `ti_dfd`, `ti_kd`, `ti_ce` trên dữ liệu thật, teacher thật, CPU | Cả ba `smoke_completed`. Student 6,0 triệu tham số, khoảng 70 giây mỗi run |
| Checkpoint student có head phân rã qua `eval.py` | Nạp lại cho cùng logit |

## Các nhánh và ảnh VinText

Ảnh crop VinText (output của `02_crop_vintext` cell 7 và `03_EDA` cell 19, 20) chỉ còn trong **lịch sử** commit `9b3ab7e`. Bản mới nhất của mọi nhánh chỉ còn biểu đồ thống kê. Nhưng ba nhánh remote đều có `9b3ab7e` trong lịch sử: `feat/vintext-data-preparation`, `feat/metrics-evaluation`, `feat/teacher-parseq-s`. Merge bất kỳ nhánh nào trong số này vào `main` sẽ đưa ảnh vào `main` vĩnh viễn.

Cách xử lý: nhánh `integrate/clean-history` dựng lại từ `main`, gồm:

1. `921bb6c`: notebook của Tiến ở trạng thái cuối, kèm các sửa của Vinh, không có ảnh. Tác giả là Tiến.
2. Cherry-pick `36ce735`, `9d56af0`, `30b9ba4` của Tuấn, giữ tác giả. Bỏ commit merge `b51a8c2` vì nó kéo `9b3ab7e` vào. Khi xung đột, notebook giữ bản sạch, README và IMPLEMENTATION gộp nội dung của cả hai bên.
3. Commit của Vinh: requirements, train student, tài liệu.

So với bản cuối của Tuấn, code teacher không đổi gì ngoài hai chỗ ở mục "Train student".

Khi nhóm đồng ý: fast-forward `main` lên nhánh này, rồi **xóa** ba nhánh remote trên. Tiến và Tuấn tạo nhánh mới từ `main`, không push lại nhánh cũ. Sau khi xóa nhánh, GitHub vẫn có thể giữ commit cũ một thời gian nếu ai đó biết hash, và người đã clone vẫn còn bản sao. Muốn gỡ hẳn thì phải liên hệ GitHub Support.

## Dữ liệu thật (nhận 16/11)

Tiến gửi `lmdb/` và `splits/` qua kênh private. Đặt ngoài repo đúng cấu trúc trong README của nhánh dữ liệu:

```text
Downloads/
├── processed/vintext_notebook/
│   ├── lmdb/{train,val,test,unseen_test}/
│   └── splits/{manifests,subsets,duplicates}/, freeze_manifest.json
└── vi-dfd/
```

Lệnh chấm điểm trên máy này: `--data-root ../processed/vintext_notebook/lmdb --subset-dir ../processed/vintext_notebook/splits/subsets`.

| Kiểm tra | Kết quả |
|---|---|
| SHA-256 của 21 file trong `freeze_manifest.json` | Khớp cả 21 |
| Số crop đọc từ LMDB | train 23.269, val 2.507, test 7.201, unseen test 10.068. Khớp manifest |
| Subset nối bằng `source-id` | Khớp hết. val: diacritic 183, small 489, rare 6. test: 587, 1.545, 24. unseen: 776, 1.916, 33 |
| Ký tự ngoài bảng 229 lớp | Chỉ dấu cách (308 nhãn: train 223, val 29, test 8, unseen 48) và `°` (2 crop train: `36:3`, `36:7`). Đúng quy tắc đã chốt |
| Nhãn dài hơn 25 ký tự | val 1, test 2, train và unseen 0 |
| `eval.py --predictions-dir`, dự đoán = nhãn | Word acc, 1-NED = 1,0000 và mọi ER = 0 ở cả 3 split × 4 subset |
| `eval.py --predictions-dir`, dự đoán = nhãn bỏ dấu thanh | Base ER = modifier ER = 0, chỉ tone ER tăng (0,126 / 0,128 / 0,132). Word acc còn 0,537 / 0,522 / 0,512, tức khoảng một nửa số từ có dấu thanh |
| `eval.py --checkpoint`, student nhỏ khởi tạo ngẫu nhiên, head phân rã, CPU | Chạy hết 2.507 mẫu val, ghi đủ `metrics.csv`, `predictions/`, `tone_confusion/`, `eval_info.json` |

Fingerprint LMDB (`evaluation.lmdb_fingerprint`, ghi vào `eval_info.json` của mỗi run):

| Split | SHA-256 |
|---|---|
| train | `af658c68e81f2eafddbe82898ac117f829df787198c961422ac70c35b77a7246` |
| val | `e8648b1bdcd5fd36a7cd5c0a7439b05463917176960e72e0c148e76cc1e5a802` |
| test | `56f30f3496f33ec1116274f1cf72c43faae1d645a087c135c1b44b112b4fe41f` |
| unseen_test | `c128a6ff4e3a9e25591f4c96a60392aea31570aa050cb1fa81c8221549f11419` |

**Lưu ý cho Task 9 (Tuấn):** `LmdbDataset` của PARSeq (charset 228 ký tự, `remove_whitespace=True`, `normalize_unicode=False`) giữ **cả 23.269** crop train. Nó **không bỏ** 2 crop chứa `°` mà biến nhãn thành chuỗi đã xóa `°`, trái với quy tắc đã chốt. Script train phải lọc riêng 2 crop `36:3` và `36:7`. Ở val/test, nó chỉ giữ 2.506 / 7.199 / 10.068 mẫu vì bỏ nhãn dài hơn 25 ký tự. Vì vậy validation trong lúc train sẽ lệch nhẹ so với `eval.py`, còn số liệu báo cáo phải lấy từ `eval.py`.

## Góp ý cho module metric (gửi Tiến)

Phát hiện khi ghép `eval.py` với `dfd/metrics.py`:

1. **Ký hiệu không hợp lệ `|` trùng một lớp hợp lệ của bảng 229 lớp** (`|` là ký tự ASCII trong bảng). Nếu model dự đoán ra `|`, crop đó bị cho 1-NED = 0 và lỗi toàn bộ thành phần, thay vì được tính một phần. Nhãn VinText không có `|`, nên word acc không đổi, nhưng 1-NED và ER bị lệch nhẹ. Đề xuất đổi `INVALID_CHARACTER` sang một ký tự ngoài bảng, ví dụ `"�"`. `eval.py` dùng hằng số này nên tự theo.
2. **Module chưa xử lý không phân biệt hoa/thường.** `eval.py` đã chuyển chữ thường trước khi gọi, nên không cần sửa, chỉ cần biết để không làm hai lần.
3. `evaluate_predictions` được định nghĩa hai lần (bản stub cũ ở đầu file).
4. `rapidfuzz` chưa pin phiên bản. Trên nhánh này em đã pin `rapidfuzz==3.10.1`.
5. Vẫn cần chốt và ghi lại: mẫu số của Component ER (hiện là `max(len)`), và việc insert/delete tính lỗi cho cả ba thành phần. `eval.py` ghi thêm tổng số ký tự chèn và xóa vào cột `insertions`, `deletions`.

## Rà soát việc của Tiến (Task 4, 5, 6)

Nhánh `feat/vintext-data-preparation`, commit `9b3ab7e`. Không có dữ liệu VinText trên máy này, nên em rà soát bằng mã nguồn và output đã lưu trong notebook, đồng thời chạy lại các hàm chính trên dữ liệu giả.

**Kết luận: pipeline dữ liệu đúng protocol, không cần tạo lại crop, split hay LMDB.**

| Hạng mục | Kết quả |
|---|---|
| Crop, 10% lề trên/dưới, NFC, loại `###` và nhãn rỗng | Đúng. 43.045 crop, 0 lỗi, khớp 56.084 − 12.984 − 55 |
| Split theo ảnh, gom nhóm trùng, val 10%, seed cố định | Đúng. 120/1.200 ảnh vào val, train và val không chung ảnh |
| Tập con đóng băng, tần suất hiếm chỉ tính trên train | Đúng |
| LMDB | Đúng. Chạy hàm xuất của notebook rồi đọc bằng `LmdbDataset` của PARSeq thành công |

### Đã sửa

| Vấn đề | Cách sửa |
|---|---|
| **Notebook nhúng ảnh VinText trong khi repo public.** Proposal ghi rõ không phân phối lại | Xóa output ảnh ở `02` cell 7, `03` cell 19 và 20, thêm ghi chú trong cell. Giữ các biểu đồ thống kê |
| EDA: `~boxes.empty` là thuộc tính của DataFrame nên không lọc được box rỗng (43.100 thay vì 43.045) | Sửa thành `boxes["empty"]`. Không ảnh hưởng crop, split, LMDB |
| EDA: phần tổng hợp phát hiện vẫn là mẫu trống | Điền bằng số liệu từ output đã lưu, mục kiểm tra trực quan để Tiến điền |
| `04` dùng đường dẫn cứng `D:\BTL_DL\...` | Dò đường dẫn như notebook 02, 03. Trên máy Tiến vẫn ra đúng thư mục cũ |
| `requirements.txt` bị thêm jupyterlab, pandas... không pin và thiếu `lmdb` | Trả lại bản pin, chuyển sang `requirements-data.in`, thêm `lmdb` |
| Tên tập con lệch: config ghi `small_crop`, `rare_combination`; file là `small_crops`, `rare_combinations` | Sửa 76 YAML theo tên file đã đóng băng |
| README lỗi thời và cây thư mục sai | Cập nhật |
| Proposal ghi nhầm 500 test / 300 unseen | Sửa thành 300 test / 500 unseen, đúng dữ liệu |
| `scripts/check_scaffold.py` luôn lỗi trên Windows | So đường dẫn dạng POSIX, nay pass |

### Đã chốt

Các quyết định đã ghi vào `docs/proposal/` và `docs/IMPLEMENTATION.md`.

- **Số lớp:** 229 = 228 ký tự + EOS, teacher và student như nhau. `eos_policy` trong YAML đặt là `eos_class_0_none_level`.
- **Ký tự ngoài bảng:** train bỏ crop đó; val/test giữ nguyên nhãn và tính là sai.
- **"Mang dấu":** gồm modifier, tone hoặc cả hai, tính cả `đ/Đ`.
- **Rare combination:** theo (base không phân biệt hoa/thường, modifier, tone); từ thuộc tập con nếu chứa ít nhất một tổ hợp hiếm.
- **Chiều cao crop nhỏ:** đo trên tứ giác gốc, trước margin và resize.
- **Vị trí dấu thanh:** giữ như nhãn gốc, chỉ NFC, không viết lại `hoà/hòa`, vì nhãn phải khớp chữ trong ảnh.
- **Cặp ảnh gần trùng xuyên split** `im0894` (train) và `im1670` (unseen test): giữ split chính thức, ghi vào phần giới hạn của báo cáo.

### Không sửa, có chủ ý

- `đ` và `Đ` được đếm thành hai tổ hợp riêng trong khi nguyên âm gộp hoa/thường. Cả hai đều phổ biến nên không đổi thành viên tập hiếm, và sửa sẽ làm đổi checksum đã đóng băng.
- `freeze_manifest.json` chưa có checksum LMDB và commit hash. Sẽ ghi fingerprint ở metadata của từng run.

## Kết quả chạy

| Kiểm tra | Kết quả |
|---|---|
| `pytest` toàn repo, PyTorch CPU, có PARSeq | 132 passed (15/11, gồm test của Task 14, 15, 26 và 43 test metric của Tiến) |
| `scripts/check_scaffold.py` | Pass |
| Biên dịch mọi code cell của 4 notebook | Pass |

Các notebook **chưa được chạy lại trên dữ liệu thật**, vì dữ liệu chỉ có trên máy Tiến.

## Việc còn lại

1. **Xóa ảnh khỏi lịch sử GitHub (cần nhóm quyết định).** Commit `9b3ab7e` đã push vẫn chứa ảnh VinText. Sửa trong working tree không gỡ được phần đã public. Cách gỡ là thay commit đó rồi force-push nhánh của Tiến, hoặc xóa nhánh remote và push lại, sau đó báo Tiến lấy lại nhánh.
2. **Tiến chạy lại cell 6, 9 của `03_EDA`** để cập nhật số bị lẫn box rỗng, rồi xóa output ảnh ở cell 19, 20 trước khi commit.
3. ~~Tiến gửi riêng cho nhóm LMDB và thư mục `splits/`~~. Đã nhận ngày 16/11, xem mục "Dữ liệu thật". Vẫn không commit manifest vào repo vì có chứa nhãn.

## Kết quả đối chiếu charset

Tiến chạy `scripts/check_vocab_coverage.py` trên `04_character_vocabulary.csv` của VinText.

| Nhóm | Ký tự | Xử lý |
|---|---|---|
| Có trong dữ liệu, ngoài bảng | dấu cách (U+0020) | Bỏ khỏi nhãn ở cả train và đánh giá, đúng mặc định `remove_whitespace=True` của PARSeq |
| Có trong dữ liệu, ngoài bảng | `°` (U+00B0) | Train bỏ crop chứa ký tự này; val/test giữ nguyên nhãn và tính là sai |
| Có trong bảng, không có trong dữ liệu | `\|`, `Ỵ` | Giữ nguyên, các lớp này không ảnh hưởng gì |

Bảng giữ nguyên 229 lớp, fingerprint không đổi. Không thêm `°` vì nó không phân rã được theo base ASCII và sẽ đổi kích thước mọi head. Số crop chứa `°` nằm ở cột `count` của CSV, cần ghi vào báo cáo.

## Lưu ý cho các task khác

- **Task 9, 10, 12:** crop chứa `°` phải được xử lý riêng. Nếu để mặc định, `CharsetAdapter` biến `25°C` thành `25C`, sai với quy tắc đã chốt. Khi tính metric cũng phải bỏ dấu cách khỏi nhãn.

- **Task 9, 12 (Tuấn):** phải đặt `normalize_unicode=False` cho dataset của PARSeq. Mặc định nó biến `Việt` thành `Viet` và `ĐẸP` thành `EP` (đã chạy thử).
- **Task 9, 12 (Tuấn):** dùng `charset = "".join(table.characters[1:])` và `num_tokens = 231`.
- **Task 9, 10, 12:** `LmdbDataset` tự bỏ nhãn dài hơn 25 ký tự và âm thầm xóa ký tự ngoài charset (`€uro` thành `uro`), kể cả ở tập test. VinText có nhãn dài tới 30, nên đánh giá phải tính cả các mẫu này là sai để mẫu số không đổi giữa các model.
- **Task 10, 17, 18 (Tiến):** nối dự đoán với tập con bằng key `source-id` (`image_id:line_number`). `LmdbDataset` không đọc key này.
- **Task 18 (Tiến):** tập `rare_combinations` rất nhỏ: 6 val, 24 test, 33 unseen. Khi báo cáo phải ghi rõ số mẫu và không kết luận mạnh từ tập này.
- **Task 9, 12 (Tuấn):** để train `ti_factorized`, `ti_kd`, `ti_dfd` và E3, dùng `DistilledPARSeq` thay cho PARSeq gốc, theo mục "Ghép vào script train" ở trên.
