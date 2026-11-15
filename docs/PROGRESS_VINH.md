# Tiến độ các task của Vinh

Cập nhật: 15/11/2026. Task 7 và 11 nằm ở nhánh `feat/vocab-factorized-head`, tách từ `main`. Task 14 và phần chuẩn bị Task 15 nằm ở nhánh `feat/dfd-distillation`, tách từ `feat/vocab-factorized-head`. Các sửa cho notebook của Tiến nằm ở nhánh `feat/vintext-data-preparation`.

## Trạng thái

| # | Task | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Proposal chi tiết | Done (từ trước) | |
| 7 | Bảng phân rã ký tự | **Done** | Đã đối chiếu với charset VinText, bảng giữ nguyên 229 lớp |
| 11 | Head phân rã cho student | **Done** | Đã kiểm chứng trên PARSeq thật |
| 14 | Cài DFD và KD thường | **Done về code** | Đã chạy `Trainer.fit` trên PARSeq thật với teacher giả. Còn ghép vào script train của Tuấn khi Task 9 được push |
| 15 | E2 sơ bộ, chọn siêu tham số | **Đang chờ** | Đã có script sinh grid và chọn cấu hình. Chờ teacher, recipe, pipeline đánh giá, quyết định ngân sách, mốc 13. Xem "Cần nhắc" |
| 16 | E2 đủ 3 seed + E3 | Chờ Task 15 | |
| 25 | Đóng gói và hoàn tất nội bộ | Chờ thí nghiệm và báo cáo | |

Các task "Cả nhóm" (13, 20, 23, 24) phụ thuộc kết quả chưa có. Cho mốc 13 em đã chuẩn bị script đo thời gian step có KD.

## Cần nhắc (việc đang chặn task của Vinh)

| Ai | Việc | Chặn task | Ghi chú |
|---|---|---|---|
| **Tuấn** | Push code train teacher (Task 9) và chia sẻ checkpoint `.ckpt` qua kênh private | 14 (ghép), 15 | Repo chưa có code hay checkpoint của Task 9 dù bảng ghi Done. Checkpoint phải là checkpoint Lightning của `strhub` PARSeq, `charset_train` đúng 228 ký tự theo thứ tự bảng. Kiểm tra bằng lệnh ở mục "Ghép vào script train" |
| **Tuấn** | Báo batch size, số epoch, lr đã chốt ở Task 9 | 15 | Để điền `training.*` (đang `null`) trong YAML của E2/E3. Báo kèm word acc validation của teacher |
| **Tuấn** | Kết quả E1 (Task 12), nhất là Ti vanilla | 15, mốc 13 | Ti vanilla là mốc so sánh của E2 |
| **Tiến** | Hoàn tất Task 10 và mở PR | 15 | Nhánh `origin/feat/metrics-evaluation` (commit `03109a4`, 15/11) đã có `dfd/metrics.py`, bảng vẫn ghi In progress |
| **Tiến** | Gửi LMDB và `splits/` cho Vinh qua kênh private | 15 | Tài khoản GPU của Vinh cần dữ liệu để chạy |
| **Cả nhóm** | Chọn người làm `eval.py`: chạy checkpoint trên LMDB, gọi `dfd/metrics.py`, ghi `metrics.csv` | 15 | Task 10 chỉ là module metric, Task 12 chỉ dùng nó. `scripts/e2_tuning.py select` cần mỗi run có một dòng `split=validation`, `subset=all` trong `metrics.csv` |
| **Cả nhóm** | Chốt `equal_budget_policy` trong `configs/tuning.yaml` | 15 | Grid DFD có 12 trial, grid KD thường chỉ 6. Em đề xuất mở rộng $\lambda$ của KD thường thành {0.25, 0.5, 1, 2} để có 3 × 4 = 12 trial. `e2_tuning.py make` từ chối chạy khi giá trị này còn `null` |
| **Cả nhóm (mốc 13)** | Chạy `scripts/time_kd_step.py` trên GPU thật | 13, 15 | Ra ms/step cho 4 biến thể và thời gian 1 epoch dự kiến, để chốt ngân sách GPU và số seed |
| **Người chạy E3 (Tuấn)** | Chốt `joint_lambda` cho `marginal_joint_kd` | 16 | Loss báo lỗi khi giá trị còn `null`. Ngoài ra vẫn còn mở cách decode khi bỏ validity mask |

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

- `scripts/e2_tuning.py make` sinh 1 YAML cho mỗi trial, seed 42, chỉ đánh giá trên validation, vào `configs/tuning_runs/E2/`. Grid lấy từ `configs/tuning.yaml`. Script từ chối chạy khi `equal_budget_policy` còn `null`.
- `scripts/e2_tuning.py select` đọc `results/runs/*/resolved.yaml` và `metrics.csv`, xếp hạng theo word acc validation, hòa thì xét 1-NED rồi thứ tự grid. Nó in cả các trial chưa có kết quả.
- `scripts/time_kd_step.py` đo ms/step và bộ nhớ đỉnh cho `ti_ce`, `ti_factorized`, `ti_kd`, `ti_dfd` với kích thước Ti/S chính thức. Lệnh cho mốc 13, thay số crop train thật vào:

  ```bash
  python scripts/time_kd_step.py --device cuda --batch-size <batch của Task 9> --train-samples <số crop train>
  ```

- Khi đủ điều kiện: điền `training.*`, `data.*`, `model.teacher_checkpoint` vào hai YAML gốc E2 (`ti_dfd_s42`, `ti_kd_s42`), chạy `make`, train từng trial, chạy `select`, rồi ghi cấu hình đã chọn vào 12 YAML E2 ba seed. Nếu xin compute, gửi bảng kết quả `select` cho giảng viên.

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
| `pytest` toàn repo, PyTorch CPU, có PARSeq | 83 passed (15/11, gồm 36 test mới của Task 14, 15) |
| `scripts/check_scaffold.py` | Pass |
| Biên dịch mọi code cell của 4 notebook | Pass |

Các notebook **chưa được chạy lại trên dữ liệu thật**, vì dữ liệu chỉ có trên máy Tiến.

## Việc còn lại

1. **Xóa ảnh khỏi lịch sử GitHub (cần nhóm quyết định).** Commit `9b3ab7e` đã push vẫn chứa ảnh VinText. Sửa trong working tree không gỡ được phần đã public. Cách gỡ là thay commit đó rồi force-push nhánh của Tiến, hoặc xóa nhánh remote và push lại, sau đó báo Tiến lấy lại nhánh.
2. **Tiến chạy lại cell 6, 9 của `03_EDA`** để cập nhật số bị lẫn box rỗng, rồi xóa output ảnh ở cell 19, 20 trước khi commit.
3. **Tiến gửi riêng cho nhóm** LMDB và thư mục `splits/` qua kênh private. Task 9 cần chúng. Không commit manifest vào repo public vì có chứa nhãn.

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
