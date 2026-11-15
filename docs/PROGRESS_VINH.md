# Tiến độ các task của Vinh

Cập nhật: 09/11/2026. Task 7 và 11 nằm ở nhánh `feat/vocab-factorized-head`, tách từ `main`. Các sửa cho notebook của Tiến nằm ở nhánh `feat/vintext-data-preparation`.

## Trạng thái

| # | Task | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Proposal chi tiết | Done (từ trước) | |
| 7 | Bảng phân rã ký tự | **Done** | Còn một lệnh đối chiếu charset do Tiến chạy, xem mục "Việc còn lại" |
| 11 | Head phân rã cho student | **Done** | Đã kiểm chứng trên PARSeq thật |
| 14 | Cài DFD và KD thường | Chưa làm | Cần Task 9: "teacher và student chạy cùng hoán vị" cần vòng train có teacher thật |
| 15 | E2 sơ bộ, chọn siêu tham số | Chưa làm | Cần Task 9, 10, 12, 14 và mốc 13 |
| 16 | E2 đủ 3 seed + E3 | Chưa làm | Cần Task 15 |
| 25 | Đóng gói và hoàn tất nội bộ | Chưa làm | Cần toàn bộ thí nghiệm và báo cáo |

Các task "Cả nhóm" (13, 20, 23, 24) cũng phụ thuộc kết quả chưa có nên không làm.

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
| `pytest` toàn repo, PyTorch CPU, có PARSeq | 47 passed |
| `scripts/check_scaffold.py` | Pass |
| Biên dịch mọi code cell của 4 notebook | Pass |

Các notebook **chưa được chạy lại trên dữ liệu thật**, vì dữ liệu chỉ có trên máy Tiến.

## Việc còn lại

1. **Xóa ảnh khỏi lịch sử GitHub (cần nhóm quyết định).** Commit `9b3ab7e` đã push vẫn chứa ảnh VinText. Sửa trong working tree không gỡ được phần đã public. Cách gỡ là thay commit đó rồi force-push nhánh của Tiến, hoặc xóa nhánh remote và push lại, sau đó báo Tiến lấy lại nhánh.
2. **Tiến chạy đối chiếu charset:**

   ```bash
   python scripts/check_vocab_coverage.py results/eda/tables/04_character_vocabulary.csv
   ```

   Bằng nhau về số lượng (228) chưa chứng minh trùng nhau về nội dung. Nếu chỉ dư dấu cách thì không sao, vì PARSeq bỏ khoảng trắng. Nếu có ký tự khác thì áp dụng quy tắc ở trên, và cân nhắc thêm vào bảng (fingerprint sẽ đổi).
3. **Tiến chạy lại cell 6, 9 của `03_EDA`** để cập nhật số bị lẫn box rỗng, rồi xóa output ảnh ở cell 19, 20 trước khi commit.
4. **Tiến gửi riêng cho nhóm** LMDB và thư mục `splits/` qua kênh private. Task 9 cần chúng. Không commit manifest vào repo public vì có chứa nhãn.

## Lưu ý cho các task khác

- **Task 9, 12 (Tuấn):** phải đặt `normalize_unicode=False` cho dataset của PARSeq. Mặc định nó biến `Việt` thành `Viet` và `ĐẸP` thành `EP` (đã chạy thử).
- **Task 9, 12 (Tuấn):** dùng `charset = "".join(table.characters[1:])` và `num_tokens = 231`.
- **Task 9, 10, 12:** `LmdbDataset` tự bỏ nhãn dài hơn 25 ký tự và âm thầm xóa ký tự ngoài charset (`€uro` thành `uro`), kể cả ở tập test. VinText có nhãn dài tới 30, nên đánh giá phải tính cả các mẫu này là sai để mẫu số không đổi giữa các model.
- **Task 10, 17, 18 (Tiến):** nối dự đoán với tập con bằng key `source-id` (`image_id:line_number`). `LmdbDataset` không đọc key này.
- **Task 18 (Tiến):** tập `rare_combinations` rất nhỏ: 6 val, 24 test, 33 unseen. Khi báo cáo phải ghi rõ số mẫu và không kết luận mạnh từ tập này.
- **Task 14:** dùng `torch.tensor(table.matrix(x), dtype=torch.float)` cho ma trận marginalize và `head.component_logits` cho logit student.
