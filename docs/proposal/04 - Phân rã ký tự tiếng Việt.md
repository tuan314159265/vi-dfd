---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Phân rã ký tự tiếng Việt

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Ánh xạ xác định

Mỗi ký tự có ánh xạ cố định, không học:

$$c \longmapsto (b(c),m(c),t(c))$$

| Ký tự | Base b | Modifier m | Tone t |
|---|---|---|---|
| a | a | none | ngang |
| ắ | a | breve | sắc / acute |
| ệ | e | circumflex | nặng / dot below |
| ờ | o | horn | huyền / grave |
| ữ | u | horn | ngã / tilde |
| đ | d | bar | ngang |
| 7 | 7 | none | ngang |

## Component vocabularies dự kiến

- **Base:** chữ hoa/thường, chữ số, dấu câu và EOS; khoảng 95 lớp, chốt từ character set thực tế.
- **Modifier:** 5 lớp — none, circumflex, breve, horn, bar.
- **Tone:** 6 lớp — ngang, sắc, huyền, hỏi, ngã, nặng.
- **Composed vocabulary:** 229 lớp gồm 228 ký tự (94 ASCII + 134 chữ tiếng Việt) và EOS.

Chỉ các triple có trong vocabulary mới hợp lệ. Ví dụ `q + tilde` không đại diện cho ký tự hợp lệ trong bộ này.

## Ma trận phân rã

Với $x\in\{b,m,t\}$, lập ma trận nhị phân:

$$M_x[c,k]=\mathbb{1}[x(c)=k].$$

Mỗi hàng của $M_x$ ánh xạ một ký tự sang đúng một component class. Cùng bảng được dùng cho nhãn ground truth, marginalization của teacher và ghép đầu ra của student.

## Checklist cho `dfd/vietchar.py`

- [ ] Chốt danh sách ký tự thực tế và thứ tự class; không suy ra chỉ từ con số 229.
- [ ] Chuẩn hóa NFC trước khi phân rã.
- [ ] Giữ phân biệt hoa/thường trong bảng; xử lý case-insensitive tại evaluation theo protocol.
- [ ] Kiểm tra round trip `compose(decompose(c)) == c` cho toàn bộ vocabulary.
- [x] Chốt cách biểu diễn EOS ở ba head và các token điều khiển của PARSeq.
- [x] Chốt cách xử lý ký tự ngoài vocabulary: train bỏ crop đó; val/test giữ nguyên nhãn và tính là sai; không để `CharsetAdapter` của PARSeq âm thầm xóa ký tự.
- [ ] Kiểm tra không có hai ký tự khác nhau dùng cùng triple.
- [ ] Hand-verify các dấu kép và đ/Đ; lưu bảng với phiên bản cố định.

> [!note] Đã chốt
> Teacher và student đều 229 lớp, gồm EOS ở lớp 0 theo thứ tự đầu ra của PARSeq. EOS là triple `([EOS], none, level)`. Ba ma trận có shape `229 × 95`, `229 × 5`, `229 × 6`. Còn phải đối chiếu danh sách ký tự thực tế bằng `scripts/check_vocab_coverage.py`.

Xem [05 - Factorized output head](05%20-%20Factorized%20output%20head.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md).
