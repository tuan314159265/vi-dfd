---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Factorized output head

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

Thiết kế kế thừa ý tưởng của VietTDR [3], không nhận là đóng góp mới. Đầu vào là decoder hidden state $h_i$ của PARSeq-Ti tại vị trí $i$.

## Ba head song song

Thay classifier đơn khoảng 230 lớp bằng:

$$p_S^x(\cdot\mid i)=\operatorname{softmax}(W_xh_i),\qquad x\in\{b,m,t\}.$$

Ba head dự đoán base, modifier và tone trên cùng hidden state.

## Ghép thành ký tự hợp lệ

$$p_S(c\mid i)\propto p_S^b(b(c))\,p_S^m(m(c))\,p_S^t(t(c))\,\mathbb{1}[c\in\mathcal V].$$

Ở đây $\mathcal V$ gồm các ký tự hợp lệ và EOS. Chuẩn hóa tích xác suất trên $\mathcal V$ để có phân phối composed characters. Ký tự ghép được đưa vào context embedding của bước tiếp theo.

Có thể tính bằng tổng log-probabilities rồi chuẩn hóa trên danh sách triple hợp lệ để tránh underflow; đây là gợi ý triển khai, không thay đổi công thức.

## So với VietTDR

Không thêm tone-position attention prior hoặc triple input embedding. Product-with-mask composition là lựa chọn của proposal; backbone và input context embeddings giữ nguyên.

## Chi phí dự kiến

Theo ước lượng của bản gốc: ba head khoảng 20K tham số, classifier cũ khoảng 44K. Phần giảm kích thước chính đến từ Ti backbone. Cần đo latency của cả bước composition; không suy ra tốc độ chỉ từ số tham số.

| Thành phần | Vanilla Ti | DFD student |
|---|---|---|
| Encoder/decoder | PARSeq-Ti | Giữ nguyên |
| Head | Một classifier | Ba classifier + composition |
| Ground-truth objective | CE ký tự | CE components |
| Teacher | Không có | PARSeq-S frozen khi KD |
| Chi phí KD ở inference | — | Không có |

## Đối chứng cần thiết

- [Factorized head không KD](09%20-%20Baselines%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20hu%E1%BA%A5n%20luy%E1%BB%87n.md): tách lợi ích của head.
- [Joint KD trên factorized student](11%20-%20Ablation%20E3.md): tách lợi ích marginalization.
- [Bỏ validity mask](11%20-%20Ablation%20E3.md): kiểm tra tổ hợp không hợp lệ.

Xem [04 - Phân rã ký tự tiếng Việt](04%20-%20Ph%C3%A2n%20r%C3%A3%20k%C3%BD%20t%E1%BB%B1%20ti%E1%BA%BFng%20Vi%E1%BB%87t.md), [06 - DFD và hàm mất mát](06%20-%20DFD%20v%C3%A0%20h%C3%A0m%20m%E1%BA%A5t%20m%C3%A1t.md).
