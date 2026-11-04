---
date: 2026-11-04
status: proposal
tags: [BTL, DeepLearning, OCR, Vi-DFD]
---

# Diacritic-Factorized Distillation

[← Tổng quan BTL](00%20-%20T%E1%BB%95ng%20quan.md)

## Tín hiệu từ teacher

Teacher PARSeq-S được fine-tune với ordinary full-vocabulary head rồi đóng băng. Tại vị trí $i$, làm mềm logits bằng temperature $\tau$:

$$p_{T,i}(\tau)=\operatorname{softmax}(z_{T,i}/\tau).$$

Cộng xác suất theo bảng phân rã cố định:

$$p_{T,i}^{x}=p_{T,i}(\tau)M_x,\qquad x\in\{b,m,t\}.$$

Ví dụ tone marginal:

$$p_{T,i}^{t}(k)=\sum_{c:t(c)=k}p_T(c\mid i,\tau).$$

Teacher không cần thay head hay huấn luyện lại theo factorized architecture. “Không retrain teacher” ở đây không bỏ bước fine-tune tiếng Việt đã nêu trong [03 - PARSeq và cấu hình mô hình](03%20-%20PARSeq%20v%C3%A0%20c%E1%BA%A5u%20h%C3%ACnh%20m%C3%B4%20h%C3%ACnh.md).

## Objective trong proposal

$$
\mathcal L=\sum_i\sum_{x\in\{b,m,t\}}\operatorname{CE}(y_i^x,p_{S,i}^x)
+\lambda\tau^2\sum_i\alpha_i\sum_{x\in\{b,m,t\}}w_x\operatorname{KL}(p_{T,i}^x\Vert p_{S,i}^x(\tau)).
$$

Trong đó:

- CE dùng nhãn ground truth của ba components, với phân phối student ở temperature 1.
- Student softened distribution: $p_{S,i}^x(\tau)=\operatorname{softmax}(z_{S,i}^x/\tau)$.
- KL có chiều teacher → student.
- $\lambda$ cân bằng CE và KD; hệ số $\tau^2$ theo cách dùng KD trong proposal.
- $w_x$ là trọng số component; $\alpha_i$ là trọng số confidence tại vị trí.

## Điều kiện alignment trong PARSeq

Teacher và student nhận **cùng ảnh đã augmentation, cùng permutation, cùng ground-truth context**. Teacher chạy không gradients. Điều này đảm bảo matching đúng vị trí và cùng thông tin context; không dùng hai chuỗi tự sinh độc lập rồi KL theo index.

- [ ] Mask padding và xử lý EOS nhất quán.
- [ ] Chốt loss reduction theo valid positions, batch và các permutation.
- [ ] Log riêng CE/KL theo component để theo dõi độ lớn loss.

## Tone-focused weighting — biến thể nhỏ

Đặt $w_t>w_b=w_m$. Confidence weight dựa trên entropy tone marginal của teacher:

$$\alpha_i=1-\frac{H(p_{T,i}^{t})}{\log6}.$$

Teacher phân phối đều sáu thanh → $\alpha_i=0$; teacher chắc chắn → trọng số tiến gần 1. Theo công thức gốc, $\alpha_i$ nhân **toàn bộ KD term của vị trí**, không chỉ tone KL. CE không bị nhân $\alpha_i$.

Tone-focused weighting phải được kiểm tra riêng bằng ablation $w_t=1,\alpha_i=1$ trong [11 - Ablation E3](11%20-%20Ablation%20E3.md).

## Vì sao kỳ vọng có ích?

1. Sáu tone classes nhận tín hiệu cộng gộp từ nhiều nguyên âm thay vì rải trên nhiều ký tự hoa/thường hiếm.
2. Có thể dùng teacher full-vocabulary sẵn có bằng bảng marginalization.
3. KD chỉ chạy khi train; inference chỉ dùng student.

Đây là lập luận cần kiểm chứng, chưa có kết quả chứng minh. So sánh chính ở [E2](10%20-%20K%E1%BA%BF%20ho%E1%BA%A1ch%20th%C3%AD%20nghi%E1%BB%87m%20E1-E7.md) và ablation ở [11 - Ablation E3](11%20-%20Ablation%20E3.md).

Liên quan: Ghi chú chưng cất mô hình, [04 - Phân rã ký tự tiếng Việt](04%20-%20Ph%C3%A2n%20r%C3%A3%20k%C3%BD%20t%E1%BB%B1%20ti%E1%BA%BFng%20Vi%E1%BB%87t.md), [05 - Factorized output head](05%20-%20Factorized%20output%20head.md).
