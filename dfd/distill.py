"""DFD loss interface (implementation pending).

Teacher and student must share augmentation, permutation and ground-truth context.
Marginals: p_teacher_x = softmax(teacher_logits / tau) @ M_x.
Loss: component CE + lambda * tau**2 * sum_i alpha_i * sum_x w_x * KL(T_x || S_x).
alpha_i = 1 - H(teacher_tone_marginal) / log(6); mask padding consistently.
Follow Structural KD for the marginal-distillation principle; do not claim it as new.
"""


def marginalize_teacher(teacher_probabilities, component_matrices):
    raise NotImplementedError("Build fixed mapping matrices from the verified vocabulary, then multiply probabilities by each matrix")


def dfd_loss(student_logits, teacher_logits, targets, masks, config):
    raise NotImplementedError("Implement and verify component CE, temperature KL, tone/confidence weights and padding/EOS masking")
