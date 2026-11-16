# E1 — PARSeq-S teacher result (2026-11-09)

Task 9: completed locally on branch `feat/teacher-parseq-s`.

| Item | Value |
|---|---|
| Model | PARSeq-S, 23,935,717 parameters |
| Output vocabulary | 229 classes: 228 characters + EOS; tokenizer 231 tokens |
| Initialization | Official English `parseq-bb5792a6.pt`; new output head; 97 shared context embeddings remapped |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB |
| Training | 20 epochs, batch 64, seed 42, 7,280 optimizer steps |
| Learning rate | AdamW + OneCycleLR; base LR 7e-4; actual peak LR 1.75e-4 on one GPU |
| Training duration | 29.63 minutes |
| Train samples | 23,267; two unsupported degree-symbol samples excluded |
| Validation samples | 2,507, including one label longer than 25 characters |
| Selected checkpoint | Epoch 18, by full validation word accuracy |
| Validation word accuracy | 88.6717% |
| Diacritic-heavy word accuracy | 91.2568% on 183 words |
| Validation 1 − NED | 94.5966% |
| Final epoch word accuracy | 88.5919% |

Heavy-word accuracy and 1-NED above correspond to the same selected checkpoint.
Validation was performed by the AMP trainer; E4 requires its own FP32 evaluation.

## Provenance

- Run ID: `20261109T104440Z_teacher_s42_e478bc06`.
- Training source commit: `36ce735b81030530b64ddc6f1ce89f27da1f1304`.
- Effective config SHA-256: `3b5dc6e3054c618527e4a122ead49d19557e5107a66dbbe2ae9d3723732a212a`.
- Upstream PARSeq: `1902db043c029a7e03a3818c616c06600af574be`.
- Exported weights SHA-256: `83144cf851e1850671da94d7dc2e4d86e26688c5b621bdb1ef63ee8f2b1a99c1`.
- `configs/training_budget.yaml` records the common batch/epoch budget and CUDA probe.

## Handoff and machine-readable results (2026-11-16)

- [Teacher summary CSV](e1/teacher_s42_summary.csv): source/config hash, seed, budget and selected-checkpoint metrics.
- [Validation history CSV](e1/teacher_s42_validation.csv): all 20 epochs.
- Best Lightning `.ckpt` SHA-256: `e185cec4c1cb68d970a8cdd84e4c482ecd1dd4799c26a642f244117ad37bd609`.
- Private handoff ZIP: `Vi-DFD_Task9_Teacher_E1_20261109.zip`, containing the best `.ckpt`, results, configs and checksums. Share the ZIP privately; it is excluded from Git.
- The selected checkpoint filename uses zero-based epoch 17, corresponding to epoch 18 in the tables.
- This provides the teacher portion of Task 12; the full E1 baseline comparison remains pending.

## Local artifacts

Under `results/runs/20261109T104440Z_teacher_s42_e478bc06/` (ignored by Git):

- `teacher_parseq_s_229.pt`: compact weights and vocabulary/provenance metadata.
- `checkpoints/best-epoch=17-validation_word_acc=0.8867.ckpt`: selected Lightning checkpoint.
- `checkpoints/last.ckpt`: resumable epoch-20 checkpoint.
- `validation.csv`: all 20 epochs with source/config hash and seed.
- `summary.csv`, `summary.json`, `metrics.csv`: selected-checkpoint results.
- `initialization.json`, `data_identity.json`, `dataset_audit.json`, `environment.txt`.
- `export_verification.json`: strict reload and real-image forward checks.

Raw/processed data remains outside Git under `../datasets/vintext/`.
Frozen data counts match the team's saved notebook outputs; the heavy validation
subset matches the frozen CSV exactly by source ID. Compare original manifest
fingerprints before combining results produced independently on other machines.

## Scope

This run fine-tunes the teacher and fixes a shared budget. Full E1 credibility
comparisons with Ti, VietOCR and CRNN remain pending. Test and unseen-test accuracy
have not been evaluated. Student/KD/DFD training and component error metrics are
separate tasks. See [teacher instructions](TEACHER.md).
