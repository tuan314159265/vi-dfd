# Coverage of the supplied artifact

Source: proposal pasted by the user in this conversation. The original artifact URL
could not be fetched; the pasted content is the source of truth for this scaffold.

## Original repository layout

```text
vi-dfd/
├── configs/            # 1 file YAML cho mỗi lần chạy trong ma trận E1–E6
├── data/               # chỉ script, KHÔNG chứa ảnh VinText
│   ├── crop_vintext.py
│   ├── make_splits.py  # chia theo ảnh + nhóm ảnh gần trùng, có seed
│   └── subsets.py      # định nghĩa tập con, đóng băng ở tuần 1
├── dfd/
│   ├── vietchar.py     # bảng phân rã/ghép ký tự + unit test
│   ├── heads.py        # head phân rã + ghép có mask
│   ├── distill.py      # ma trận marginalize + loss DFD
│   └── metrics.py      # word acc, 1-NED, Tone/Modifier/Base ER, ma trận nhầm
├── train.py, eval.py
├── bench/              # đo độ trễ ONNX Runtime, script có ghi phần cứng
├── serve/              # FastAPI + Dockerfile (demo)
├── results/            # CSV kết quả từng lần chạy + commit hash
└── README.md
```

## What exists now

| Requested item | Initial state |
|---|---|
| `configs/` | 76 standalone run YAMLs, job manifest, tuning policy; candidate/unset values explicitly marked |
| `data/crop_vintext.py` | CLI scaffold with required preprocessing documented |
| `data/make_splits.py` | CLI scaffold; image-level grouping protocol documented |
| `data/subsets.py` | CLI scaffold; subset definitions documented |
| `dfd/vietchar.py` | Working NFC decomposition/composition; tests; vocabulary must come from actual data |
| `dfd/heads.py` | Explicit pending interface and composition specification |
| `dfd/distill.py` | Explicit pending interfaces and DFD loss specification |
| `dfd/metrics.py` | Explicit pending interface and metric conventions |
| `train.py` | Working config check and artifact preparation; training backend pending |
| `eval.py` | CLI scaffold; shared evaluation backend pending |
| `bench/` | Export/quantization/latency CLI scaffolds plus benchmark protocol |
| `serve/` | FastAPI skeleton with health, `/recognize` returning 501, CPU Dockerfile draft |
| `results/` | CSV schema documented; prepared runs record commit/config hash and seed; no scores fabricated |
| `README.md` | Setup, status, reproducibility, data restrictions and attribution |

## Additional requirements from the conversation

- Pinned scaffold requirements and a clearly separated draft GPU dependency list.
- Fixed seeds 42/43/44 and resolved YAML saved per prepared run.
- Full Git commit hash and config hash in CSV metadata.
- Each member prioritizes Colab Free + Kaggle; AWS fallback if needed.
- Same-machine CPU latency comparison, despite distributed model training.
- Proposal notes exported under `docs/proposal/` to retain RQs, method, E1–E7,
  ablations, result templates, limitations, timeline, references and rubric.

Run `python scripts/check_scaffold.py` to check all original paths, every planned
config and the manifest. This checks scaffold completeness, not research correctness.
