# Teacher split verification — 24/11/2026

Re-ran the comparison of the reference data and the original local teacher data build:

```bash
python scripts/check_data.py --data-root ../datasets/vintext/tien_20261116 --compare-assignments ../datasets/vintext/processed/vintext_notebook
```

Observed output:

```text
2000/2000 images in the same split; 0 differ
```

The two builds differ in freeze manifest / crop bytes, but the image_id-to-final_split assignments match. This satisfies decision 2 for retaining the frozen teacher. All students use the reference build from Tien. Do not claim byte-identical teacher/student data. The warning `teacher_same_freeze_manifest: false` remains factually correct.

The original local freeze manifest SHA-256 is `63f1e1ca4e7c3a800ade2b89b7e60861fb70cf0dcb8e7fda81187cea80a28669`, matching the teacher run data_identity.json. This ties the compared original build to the recorded teacher training build.
