# Results

Generated run directories are ignored by Git. No experiment has been completed.

`train.py --prepare-run` creates `results/runs/<run_id>/` with:

- `resolved.yaml`: effective config including seed overrides.
- `metadata.json` and `run.csv`: full source commit, config hash, seed, platform,
  detected GPU, runtime, unresolved choices, and status **prepared**.
- `metrics.csv`: headers only; actual evaluation must populate scores later.
- `environment.txt`: exact installed package versions at preparation time.

Seed values: 42/43/44; one-seed studies: 42. Reproducibility additionally needs
vocabulary/split/subset/data fingerprints and teacher checkpoint hash once data/model
integration is ready. These are not invented in a scaffold run.

Metric units for future evaluation: fractions [0,1] for accuracy/ER/1-NED, ms for
CPU latency, samples/s for GPU throughput, MiB for memory. Define component ER
denominators and alignment before populating the schema.

Collect separate per-run files from Colab/Kaggle/AWS; do not have notebooks append
concurrently to one CSV. Curated aggregate CSVs can be explicitly versioned later;
large checkpoints and dataset images remain outside Git.
