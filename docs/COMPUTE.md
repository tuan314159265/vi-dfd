# Compute and collaboration

Each member prioritizes their own Colab Free and Kaggle accounts. Record which
platform/GPU actually ran each job; no specific free quota or GPU model is assumed.
AWS is the fallback if free compute is insufficient for the experiment deadline.
No cloud instance is created by this scaffold.

Use `configs/run_manifest.csv` to assign each exact experiment/config/seed before
running it. Owner and actual platform are not assigned yet. Keep the same data manifests,
source commit, teacher checkpoint and resolved configuration across accounts.

Save periodic resumable checkpoints outside ephemeral notebook storage. On resume,
restore model, optimizer, scheduler, AMP scaler, sampler and RNG states. Preserve run
identity so resumed execution is not counted as a new independent seed.

Artifacts to hand over: resolved YAML, run metadata, environment snapshot,
checkpoint/hash, predictions and per-run CSV. Merge only after checking job identity.
CPU latency must be measured centrally on the same machine/session. GPU training
hardware differences must not be presented as model-inference speed differences.

Before using AWS, settle operator, spending/time budget and job-completion shutdown.
Internal deadline: 2026-12-10. Baseline milestone: 2026-11-13. Experiment freeze: 2026-11-27.
