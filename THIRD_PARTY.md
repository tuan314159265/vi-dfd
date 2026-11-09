# Third-party sources and attribution

No upstream PARSeq/VietTDR implementation or dataset is vendored in this repository.

The teacher imports PARSeq from a separate checkout pinned to
`1902db043c029a7e03a3818c616c06600af574be`. Training/permutation logic and the
AdamW/one-cycle optimizer remain upstream implementations. Keep that checkout's
LICENSE/NOTICE. The pretrained English checkpoint URL and full SHA-256 are recorded
with each run. Direct teacher dependencies are pinned in `requirements-teacher.txt`;
the actual package environment is snapshotted in the run artifacts.

Before integration:

- Pin PARSeq to a full upstream commit and record the pretrained checkpoint hash.
- Review and retain the applicable upstream LICENSE/NOTICE.
- Verify the exact VietTDR repository and credit borrowed implementation separately
  from the inherited head concept.
- Cite Structural KD for marginal distillation and Hinton et al. for standard KD.
- Verify VinText and optional dataset terms; do not redistribute images/crops here.

Primary PARSeq references inspected while preparing the scaffold:

- https://github.com/baudm/parseq
- https://github.com/baudm/parseq/blob/main/requirements/core.txt
- https://github.com/baudm/parseq/blob/main/requirements/train.txt
- https://github.com/baudm/parseq/blob/main/LICENSE
- https://github.com/baudm/parseq/blob/main/NOTICE

These links identify upstream references. The pinned checkout and actual teacher
environment are documented in `docs/TEACHER.md` and each run's metadata.
