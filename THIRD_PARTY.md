# Third-party sources and attribution

No PARSeq/VietTDR source code or dataset has been copied into this scaffold.

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

These links identify upstream references, not a validated dependency lock or code import.
