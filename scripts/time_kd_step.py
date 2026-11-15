"""Time one training step per student variant, to budget GPU hours at milestone 13.

Uses PARSeq-Ti (student) and PARSeq-S (teacher) at their official sizes, random
weights and random images: timing does not depend on the weights or the data.
Measures forward + backward + AdamW step, with AMP on CUDA, after warmup.

    python scripts/time_kd_step.py --device cuda --batch-size 384 --train-samples 38000

Requires strhub (PARSeq) on the path. Prints a CSV table; nothing is saved.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfd.distill import DistillationObjective, marginal_matrices  # noqa: E402
from dfd.parseq_kd import DistilledPARSeq, freeze, shared_permutation_loss  # noqa: E402
from dfd.vietchar import DecompositionTable  # noqa: E402


SIZES = {  # strhub configs/model/parseq.yaml and configs/experiment/parseq-tiny.yaml
    "tiny": dict(embed_dim=192, enc_num_heads=3, dec_num_heads=6),
    "small": dict(embed_dim=384, enc_num_heads=6, dec_num_heads=12),
}
KD = dict(temperature=2.0, weight=1.0, component_weights={"base": 1.0, "modifier": 1.0, "tone": 2.0})
VARIANTS = {
    "ti_ce": (False, DistillationObjective(enabled=False)),
    "ti_factorized": (True, DistillationObjective(enabled=False)),
    "ti_kd": (False, DistillationObjective(enabled=True, mode="joint", **KD)),
    "ti_dfd": (True, DistillationObjective(enabled=True, mode="marginal", tone_confidence_weight=True, **KD)),
}


def build(size: str, charset: str, batch_size: int) -> DistilledPARSeq:
    return DistilledPARSeq(
        charset_train=charset, charset_test=charset, max_label_length=25, batch_size=batch_size,
        lr=7e-4, warmup_pct=0.075, weight_decay=0.0, img_size=[32, 128], patch_size=[4, 8],
        enc_mlp_ratio=4, enc_depth=12, dec_mlp_ratio=4, dec_depth=1, perm_num=6,
        perm_forward=True, perm_mirrored=True, decode_ar=True, refine_iters=1, dropout=0.1,
        **SIZES[size],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--label-length", type=int, default=8, help="Characters per synthetic label")
    parser.add_argument("--train-samples", type=int, help="Training crops per epoch, to project epoch time")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    device = torch.device(args.device)
    table = DecompositionTable.from_characters()
    charset = "".join(table.characters[1:])
    matrices = {name: matrix.to(device) for name, matrix in marginal_matrices(table).items()}
    # Vietnamese-looking labels of a fixed length so every variant sees the same shapes.
    pool = "aăâeêioôơuưyđàảãáạằẳẵắặầẩẫấậbcdghklmnpqrstvx"
    labels = [(pool * 4)[index % len(pool):][: args.label_length] for index in range(args.batch_size)]
    images = torch.randn(args.batch_size, 3, 32, 128, device=device)
    teacher = freeze(build("small", charset, args.batch_size)).to(device)
    amp = device.type == "cuda"

    print("variant,ms_per_step_median,ms_per_step_stdev,peak_memory_mb,projected_epoch_min")
    for name in args.variants:
        factorized, objective = VARIANTS[name]
        torch.manual_seed(0)
        student = build("tiny", charset, args.batch_size)
        if factorized:
            student.use_factorized_head(table)
        student = student.to(device).train()
        optimizer = torch.optim.AdamW(student.parameters(), lr=7e-4)
        scaler = torch.amp.GradScaler(enabled=amp)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for step in range(args.warmup + args.steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device.type, enabled=amp):
                loss = shared_permutation_loss(
                    student, teacher if objective.enabled else None, images, labels, objective, matrices
                )["loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= args.warmup:
                times.append((time.perf_counter() - start) * 1000)
        median = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
        epoch = median * -(-args.train_samples // args.batch_size) / 60000 if args.train_samples else float("nan")
        print(f"{name},{median:.1f},{stdev:.1f},{memory:.0f},{epoch:.2f}")
    print(f"# device={device} batch={args.batch_size} label_length={args.label_length} amp={amp}; excludes data loading and validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
