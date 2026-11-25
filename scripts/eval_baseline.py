"""Export saved E1 baseline predictions and score using the shared evaluator."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['test', 'unseen_test'], choices=['validation', 'test', 'unseen_test'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from dfd.baselines import build_baseline, digest, predict_loader, write_predictions
    from dfd.teacher_data import VietnameseLmdb
    from dfd.vietchar import DecompositionTable
    from eval import main as evaluate
    run = args.run_dir.resolve()
    config = yaml.safe_load((run / 'resolved.yaml').read_text())
    summary = json.loads((run / 'training.json').read_text())
    if summary.get('status') != 'completed' or summary.get('smoke_test'):
        parser.error('Only completed non-smoke baseline runs can be scored')
    checkpoints = list((run / 'checkpoints').glob('best-*.pt'))
    if len(checkpoints) != 1:
        parser.error('Expected exactly one best .pt checkpoint')
    model = build_baseline(config, pretrained=False).to(args.device)
    checkpoint = torch.load(checkpoints[0], map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    chars = ''.join(DecompositionTable.from_characters().characters[1:])
    destination = run / 'baseline_predictions'
    for split in args.splits:
        folder = 'val' if split == 'validation' else split
        dataset = VietnameseLmdb(args.data_root / 'lmdb' / folder, chars)
        rows = predict_loader(model, DataLoader(dataset, batch_size=args.batch_size, num_workers=0), args.device)
        assert len(rows) == len(dataset)
        write_predictions(destination / f'{split}.csv', rows)
        print(f'{split}: exported {len(rows)} predictions', flush=True)
    result = evaluate(['--run-dir', str(run), '--predictions-dir', str(destination),
                       '--data-root', str(args.data_root / 'lmdb'),
                       '--subset-dir', str(args.data_root / 'splits/subsets'),
                       '--splits', *args.splits, '--replace'])
    # Preserve which checkpoint generated these CSVs alongside evaluator provenance.
    (run / 'baseline_export.json').write_text(json.dumps({
        'checkpoint': checkpoints[0].name, 'checkpoint_sha256': digest(checkpoints[0]),
        'splits': args.splits, 'source_commit': __import__('subprocess').check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    }, indent=2) + '\n')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
