"""Compare a dataset's character list (EDA 04_character_vocabulary.csv) with the fixed vocabulary."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfd.vietchar import DecompositionTable  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with a 'character' column")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as handle:
        observed = [row["character"] for row in csv.DictReader(handle)]
    table = DecompositionTable.from_characters()
    missing, unused = table.coverage(observed)
    print(f"Dataset characters: {len(set(observed))}; vocabulary classes: {len(table)} (including EOS)")
    print(f"Out of vocabulary ({len(missing)}): {[f'{c!r} U+{ord(c):04X}' for c in missing]}")
    print(f"In vocabulary but absent from this dataset ({len(unused)}): {''.join(unused)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
