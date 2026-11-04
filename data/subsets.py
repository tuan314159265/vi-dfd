"""Freeze evaluation subsets (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Implement heavy-word, height<16px and rare<20 train-occurrence subsets before model training.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
