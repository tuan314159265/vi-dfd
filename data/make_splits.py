"""Freeze image-level dataset splits (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Implement fixed-seed image splitting and near-duplicate groups before creating any crops for training.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
