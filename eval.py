"""Evaluate a recognition checkpoint (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Implement the shared NFC, word accuracy, 1-NED, aligned component ER and tone-confusion evaluation pipeline.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
