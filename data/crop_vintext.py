"""Prepare VinText word crops (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Verify annotations and implement perspective warp with vertical margin, NFC labels and illegible filtering.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
