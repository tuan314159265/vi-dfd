"""Export a PARSeq checkpoint to ONNX (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Integrate composed-character decoding and validate ONNX output equivalence against PyTorch.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
