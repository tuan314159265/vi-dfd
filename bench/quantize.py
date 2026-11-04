"""Apply dynamic INT8 quantization (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Implement ONNX Runtime quantization and compare accuracy to FP32 for the same checkpoint.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
