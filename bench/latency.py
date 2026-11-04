"""Benchmark recognition latency (scaffold; backend pending)."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Standalone run YAML")
    args = parser.parse_args()
    parser.error('Not implemented: Implement same-machine ONNX Runtime timing: batch 1, four CPU threads, warm-up 50, median of 500 runs.')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
