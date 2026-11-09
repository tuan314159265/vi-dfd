"""Run the team's existing crop/split notebooks headlessly, outside the public repo.

Notebook source cells remain the implementation of Tasks 4–6. Only path assignments
and display are adapted; no split seed, crop geometry or subset threshold is changed.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile


REPO = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NotebookPaths(ast.NodeTransformer):
    def __init__(self, values):
        self.values = values

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in self.values:
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(self.values[name]))], keywords=[])
        return node


def execute_notebook(path, overrides):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "display": lambda *args, **kwargs: None}
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        sources.append(source)
        if path.name == "02_crop_vintext.ipynb" and "for record in random.sample" in source:
            continue  # Private example images/transcripts are unnecessary for preparation.
        tree = NotebookPaths(overrides).visit(ast.parse(source))
        ast.fix_missing_locations(tree)
        print(f"Running {path.name} cell {index}", flush=True)
        exec(compile(tree, f"{path.name}:cell{index}", "exec"), namespace)
        namespace["display"] = lambda *args, **kwargs: None
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def verify_existing(processed):
    import lmdb

    freeze_path = processed / "splits/freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for name, expected in freeze["manifest_sha256"].items():
        if sha256_file(processed / "splits" / name) != expected:
            raise ValueError(f"Frozen manifest changed: {name}")
    for split, expected in freeze["word_crop_counts"].items():
        with lmdb.open(str(processed / "lmdb" / split), readonly=True, lock=False) as env:
            with env.begin() as txn:
                if int(txn.get(b"num-samples")) != expected:
                    raise ValueError(f"LMDB count mismatch: {split}")
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == REPO or REPO in workspace.parents:
        parser.error("Keep VinText and generated labels outside the public Git repository")
    raw_parent = workspace / "raw"
    raw = raw_parent / "vietnamese"
    processed = workspace / "processed/vintext_notebook"
    workspace.mkdir(parents=True, exist_ok=True)
    if (processed / "splits/freeze_manifest.json").exists():
        freeze = verify_existing(processed)
        print("Reusing verified frozen data:", freeze["word_crop_counts"])
        return
    if not raw.exists():
        raw_parent.mkdir(exist_ok=True)
        with ZipFile(args.archive) as archive:
            for entry in archive.infolist():
                destination = (raw_parent / entry.filename).resolve()
                if not destination.is_relative_to(raw_parent.resolve()):
                    raise ValueError("Archive contains a path outside the extraction directory")
            archive.extractall(raw_parent)
    crop = REPO / "notebooks/02_crop_vintext.ipynb"
    split = REPO / "notebooks/04_dataset_splitting.ipynb"
    crop_hash = execute_notebook(crop, {"DATASET": raw, "OUTPUT": processed})
    split_hash = execute_notebook(split, {"DATASET_ROOT": raw, "PROCESSED_ROOT": processed})
    freeze = verify_existing(processed)

    import sys
    sys.path.insert(0, str(REPO))
    from dfd.vietchar import DecompositionTable

    table = DecompositionTable.from_characters()
    (processed / "vocabulary.json").write_text(json.dumps({
        "characters": table.characters, "fingerprint": table.fingerprint(),
        "eos_id": 0, "output_classes": len(table), "num_tokens": len(table) + 2,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "source_archive_sha256": sha256_file(args.archive),
        "crop_notebook_source_sha256": crop_hash,
        "split_notebook_source_sha256": split_hash,
        "commit_hash": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "word_crop_counts": freeze["word_crop_counts"],
        "vocabulary_fingerprint": table.fingerprint(),
        "lmdb_sha256": {name: sha256_file(processed / "lmdb" / name / "data.mdb") for name in freeze["word_crop_counts"]},
    }
    (processed / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified and frozen:", report["word_crop_counts"], flush=True)
    print("Processed root:", processed, flush=True)


if __name__ == "__main__":
    main()
