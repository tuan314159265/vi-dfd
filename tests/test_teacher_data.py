import io

import lmdb
import pytest
from PIL import Image

from dfd.teacher_data import VietnameseLmdb, is_heavy, normalize_label
from dfd.vietchar import DecompositionTable


@pytest.fixture
def lmdb_path(tmp_path):
    folder = tmp_path / "lmdb"
    labels = ["Việt", "ĐẸP", "€uro", "a" * 30, "a\u0306\u0301", "Hà Nội"]
    content = io.BytesIO()
    Image.new("RGB", (80, 24), "white").save(content, format="PNG")
    with lmdb.open(str(folder), map_size=1 << 20) as env:
        with env.begin(write=True) as txn:
            txn.put(b"num-samples", str(len(labels)).encode())
            for index, label in enumerate(labels, 1):
                suffix = f"{index:09d}".encode()
                txn.put(b"label-" + suffix, label.encode())
                txn.put(b"image-" + suffix, content.getvalue())
                txn.put(b"source-id-" + suffix, f"42:{index}".encode())
    return folder


def test_training_preserves_marks_and_drops_whole_unsupported_labels(lmdb_path):
    table = DecompositionTable.from_characters()
    dataset = VietnameseLmdb(lmdb_path, table.characters[1:], training=True, augment=False)
    assert [row[1] for row in dataset.rows] == ["Việt", "ĐẸP", "ắ", "HàNội"]
    assert dataset.audit["unsupported"] == 1
    assert dataset.audit["too_long"] == 1
    image, label = dataset[0]
    assert image.shape == (3, 32, 128) and label == "Việt"


def test_validation_keeps_oov_long_labels_and_source_ids(lmdb_path):
    table = DecompositionTable.from_characters()
    dataset = VietnameseLmdb(lmdb_path, table.characters[1:])
    assert len(dataset) == 6
    assert dataset[2][1:] == ("€uro", "42:3")
    assert dataset[3][1] == "a" * 30


def test_normalization_never_transliterates_or_moves_a_tone():
    assert normalize_label(" ĐẸP Việt ") == "ĐẸPViệt"
    assert normalize_label("hoa\u0300") == "hoà"
    assert normalize_label("hòa") == "hòa"
    assert is_heavy("ĐẸP") and is_heavy("ắệ")
    assert not is_heavy("abc")
