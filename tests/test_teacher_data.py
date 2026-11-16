import io

import pytest

lmdb = pytest.importorskip("lmdb")
pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

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


def test_e5_subset_keeps_whole_duplicate_groups_and_is_nested(tmp_path, lmdb_path):
    from dfd.teacher_data import select_training_images, subsample_training

    images = {image: image // 2 for image in range(1, 101)}  # 50 duplicate groups of 2 images
    quarter, half = select_training_images(images, 0.25), select_training_images(images, 0.5)
    assert len(quarter) >= 25 and len(half) >= 50 and quarter <= half
    assert all((image // 2 in {i // 2 for i in quarter}) == (image in quarter) for image in images)
    assert select_training_images(images, 0.25) == quarter  # deterministic
    with pytest.raises(ValueError):
        select_training_images(images, 0)

    manifests = tmp_path / "splits/manifests"
    manifests.mkdir(parents=True)
    rows = ["image_id,source_split,duplicate_group_id,final_split", "42,train,1,train", "43,train,2,train", "44,train,3,val"]
    (manifests / "image_split_assignments.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = VietnameseLmdb(lmdb_path, DecompositionTable.from_characters().characters[1:], training=True)
    record = subsample_training(dataset, tmp_path, 1.0)
    assert record["images_total"] == 2 and record["crops"] == record["crops_before"] == len(dataset)
