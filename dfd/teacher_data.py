"""LMDB loading that retains Vietnamese marks and the complete validation denominator."""

import io
import os
import unicodedata
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from dfd.vietchar import decompose


def normalize_label(label):
    # Match PARSeq's whitespace convention without its ASCII transliteration.
    return "".join(unicodedata.normalize("NFC", label).split())


def is_heavy(label):
    marked = 0
    for character in label:
        try:
            parts = decompose(character)
        except ValueError:
            continue
        marked += parts.modifier != "none" or parts.tone != "level"
    return marked >= 2


def image_transform(training=False):
    operations = []
    if training:
        from timm.data import auto_augment
        # Preserve geometry: no flips, rotations, shear, translations or crops.
        names = [name for name in auto_augment._RAND_INCREASING_TRANSFORMS
                 if not name.startswith(("Rotate", "Shear", "Translate"))]
        ops = auto_augment.rand_augment_ops(5, transforms=names)
        operations.append(auto_augment.RandAugment(ops, 3, [1 / len(ops)] * len(ops)))
    operations.extend([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])
    return T.Compose(operations)


class VietnameseLmdb(Dataset):
    def __init__(self, path, charset, training=False, max_length=25, augment=True):
        self.path = str(Path(path).resolve())
        self.training = training
        self.transform = image_transform(training and augment)
        self.env = None
        self.pid = None
        self.rows = []
        self.audit = {"total": 0, "kept": 0, "unsupported": 0, "too_long": 0, "empty": 0}
        supported = set(charset)
        with lmdb.open(self.path, readonly=True, lock=False, readahead=False) as env:
            with env.begin() as txn:
                count = txn.get(b"num-samples")
                if count is None:
                    raise ValueError(f"Not a PARSeq LMDB: {path}")
                self.audit["total"] = int(count)
                for index in range(1, int(count) + 1):
                    suffix = f"{index:09d}".encode("ascii")
                    raw = txn.get(b"label-" + suffix)
                    if raw is None:
                        raise ValueError(f"Missing LMDB label {path}:{index}")
                    label = normalize_label(raw.decode("utf-8"))
                    unsupported = not set(label).issubset(supported)
                    too_long = len(label) > max_length
                    self.audit["unsupported"] += unsupported
                    self.audit["too_long"] += too_long
                    self.audit["empty"] += not label
                    if training and (unsupported or too_long or not label):
                        continue
                    source_id = txn.get(b"source-id-" + suffix)
                    source_id = source_id.decode("utf-8") if source_id else f"lmdb:{index}"
                    self.rows.append((index, label, source_id))
        self.audit["kept"] = len(self.rows)
        if not self.rows:
            raise ValueError(f"Empty dataset after filtering: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.env is None or self.pid != os.getpid():
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
            self.pid = os.getpid()
        row, label, source_id = self.rows[index]
        with self.env.begin() as txn:
            content = txn.get(f"image-{row:09d}".encode("ascii"))
        if content is None:
            raise ValueError(f"Missing LMDB image: {self.path}:{row}")
        with Image.open(io.BytesIO(content)) as image:
            tensor = self.transform(image.convert("RGB"))
        return (tensor, label) if self.training else (tensor, label, source_id)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state
