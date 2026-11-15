"""Deterministic NFC character decomposition and the fixed 229-class vocabulary.

The vocabulary is 94 printable ASCII characters + 134 Vietnamese letters + EOS.
"""

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


EOS = "[EOS]"
COMPONENTS = ("base", "modifier", "tone")
MODIFIERS = ("none", "circumflex", "breve", "horn", "bar")
TONES = ("level", "acute", "grave", "hook", "tilde", "dot")
MODIFIER_MARKS = {"\u0302": "circumflex", "\u0306": "breve", "\u031b": "horn"}
TONE_MARKS = {"\u0301": "acute", "\u0300": "grave", "\u0309": "hook", "\u0303": "tilde", "\u0323": "dot"}
# Same order as PARSeq's configs/charset/94_full.yaml.
ASCII_CHARSET = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)


@dataclass(frozen=True)
class Components:
    base: str
    modifier: str = "none"
    tone: str = "level"


def decompose(character: str) -> Components:
    character = unicodedata.normalize("NFC", character)
    if character == "[EOS]":
        return Components("[EOS]")
    if len(character) != 1:
        raise ValueError("Expected one NFC character or [EOS]")
    if character in "đĐ":
        return Components("d" if character == "đ" else "D", "bar")
    parts = unicodedata.normalize("NFD", character)
    base, modifier, tone = parts[0], "none", "level"
    if not base.isascii():
        raise ValueError(f"Unsupported base character: {character!r}")
    for mark in parts[1:]:
        if mark in MODIFIER_MARKS and modifier == "none":
            modifier = MODIFIER_MARKS[mark]
        elif mark in TONE_MARKS and tone == "level":
            tone = TONE_MARKS[mark]
        else:
            raise ValueError(f"Unsupported or duplicate combining mark: {character!r}")
    if (modifier != "none" or tone != "level") and base.lower() not in "aeiouy":
        raise ValueError(f"Marks on unsupported base: {character!r}")
    allowed_modifiers = {"a": {"none", "circumflex", "breve"}, "e": {"none", "circumflex"}, "o": {"none", "circumflex", "horn"}, "u": {"none", "horn"}}
    if modifier not in allowed_modifiers.get(base.lower(), {"none"}):
        raise ValueError(f"Invalid vowel modifier: {character!r}")
    return Components(base, modifier, tone)


def compose(parts: Components) -> str:
    if parts.base == "[EOS]":
        if parts.modifier != "none" or parts.tone != "level":
            raise ValueError("EOS cannot carry marks")
        return parts.base
    if parts.modifier not in MODIFIERS or parts.tone not in TONES:
        raise ValueError("Unknown modifier or tone")
    if parts.modifier == "bar":
        if parts.base not in {"d", "D"} or parts.tone != "level":
            raise ValueError("Bar is valid only for d/D without a tone")
        return "đ" if parts.base == "d" else "Đ"
    modifier = next((mark for mark, name in MODIFIER_MARKS.items() if name == parts.modifier), "")
    tone = next((mark for mark, name in TONE_MARKS.items() if name == parts.tone), "")
    character = unicodedata.normalize("NFC", parts.base + modifier + tone)
    if decompose(character) != parts:
        raise ValueError("Components do not form a supported character")
    return character


def build_table(characters: list[str]) -> dict[str, Components]:
    if len(set(characters)) != len(characters):
        raise ValueError("Vocabulary contains duplicate characters")
    table = {character: decompose(character) for character in characters}
    if len(set(table.values())) != len(table):
        raise ValueError("Different vocabulary entries map to the same triple; normalize NFC first")
    if any(compose(parts) != character for character, parts in table.items()):
        raise ValueError("Vocabulary must be NFC-normalized before building the table")
    return table


def default_characters() -> tuple[str, ...]:
    """Return the 228 non-EOS characters in their fixed class order."""
    marked = (unicodedata.normalize("NFC", vowel + mark) for vowel in "aăâeêioôơuưy" for mark in ("", *TONE_MARKS))
    lower = [character for character in marked if not character.isascii()] + ["đ"]
    return (*ASCII_CHARSET, *lower, *(character.upper() for character in lower))


@dataclass(frozen=True)
class DecompositionTable:
    """Class-indexed character decomposition, in PARSeq output order (EOS is class 0)."""

    characters: tuple[str, ...]
    parts: tuple[Components, ...]
    bases: tuple[str, ...]

    @classmethod
    def from_characters(cls, characters: Iterable[str] | None = None) -> "DecompositionTable":
        characters = default_characters() if characters is None else tuple(characters)
        if EOS in characters:
            raise ValueError("EOS is added by the table; do not list it in the vocabulary")
        table = build_table([EOS, *characters])
        bases = tuple(dict.fromkeys(parts.base for parts in table.values()))
        return cls(tuple(table), tuple(table.values()), bases)

    def __len__(self) -> int:
        return len(self.characters)

    def classes(self, component: str) -> tuple[str, ...]:
        return {"base": self.bases, "modifier": MODIFIERS, "tone": TONES}[component]

    def component_ids(self, component: str) -> list[int]:
        """Map each character class to its class index in `component`."""
        lookup = {name: index for index, name in enumerate(self.classes(component))}
        return [lookup[getattr(parts, component)] for parts in self.parts]

    def matrix(self, component: str) -> list[list[int]]:
        """Return binary M_x[c, k] = 1 iff x(c) = k, shape (len(self), number of component classes)."""
        size = len(self.classes(component))
        return [[int(index == target) for index in range(size)] for target in self.component_ids(component)]

    def coverage(self, observed: Iterable[str]) -> tuple[list[str], list[str]]:
        """Compare dataset characters with the vocabulary: (out_of_vocabulary, unused)."""
        seen = {unicodedata.normalize("NFC", character) for character in observed}
        known = set(self.characters)
        return sorted(seen - known), [character for character in self.characters[1:] if character not in seen]

    def fingerprint(self) -> str:
        """SHA-256 of the class order, to record with checkpoints and run metadata."""
        return hashlib.sha256("\n".join(self.characters).encode("utf-8")).hexdigest()
