"""Deterministic NFC character decomposition, independent of an assumed 229-class list."""

import unicodedata
from dataclasses import dataclass


MODIFIERS = ("none", "circumflex", "breve", "horn", "bar")
TONES = ("level", "acute", "grave", "hook", "tilde", "dot")
MODIFIER_MARKS = {"\u0302": "circumflex", "\u0306": "breve", "\u031b": "horn"}
TONE_MARKS = {"\u0301": "acute", "\u0300": "grave", "\u0309": "hook", "\u0303": "tilde", "\u0323": "dot"}


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
