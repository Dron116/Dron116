from __future__ import annotations

import re

# Буквы, разрешённые на российских номерных знаках (кириллица и латинские двойники).
CYR_LETTERS = "АВЕКМНОРСТУХ"
LAT_LETTERS = "ABEKMHOPCTYX"
LETTER_SET = set(CYR_LETTERS + LAT_LETTERS)

LAT_TO_CYR = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "E": "Е",
        "K": "К",
        "M": "М",
        "H": "Н",
        "O": "О",
        "P": "Р",
        "C": "С",
        "T": "Т",
        "Y": "У",
        "X": "Х",
    }
)

DIGIT_LOOKALIKES = str.maketrans(
    {
        "О": "0",
        "O": "0",
        "З": "3",
        "З".lower(): "3",
        "Ч": "4",
        "Б": "6",
        "S": "5",
        "I": "1",
        "l": "1",
        "Z": "2",
        "G": "6",
        "D": "0",
        "Q": "0",
    }
)

LETTER_LOOKALIKES = str.maketrans(
    {
        "0": "О",
        "3": "З",
        "4": "А",
        "6": "Б",
        "8": "В",
        "5": "S",
    }
)

PLATE_RE = re.compile(
    rf"^[{CYR_LETTERS}{LAT_LETTERS}]\d{{3}}[{CYR_LETTERS}{LAT_LETTERS}]{{2}}\d{{2,3}}$"
)
TOKEN_RE = re.compile(r"[A-ZА-Я0-9]+", re.IGNORECASE)


def to_cyrillic(text: str) -> str:
    return text.upper().translate(LAT_TO_CYR)


def strip_noise(text: str) -> str:
    return "".join(TOKEN_RE.findall(text.upper().replace("Ё", "Е")))


def _coerce_standard(chars: str) -> str | None:
    """Приводит 8–9 символов к формату «буква + 3 цифры + 2 буквы + регион»."""
    if len(chars) not in (8, 9):
        return None
    out = []
    for i, raw in enumerate(chars):
        ch = raw.upper()
        if i == 0 or i in (4, 5):
            ch = ch.translate(LETTER_LOOKALIKES)
            ch = to_cyrillic(ch)
            if ch not in CYR_LETTERS:
                return None
        else:
            ch = ch.translate(DIGIT_LOOKALIKES)
            if not ch.isdigit():
                return None
        out.append(ch)
    plate = "".join(out)
    return plate if PLATE_RE.match(plate) else None


def candidates_from_ocr(text: str) -> list[str]:
    compact = strip_noise(text)
    compact = to_cyrillic(compact)
    found: list[str] = []
    seen: set[str] = set()
    for length in (9, 8):
        if len(compact) < length:
            continue
        for i in range(0, len(compact) - length + 1):
            coerced = _coerce_standard(compact[i : i + length])
            if coerced and coerced not in seen:
                seen.add(coerced)
                found.append(coerced)
    nines = [p for p in found if len(p) == 9]
    if nines:
        found = [p for p in found if len(p) == 9 or not any(p == n[:8] for n in nines)]
    return found


def normalize_plate(text: str) -> str | None:
    plates = candidates_from_ocr(text)
    return plates[0] if plates else None


def format_plate(plate: str) -> str:
    plate = to_cyrillic(strip_noise(plate))
    if len(plate) == 9:
        return f"{plate[0]} {plate[1:4]} {plate[4:6]} {plate[6:9]}"
    if len(plate) == 8:
        return f"{plate[0]} {plate[1:4]} {plate[4:6]} {plate[6:8]}"
    return plate


def is_valid_plate(text: str) -> bool:
    return normalize_plate(text) is not None
