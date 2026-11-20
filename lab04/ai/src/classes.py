"""Klasy segmentacji BDD (znaki + sygnalizatory)."""

from __future__ import annotations

NUM_CLASSES = 6

CLASS_NAMES = (
    "tlo",
    "znak_drogowy",
    "sygnalizator_obudowa",
    "swiatlo_czerwone",
    "swiatlo_zolte",
    "swiatlo_zielone",
)

# BGR do wizualizacji
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 215, 255),
    2: (80, 80, 80),
    3: (0, 0, 255),
    4: (0, 255, 255),
    5: (0, 255, 0),
}

TL_COLOR_TO_CLS = {
    "red": 3,
    "yellow": 4,
    "green": 5,
}