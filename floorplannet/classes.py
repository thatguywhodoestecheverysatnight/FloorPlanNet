"""Label taxonomy for FloorPlanNet.

Two taxonomies are supported:

* ``coarse`` (default, 5 classes): background, wall, room, door, window.
  This is the "rooms / walls / openings" task that the vectorizer consumes.
* ``cubicasa12``: the 12 room classes used by the CubiCasa5K benchmark
  (Kalervo et al., 2019). Openings are rasterised into separate classes on top
  so the vectorizer still works (door=12, window=13 -> 14 classes total).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Taxonomy:
    name: str
    classes: Tuple[str, ...]
    colors: Tuple[Tuple[int, int, int], ...]
    wall: int
    door: int
    window: int
    room_ids: Tuple[int, ...]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def palette(self) -> List[int]:
        flat: List[int] = []
        for c in self.colors:
            flat.extend(c)
        return flat


COARSE = Taxonomy(
    name="coarse",
    classes=("background", "wall", "room", "door", "window"),
    colors=((255, 255, 255), (38, 38, 38), (129, 199, 235), (236, 112, 99), (88, 214, 141)),
    wall=1,
    door=3,
    window=4,
    room_ids=(2,),
)

_CC_ROOMS = (
    "background", "outdoor", "wall", "kitchen", "living_room", "bedroom",
    "bath", "entry", "railing", "storage", "garage", "undefined",
)
CUBICASA12 = Taxonomy(
    name="cubicasa12",
    classes=_CC_ROOMS + ("door", "window"),
    colors=(
        (255, 255, 255), (170, 214, 136), (38, 38, 38), (255, 190, 92),
        (129, 199, 235), (186, 148, 222), (99, 179, 196), (240, 230, 140),
        (140, 140, 140), (205, 170, 125), (160, 160, 200), (210, 225, 240),
        (236, 112, 99), (88, 214, 141),
    ),
    wall=2,
    door=12,
    window=13,
    room_ids=(1, 3, 4, 5, 6, 7, 9, 10, 11),
)

TAXONOMIES: Dict[str, Taxonomy] = {"coarse": COARSE, "cubicasa12": CUBICASA12}

# Mapping of CubiCasa5K SVG "Space <Type>" names onto cubicasa12 indices.
# Based on the room grouping used by the official CubiCasa5K code (floortrans/loaders/house.py);
# verify against your checkout before reporting benchmark numbers.
CUBICASA_ROOM_MAP: Dict[str, int] = {
    "Alcove": 11, "Attic": 11, "Ballroom": 11, "Bar": 11, "Basement": 11,
    "Bath": 6, "Bedroom": 5, "CarPort": 10, "Church": 11, "Closet": 9,
    "ConferenceRoom": 11, "Conservatory": 11, "Counter": 11, "Den": 4,
    "Dining": 4, "DraughtLobby": 7, "DressingRoom": 9, "EatingArea": 4,
    "Elevated": 11, "Elevator": 11, "Entry": 7, "ExerciseRoom": 11,
    "Garage": 10, "Garbage": 11, "Hall": 7, "HallWay": 7, "HotTub": 11,
    "Kitchen": 3, "Library": 11, "LivingRoom": 4, "Loft": 11, "Lounge": 4,
    "MediaRoom": 11, "MeetingRoom": 11, "Museum": 11, "Nook": 11,
    "Office": 11, "OpenToBelow": 11, "Outdoor": 1, "Pantry": 9,
    "Reception": 7, "RecreationRoom": 11, "RetailSpace": 11, "Room": 11,
    "Sanctuary": 11, "Sauna": 6, "ServiceRoom": 11, "ServingArea": 11,
    "Skylights": 11, "Stable": 11, "Stage": 11, "StairWell": 11,
    "Storage": 9, "SunRoom": 11, "SwimmingPool": 11, "TechnicalRoom": 11,
    "Theatre": 11, "Undefined": 11, "UserDefined": 11, "Utility": 11,
    "Railing": 8,
}


def get_taxonomy(name: str) -> Taxonomy:
    try:
        return TAXONOMIES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown taxonomy '{name}', choose from {list(TAXONOMIES)}") from exc


def room_label(room_type: str, taxonomy: Taxonomy) -> int:
    """Return the label index for a CubiCasa room type under ``taxonomy``."""
    if taxonomy.name == "coarse":
        return 0 if room_type == "Railing" else 2
    return CUBICASA_ROOM_MAP.get(room_type, 11)
