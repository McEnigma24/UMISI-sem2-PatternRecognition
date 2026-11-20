from __future__ import annotations

from models.base import SegArchitecture
from models.mask2former import Mask2FormerAdapter
from models.segformer import SegFormerAdapter
from models.unetpp import UnetPPAdapter
from models.yolo11_seg import Yolo11SegAdapter

_REGISTRY: dict[str, SegArchitecture] = {
    SegFormerAdapter().name: SegFormerAdapter(),
    Mask2FormerAdapter().name: Mask2FormerAdapter(),
    UnetPPAdapter().name: UnetPPAdapter(),
    Yolo11SegAdapter().name: Yolo11SegAdapter(),
}


def get_architecture(name: str) -> SegArchitecture:
    if name not in _REGISTRY:
        raise KeyError(f"Nieznana architektura: {name}. Dostępne: {list(_REGISTRY)}")
    return _REGISTRY[name]