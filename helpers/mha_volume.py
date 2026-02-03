from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Summary:
# - Lightweight container for a MetaImage (.mha) volume and its parsed metadata.
# - What it does: stores raw header pairs for traceability, typed header fields, and a numpy view
#   of the payload.
@dataclass
class MhaVolume:
    path: str
    meta: dict[str, str]
    ObjectType: str
    NDims: int
    BinaryData: bool
    BinaryDataByteOrderMSB: bool
    CompressedData: bool
    TransformMatrix: np.ndarray
    DimSize: np.ndarray
    Offset: Optional[np.ndarray]
    CenterOfRotation: Optional[np.ndarray]
    AnatomicalOrientation: str
    ElementSpacing: Optional[np.ndarray]
    ElementType: str
    UltrasoundImageOrientation: str
    UltrasoundImageType: str
    ElementDataFile: str
    raw_header_lines: list[str]
    data_offset: int
    dtype: np.dtype
    data: np.ndarray

    # Summary:
    # - Return the spatial shape in header order (x, y, z, ...).
    # - Input: `self`.
    # - Returns: tuple[int, ...].
    @property
    def shape_xyz(self) -> tuple[int, ...]:
        # Keep header order so callers can match metadata fields directly.
        return tuple(int(value) for value in self.DimSize)

    # Summary:
    # - Return the spatial shape reversed (z, y, x, ...).
    # - Input: `self`.
    # - Returns: tuple[int, ...].
    @property
    def shape_zyx(self) -> tuple[int, ...]:
        # Reverse header order so the last axis is X (fastest in C-order).
        return tuple(reversed(self.shape_xyz))
