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

    # Path to the source .mha file on disk.
    # - Input: parsed from the `path` argument passed into `MhaReader.read(...)`.
    # - Returns: stored as a string for logging/debugging and for re-loading if needed.
    path: str

    # Raw header key/value pairs as strings, exactly as parsed (best-effort).
    # - Input: read from the text header section before the binary payload.
    # - Returns: a dict useful for debugging and for reading fields we don't model yet.
    meta: dict[str, str]

    # MetaImage `ObjectType` header field (typically "Image").
    # - Input: from the header (string).
    # - Returns: stored as-is.
    ObjectType: str

    # MetaImage `NDims` header field: number of spatial dimensions in the volume.
    # - Input: from the header (integer).
    # - Returns: stored as an int.
    NDims: int

    # MetaImage `BinaryData` header field: indicates the payload is binary.
    # - Input: from the header (boolean-like string parsed to bool).
    # - Returns: stored as a bool; this reader currently expects it to be True.
    BinaryData: bool

    # MetaImage `BinaryDataByteOrderMSB` header field: indicates big-endian byte order when True.
    # - Input: from the header (boolean-like string parsed to bool).
    # - Returns: stored as a bool and used to build the correct numpy dtype endianness.
    BinaryDataByteOrderMSB: bool

    # MetaImage `CompressedData` header field: indicates whether payload is compressed.
    # - Input: from the header (boolean-like string parsed to bool).
    # - Returns: stored as a bool; this reader currently expects it to be False.
    CompressedData: bool

    # MetaImage `TransformMatrix` header field: 3x3 orientation/transform matrix.
    # - Input: 9 floats from the header.
    # - Returns: a numpy array shaped (3, 3).
    TransformMatrix: np.ndarray

    # MetaImage `DimSize` header field: spatial size in header order (x, y, z, ...).
    # - Input: integer array from the header.
    # - Returns: stored as a numpy array so it is easy to validate and reuse in math.
    DimSize: np.ndarray

    # MetaImage `Offset` header field: origin/offset of the volume in physical units (optional).
    # - Input: float array from the header or None if not present.
    # - Returns: numpy array or None.
    Offset: Optional[np.ndarray]

    # MetaImage `CenterOfRotation` header field: center of rotation (optional).
    # - Input: numeric array from the header or None if not present.
    # - Returns: numpy array or None.
    CenterOfRotation: Optional[np.ndarray]

    # MetaImage `AnatomicalOrientation` header field: orientation code string.
    # - Input: from the header (string).
    # - Returns: stored as-is.
    AnatomicalOrientation: str

    # MetaImage `ElementSpacing` header field: spacing between samples per axis (optional).
    # - Input: float array from the header or None if not present.
    # - Returns: numpy array or None.
    ElementSpacing: Optional[np.ndarray]

    # MetaImage `ElementType` header field: scalar type stored in the payload (e.g., MET_USHORT).
    # - Input: from the header (string).
    # - Returns: stored as-is; `dtype` is the numpy interpretation of this value.
    ElementType: str

    # Custom ultrasound header field: `UltrasoundImageOrientation` (string).
    # - Input: from the header (string).
    # - Returns: stored as-is.
    UltrasoundImageOrientation: str

    # Custom ultrasound header field: `UltrasoundImageType` (string).
    # - Input: from the header (string).
    # - Returns: stored as-is.
    UltrasoundImageType: str

    # MetaImage `ElementDataFile` header field: indicates where the payload is stored.
    # - Input: from the header (this reader expects "LOCAL" for embedded payload).
    # - Returns: stored as-is.
    ElementDataFile: str

    # Raw header lines as read from the file.
    # - Input: captured during parsing (including comments/spacing where possible).
    # - Returns: useful for debugging/round-tripping diagnostics.
    raw_header_lines: list[str]

    # Byte offset (from file start) where the binary payload begins.
    # - Input: computed while parsing the header.
    # - Returns: stored as an int; used for memmap and for reading the payload bytes.
    data_offset: int

    # NumPy dtype used to interpret the payload bytes (including endianness).
    # - Input: derived from `ElementType` and `BinaryDataByteOrderMSB`.
    # - Returns: stored as a numpy dtype.
    dtype: np.dtype

    # The volume payload as a NumPy array view.
    # - Input: the file payload is a 1D stream where X changes fastest, then Y, then Z.
    # - Returns:
    #   - A reshaped NumPy array in (z, y, x) order for single-channel volumes.
    #   - For multi-channel payloads, shape is (z, y, x, channels).
    #
    # Why (z, y, x) here (instead of (x, y, z))?
    # - NumPy images are typically accessed as (y, x) == (row, col).
    # - Extending to 3D makes (z, y, x) a natural convention where `data[z]` is a 2D slice.
    # - It also keeps X as the last axis, which matches NumPy C-order contiguity and the
    #   common "x-fastest" layout of the on-disk payload.
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
