from __future__ import annotations

import math
import os
import tempfile
from typing import Optional

import numpy as np

# Allow package import first; fall back so this file can be run directly for the sanity test.
try:
    from .mha_volume import MhaVolume
except ImportError:
    # Add the repo root to sys.path when running this module as a script.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(repo_root))
    from helpers.mha_volume import MhaVolume


# Summary:
# - Reader for MetaImage (.mha) files that parses headers and returns a volume wrapper.
# - What it does: loads metadata, validates binary payload assumptions, and builds a numpy view.
class MhaReader:
    # Summary:
    # - Read a .mha file and return a parsed MhaVolume container.
    # - Input: `self`, `path` (str), `use_memmap` (bool).
    # - Returns: MhaVolume.
    def read(self, path: str, *, use_memmap: bool = True) -> MhaVolume:
        # Normalize the path for consistent stat calls and memmap usage.
        path_str = os.fspath(path)

        # Read the header first so we know where the binary payload starts.
        with open(path_str, "rb") as fp:
            meta, raw_lines, data_offset = self._read_header(fp)

            # Parse required string values so the typed view matches header naming.
            object_type = self._require_meta_value(meta, "ObjectType")
            anatomical_orientation = self._require_meta_value(meta, "AnatomicalOrientation")
            element_type = self._require_meta_value(meta, "ElementType")
            ultrasound_image_orientation = self._require_meta_value(
                meta, "UltrasoundImageOrientation"
            )
            ultrasound_image_type = self._require_meta_value(meta, "UltrasoundImageType")
            element_data_file = self._require_meta_value(meta, "ElementDataFile")

            # Parse and validate NDims so we know how many dimensions to expect.
            try:
                ndims = self.parse_int(self._require_meta_value(meta, "NDims"))
            except ValueError as exc:
                raise ValueError("Invalid .mha: NDims is not a valid integer") from exc
            if ndims <= 0:
                raise ValueError(f"Invalid .mha: NDims must be > 0, got {ndims}")

            # Parse DimSize into integer dimensions in header order (x, y, z, ...).
            try:
                dim_size = self.parse_int_array(self._require_meta_value(meta, "DimSize"))
            except ValueError as exc:
                raise ValueError("Invalid .mha: DimSize is not a valid int array") from exc
            if len(dim_size) != ndims:
                raise ValueError(
                    f"Invalid .mha: DimSize length {len(dim_size)} "
                    f"does not match NDims {ndims}"
                )
            # Convert to a tuple so math.prod and shape building stay simple.
            dim_size_xyz = tuple(int(value) for value in dim_size)

            # Parse optional ElementSpacing when provided and validate length.
            element_spacing: Optional[np.ndarray] = None
            raw_element_spacing = meta.get("ElementSpacing")
            if raw_element_spacing is not None and raw_element_spacing.strip() != "":
                try:
                    element_spacing = self.parse_float_array(raw_element_spacing)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid .mha: ElementSpacing is not a valid float array"
                    ) from exc
                if len(element_spacing) != ndims:
                    raise ValueError(
                        f"Invalid .mha: ElementSpacing length {len(element_spacing)} "
                        f"does not match NDims {ndims}"
                    )

            # Parse optional Offset when provided and validate length.
            offset: Optional[np.ndarray] = None
            raw_offset = meta.get("Offset")
            if raw_offset is not None and raw_offset.strip() != "":
                try:
                    offset = self.parse_float_array(raw_offset)
                except ValueError as exc:
                    raise ValueError("Invalid .mha: Offset is not a valid float array") from exc
                if len(offset) != ndims:
                    raise ValueError(
                        f"Invalid .mha: Offset length {len(offset)} does not match NDims {ndims}"
                    )

            # Parse optional CenterOfRotation when provided and validate length.
            center_of_rotation: Optional[np.ndarray] = None
            raw_center_of_rotation = meta.get("CenterOfRotation")
            if raw_center_of_rotation is not None and raw_center_of_rotation.strip() != "":
                try:
                    center_of_rotation = self.parse_int_array(raw_center_of_rotation)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid .mha: CenterOfRotation is not a valid int array"
                    ) from exc
                if len(center_of_rotation) != ndims:
                    raise ValueError(
                        "Invalid .mha: CenterOfRotation length "
                        f"{len(center_of_rotation)} does not match NDims {ndims}"
                    )

            # Parse TransformMatrix and ensure we have exactly 9 values.
            try:
                transform_values = self.parse_float_array(
                    self._require_meta_value(meta, "TransformMatrix")
                )
            except ValueError as exc:
                raise ValueError("Invalid .mha: TransformMatrix is not a valid float array") from exc
            try:
                transform_matrix = self.parse_matrix_3x3(transform_values)
            except ValueError as exc:
                raise ValueError("Invalid .mha: TransformMatrix must have 9 values") from exc

            # Parse required boolean fields so unsupported formats fail early.
            try:
                binary_data = self.parse_bool(self._require_meta_value(meta, "BinaryData"))
            except ValueError as exc:
                raise ValueError("Invalid .mha: BinaryData is not a valid boolean") from exc
            if not binary_data:
                raise ValueError(
                    f"Unsupported .mha: BinaryData must be True, got {meta['BinaryData']}"
                )

            try:
                compressed_data = self.parse_bool(
                    self._require_meta_value(meta, "CompressedData")
                )
            except ValueError as exc:
                raise ValueError("Invalid .mha: CompressedData is not a valid boolean") from exc
            if compressed_data:
                raise ValueError("Unsupported .mha: CompressedData=True is not supported yet")

            # Determine byte order from metadata so dtype is interpreted correctly.
            try:
                binary_data_byte_order_msb = self.parse_bool(
                    self._require_meta_value(meta, "BinaryDataByteOrderMSB")
                )
            except ValueError as exc:
                raise ValueError(
                    "Invalid .mha: BinaryDataByteOrderMSB is not a valid boolean"
                ) from exc

            # Parse ElementType so we can map to a numpy dtype.
            dtype = self._dtype_from_element_type(element_type, binary_data_byte_order_msb)

            # Parse number of channels, defaulting to 1 when metadata is absent.
            raw_channels = meta.get("ElementNumberOfChannels")
            if raw_channels is None or raw_channels.strip() == "":
                # Default to single-channel volumes when the header omits this field.
                channels = 1
            else:
                try:
                    channels = self.parse_int(raw_channels)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid .mha: ElementNumberOfChannels is not a valid integer"
                    ) from exc
                if channels <= 0:
                    raise ValueError(
                        "Invalid .mha: ElementNumberOfChannels must be > 0, "
                        f"got {channels}"
                    )

            # Compute the number of voxels before allocating any buffers.
            n_vox = math.prod(dim_size_xyz) * channels
            expected_bytes = self._compute_expected_bytes(dim_size_xyz, channels, dtype)

            # Validate the payload fits in the file so we fail fast on corrupt data.
            file_size = os.stat(path_str).st_size
            if data_offset + expected_bytes > file_size:
                raise ValueError(
                    "Invalid .mha: payload exceeds file size "
                    f"(data_offset={data_offset}, expected_bytes={expected_bytes}, "
                    f"file_size={file_size})"
                )

            # Build a shape where X is the last axis so it is fastest in C-order.
            shape_spatial = tuple(reversed(dim_size_xyz))
            if channels == 1:
                shape = shape_spatial
            else:
                shape = shape_spatial + (channels,)

            # Load data either as a memory map or by reading bytes into memory.
            if use_memmap:
                # Memmap keeps the payload on disk and provides a numpy view.
                mm = np.memmap(
                    path_str,
                    dtype=dtype,
                    mode="r",
                    offset=data_offset,
                    shape=(n_vox,),
                )
                data = mm.reshape(shape)
            else:
                # Seek to the payload and read only the required number of bytes.
                fp.seek(data_offset)
                payload = fp.read(expected_bytes)
                if len(payload) != expected_bytes:
                    raise ValueError(
                        "Invalid .mha: could not read full payload "
                        f"(expected_bytes={expected_bytes}, got={len(payload)})"
                    )
                data = np.frombuffer(payload, dtype=dtype, count=n_vox).reshape(shape)

        # Return a lightweight container with all parsed metadata and data view.
        return MhaVolume(
            path=path_str,
            meta=meta,
            ObjectType=object_type,
            NDims=ndims,
            BinaryData=binary_data,
            BinaryDataByteOrderMSB=binary_data_byte_order_msb,
            CompressedData=compressed_data,
            TransformMatrix=transform_matrix,
            DimSize=dim_size,
            Offset=offset,
            CenterOfRotation=center_of_rotation,
            AnatomicalOrientation=anatomical_orientation,
            ElementSpacing=element_spacing,
            ElementType=element_type,
            UltrasoundImageOrientation=ultrasound_image_orientation,
            UltrasoundImageType=ultrasound_image_type,
            ElementDataFile=element_data_file,
            raw_header_lines=raw_lines,
            data_offset=data_offset,
            dtype=dtype,
            data=data,
        )

    # Summary:
    # - Read and parse the text header of a .mha file.
    # - Input: `self`, `fp` (binary file object).
    # - Returns: tuple[dict[str, str], list[str], int].
    def _read_header(self, fp) -> tuple[dict[str, str], list[str], int]:
        # Collect raw header lines for debug/round-trip use.
        raw_lines: list[str] = []
        # Store key/value pairs exactly as parsed (last write wins).
        meta: dict[str, str] = {}

        # Read line-by-line so we do not pull the full file into memory.
        while True:
            line_bytes = fp.readline()
            if line_bytes == b"":
                # EOF before ElementDataFile means the file is invalid.
                break

            # Decode conservatively so binary payload bytes do not crash decoding.
            line_text = line_bytes.decode("utf-8", errors="ignore").strip("\r\n")
            raw_lines.append(line_text)

            # Parse key/value pairs from lines that contain '='.
            parsed = self._parse_kv_line(line_text)
            if parsed is None:
                continue

            key, value = parsed
            meta[key] = value

            # Stop at ElementDataFile so data_offset points at the payload start.
            if key == "ElementDataFile":
                if value != "LOCAL":
                    raise ValueError(
                        f"Unsupported .mha: ElementDataFile must be LOCAL, got {value}"
                    )
                data_offset = fp.tell()
                return meta, raw_lines, data_offset

        # We reached EOF without finding ElementDataFile -> invalid .mha.
        raise ValueError("Invalid .mha: missing ElementDataFile")

    # Summary:
    # - Parse a single header line into (key, value) if it contains '='.
    # - Input: `self`, `line` (str).
    # - Returns: tuple[str, str] | None.
    def _parse_kv_line(self, line: str) -> Optional[tuple[str, str]]:
        # Skip empty lines and non key/value lines.
        if not line or "=" not in line:
            return None

        # Split on the first '=' so values can include '=' characters.
        left, right = line.split("=", 1)
        key = left.strip()
        if not key:
            return None
        value = right.strip()
        return key, value

    # Summary:
    # - Fetch a required metadata value or raise a clear missing-key error.
    # - Input: `self`, `meta` (dict[str, str]), `key` (str).
    # - Returns: str.
    def _require_meta_value(self, meta: dict[str, str], key: str) -> str:
        # Treat empty strings as missing so later parsing errors are more explicit.
        raw_value = meta.get(key)
        if raw_value is None or raw_value.strip() == "":
            raise ValueError(f"Invalid .mha: missing {key}")
        return raw_value

    # Summary:
    # - Parse a MetaImage boolean string into a Python bool.
    # - Input: `self`, `value` (str).
    # - Returns: bool.
    def parse_bool(self, value: str) -> bool:
        # Normalize input so True/False/1/0 values are accepted consistently.
        normalized = value.strip().lower()
        if normalized in ("true", "1"):
            return True
        if normalized in ("false", "0"):
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    # Summary:
    # - Parse a MetaImage integer string into an int.
    # - Input: `self`, `value` (str).
    # - Returns: int.
    def parse_int(self, value: str) -> int:
        # Strip whitespace so values like " 3 " still parse correctly.
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid integer value: {value}") from exc

    # Summary:
    # - Parse a MetaImage float string into a float.
    # - Input: `self`, `value` (str).
    # - Returns: float.
    def parse_float(self, value: str) -> float:
        # Strip whitespace so values like " 0.5 " still parse correctly.
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid float value: {value}") from exc

    # Summary:
    # - Parse a space-delimited integer array string into a numpy array.
    # - Input: `self`, `value` (str).
    # - Returns: np.ndarray.
    def parse_int_array(self, value: str) -> np.ndarray:
        # Normalize separators so commas and whitespace both work.
        tokens = value.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty integer array")

        values: list[int] = []
        # Convert each token to int so callers get a typed array.
        for token in tokens:
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid integer value: {token}") from exc

        return np.array(values, dtype=int)

    # Summary:
    # - Parse a space-delimited float array string into a numpy array.
    # - Input: `self`, `value` (str).
    # - Returns: np.ndarray.
    def parse_float_array(self, value: str) -> np.ndarray:
        # Normalize separators so commas and whitespace both work.
        tokens = value.replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty float array")

        values: list[float] = []
        # Convert each token to float so callers get a typed array.
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(f"Invalid float value: {token}") from exc

        return np.array(values, dtype=float)

    # Summary:
    # - Parse a flat list of 9 values into a 3x3 transform matrix.
    # - Input: `self`, `values` (sequence of numeric values).
    # - Returns: np.ndarray with shape (3, 3).
    def parse_matrix_3x3(self, values) -> np.ndarray:
        # Validate length before reshaping to avoid silent dimension errors.
        if len(values) != 9:
            raise ValueError(f"Expected 9 values for 3x3 matrix, got {len(values)}")
        return np.array(values, dtype=float).reshape((3, 3))

    # Summary:
    # - Map MetaImage ElementType + byte order into a numpy dtype.
    # - Input: `self`, `element_type` (str), `is_msb` (bool).
    # - Returns: np.dtype.
    def _dtype_from_element_type(self, element_type: str, is_msb: bool) -> np.dtype:
        # Supported scalar types mapped to numpy base dtypes.
        element_map = {
            "MET_UCHAR": np.uint8,
            "MET_CHAR": np.int8,
            "MET_USHORT": np.uint16,
            "MET_SHORT": np.int16,
            "MET_UINT": np.uint32,
            "MET_INT": np.int32,
            "MET_ULONG_LONG": np.uint64,
            "MET_LONG_LONG": np.int64,
            "MET_FLOAT": np.float32,
            "MET_DOUBLE": np.float64,
        }

        # Validate the element type before creating a dtype.
        if element_type not in element_map:
            raise ValueError(f"Unsupported .mha: ElementType {element_type} is unknown")

        dtype = np.dtype(element_map[element_type])

        # Apply byte order only for multi-byte types; uint8/int8 are unchanged.
        if dtype.itemsize > 1:
            byteorder = ">" if is_msb else "<"
            dtype = dtype.newbyteorder(byteorder)

        return dtype

    # Summary:
    # - Compute the expected payload size in bytes from dimensions and dtype.
    # - Input: `self`, `dim_xyz` (tuple[int, ...]), `channels` (int), `dtype` (np.dtype).
    # - Returns: int.
    def _compute_expected_bytes(
        self,
        dim_xyz: tuple[int, ...],
        channels: int,
        dtype: np.dtype,
    ) -> int:
        # Multiply all dimensions and channels to get the number of scalar values.
        n_vox = math.prod(dim_xyz) * channels
        return n_vox * dtype.itemsize


if __name__ == "__main__":
    # Create a tiny .mha file so developers can sanity-check parsing quickly.
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = os.path.join(tmp_dir, "tiny.mha")

        # Build a minimal header for a 2x2x2 unsigned char volume.
        header_lines = [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            "TransformMatrix = 1 0 0 0 1 0 0 0 1",
            "DimSize = 2 2 2",
            "Offset = 0 0 0",
            "CenterOfRotation = 0 0 0",
            "AnatomicalOrientation = RAI",
            "ElementSpacing = 1 1 1",
            "ElementType = MET_UCHAR",
            "UltrasoundImageOrientation = MFA",
            "UltrasoundImageType = BRIGHTNESS",
            "ElementDataFile = LOCAL",
        ]
        header_text = "\n".join(header_lines) + "\n"

        # Create a payload with values 0..7 so reshape order is easy to verify.
        payload = bytes(range(8))

        # Write header then binary payload so ElementDataFile offset is correct.
        with open(test_path, "wb") as fp:
            fp.write(header_text.encode("ascii"))
            fp.write(payload)

        # Read back with the reader and verify basic metadata and layout.
        volume = MhaReader().read(test_path, use_memmap=False)

        # Validate the core typed fields.
        assert volume.NDims == 3
        assert np.array_equal(volume.DimSize, np.array([2, 2, 2], dtype=int))
        assert volume.ElementType == "MET_UCHAR"
        assert volume.BinaryData is True
        assert volume.CompressedData is False
        assert volume.TransformMatrix.shape == (3, 3)
        assert volume.dtype == np.dtype(np.uint8)

        # With shape (z, y, x), the first axis is z and X is fastest.
        assert volume.data.shape == (2, 2, 2)
        assert volume.data[0, 0, 0] == 0
        assert volume.data[0, 0, 1] == 1
        assert volume.data[0, 1, 0] == 2
        assert volume.data[1, 0, 0] == 4
