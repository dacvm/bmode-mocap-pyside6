from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Allow package import first; fall back so this file can be run directly for tests.
try:
    from .mha_volume import MhaVolume
except ImportError:
    # Add the repo root to sys.path when running this module as a script.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(repo_root))
    from helpers.mha_volume import MhaVolume


DEFAULT_THRESHOLD = 200
DEFAULT_MAX_POINTS = 200_000
DEFAULT_CHUNK_DEPTH = 4


# Summary:
# - Lightweight container for volume geometry samples.
# - What it does: stores world-space points, their intensities, and axis-aligned bounds.
@dataclass
class GeometryBuildingResult:
    # World coordinates for each sampled voxel center, shaped (N, 3).
    # - Input: produced by MhaVolumeGeometryBuilder after applying the cached affine.
    # - Returns: numpy array of floats for plotting or downstream computation.
    points_xyz: np.ndarray

    # Intensity values for each sampled voxel, shaped (N,).
    # - Input: pulled directly from the volume data and optionally cast to float.
    # - Returns: numpy array aligned with `points_xyz`.
    intensities: np.ndarray

    # Minimum world coordinate across all points (x, y, z), or None when empty.
    # - Input: computed from `points_xyz` when N > 0.
    # - Returns: numpy array or None.
    min_xyz: Optional[np.ndarray]

    # Maximum world coordinate across all points (x, y, z), or None when empty.
    # - Input: computed from `points_xyz` when N > 0.
    # - Returns: numpy array or None.
    max_xyz: Optional[np.ndarray]


# Summary:
# - Compute-only helper that builds geometry samples from a MhaVolume.
# - What it does: samples above-threshold voxels, applies the cached affine, and returns a result.
class MhaVolumeGeometryBuilder:
    # Summary:
    # - Initialize the geometry builder with an optional volume.
    # - Input: `self`, `mha_volume` (MhaVolume | None).
    # - Returns: None.
    def __init__(self, mha_volume: Optional[MhaVolume] = None) -> None:
        # Store the active volume so build_geometry() can reuse it.
        self._mha_volume: Optional[MhaVolume] = None
        # Cache the affine matrix so slider changes do not rebuild it.
        self._affine: Optional[np.ndarray] = None
        # Keep a dedicated RNG for sampling so results are stable per instance.
        self._rng = np.random.default_rng()

        # Initialize the cached volume/affine if one was provided.
        if mha_volume is not None:
            self.set_volume(mha_volume)

    # Summary:
    # - Assign a new volume and rebuild the cached affine transform.
    # - Input: `self`, `mha_volume` (MhaVolume | None).
    # - Returns: None.
    def set_volume(self, mha_volume: Optional[MhaVolume]) -> None:
        # Store the volume so build_geometry() can reuse it later.
        self._mha_volume = mha_volume

        if mha_volume is None:
            # Clear the cached affine so we do not apply stale transforms.
            self._affine = None
            return

        # Normalize metadata to stable float arrays before building the affine.
        rotation, spacing, offset, center = self._normalize_metadata(mha_volume)
        # Cache the affine so geometry building is just a matrix multiply.
        self._affine = self._build_affine(rotation, spacing, offset, center)

    # Summary:
    # - Build geometry samples from above-threshold voxels with optional sampling.
    # - Input: `self`, `threshold` (float), `max_points` (int), `mode` (str).
    # - Returns: GeometryBuildingResult.
    def build_geometry(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_points: int = DEFAULT_MAX_POINTS,
        mode: str = "preview",
    ) -> GeometryBuildingResult:
        # Ignore mode for now; it is reserved for future quality levels.
        _ = mode

        # Bail out early when no volume is available.
        if self._mha_volume is None:
            return self._empty_result()

        # Guard against non-sensical caps to avoid divide-by-zero behaviors.
        if max_points <= 0:
            return self._empty_result()

        # Use the raw data view to avoid extra copies of large volumes.
        data = self._mha_volume.data
        if data is None or data.ndim < 3:
            return self._empty_result()

        # If a multi-channel volume is provided, use the first channel for now.
        if data.ndim > 3:
            # Keep channel handling simple until a multi-channel strategy is defined.
            data_view = data[..., 0]
        else:
            data_view = data

        z_dim, y_dim, x_dim = data_view.shape[:3]
        if z_dim == 0 or y_dim == 0 or x_dim == 0:
            return self._empty_result()

        # Keep a bounded sample so huge volumes never exhaust memory.
        sample_points = np.empty((0, 3), dtype=np.float32)
        sample_intensities = np.empty((0,), dtype=np.float32)
        sample_keys = np.empty((0,), dtype=np.float64)

        # Scan in small Z blocks so we never allocate a full-volume mask.
        chunk_depth = max(1, min(DEFAULT_CHUNK_DEPTH, z_dim))
        for z_start in range(0, z_dim, chunk_depth):
            z_end = min(z_start + chunk_depth, z_dim)

            # Slice the block as a view so memmaps stay lazy.
            block = data_view[z_start:z_end]
            # Build a temporary mask only for this small block.
            mask = block > threshold
            if not np.any(mask):
                # Skip empty blocks to keep scanning fast.
                continue

            # Convert block-local indices into global (z, y, x) indices.
            local_indices = np.argwhere(mask)
            if local_indices.size == 0:
                continue
            local_indices[:, 0] += z_start

            # Reorder indices into (x, y, z) so they match the expected index-space convention.
            candidate_points = local_indices[:, [2, 1, 0]].astype(np.float32, copy=False)

            # Pull the matching intensity values aligned with argwhere order.
            candidate_intensities = block[mask].astype(np.float32, copy=False)

            # Merge the current candidates into a capped random sample.
            (
                sample_points,
                sample_intensities,
                sample_keys,
            ) = self._merge_samples(
                sample_points,
                sample_intensities,
                sample_keys,
                candidate_points,
                candidate_intensities,
                max_points,
            )

        # Return empty outputs when nothing passes the threshold.
        if sample_points.size == 0:
            return self._empty_result()

        # Ensure we have a valid affine before applying it.
        if self._affine is None:
            return self._empty_result()

        # Build homogeneous coordinates so we can apply one affine per point.
        points_h = np.ones((sample_points.shape[0], 4), dtype=np.float64)
        points_h[:, :3] = sample_points

        # Apply the cached affine matrix in one batched multiply.
        points_world = points_h @ self._affine.T
        points_xyz = points_world[:, :3]

        # Compute axis-aligned bounds in world coordinates.
        min_xyz = points_xyz.min(axis=0)
        max_xyz = points_xyz.max(axis=0)

        return GeometryBuildingResult(
            points_xyz=points_xyz,
            intensities=sample_intensities,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
        )

    # Summary:
    # - Normalize metadata into stable float arrays for transform math.
    # - Input: `self`, `mha_volume` (MhaVolume).
    # - Returns: tuple of (rotation, spacing, offset, center) as numpy arrays.
    def _normalize_metadata(
        self, mha_volume: MhaVolume
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Read rotation as a 3x3 float matrix.
        rotation = np.array(mha_volume.TransformMatrix, dtype=np.float64)

        # Default spacing to 1s when metadata is missing.
        if mha_volume.ElementSpacing is None:
            spacing = np.ones(3, dtype=np.float64)
        else:
            spacing = np.array(mha_volume.ElementSpacing, dtype=np.float64)

        # Default offset to zeros when metadata is missing.
        if mha_volume.Offset is None:
            offset = np.zeros(3, dtype=np.float64)
        else:
            offset = np.array(mha_volume.Offset, dtype=np.float64)

        # Default center to zeros when metadata is missing.
        if mha_volume.CenterOfRotation is None:
            center = np.zeros(3, dtype=np.float64)
        else:
            center = np.array(mha_volume.CenterOfRotation, dtype=np.float64)

        # Ensure all vectors are length-3 to match the 3D assumptions here.
        spacing = self._ensure_length_three(spacing, fill_value=1.0)
        offset = self._ensure_length_three(offset, fill_value=0.0)
        center = self._ensure_length_three(center, fill_value=0.0)

        return rotation, spacing, offset, center

    # Summary:
    # - Build a 4x4 affine matrix from rotation, spacing, offset, and center.
    # - Input: `self`, `rotation` (np.ndarray), `spacing` (np.ndarray),
    #   `offset` (np.ndarray), `center` (np.ndarray).
    # - Returns: 4x4 numpy affine matrix.
    def _build_affine(
        self,
        rotation: np.ndarray,
        spacing: np.ndarray,
        offset: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        # Fold spacing into the linear part so index coordinates scale to physical units.
        scale_matrix = np.diag(spacing)
        linear = rotation @ scale_matrix

        # Translate to center (index space), rotate/scale, translate back, then offset.
        # Center is in index units, so we translate back in physical units using spacing.
        center_physical = spacing * center
        translation = (-linear @ center) + center_physical + offset

        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = linear
        affine[:3, 3] = translation
        return affine

    # Summary:
    # - Merge candidate points into the bounded random sample.
    # - Input: `self`, sample arrays, candidate arrays, `max_points` (int).
    # - Returns: updated sample arrays (points, intensities, keys).
    def _merge_samples(
        self,
        sample_points: np.ndarray,
        sample_intensities: np.ndarray,
        sample_keys: np.ndarray,
        candidate_points: np.ndarray,
        candidate_intensities: np.ndarray,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_count = candidate_points.shape[0]
        if candidate_count == 0:
            return sample_points, sample_intensities, sample_keys

        # Assign a random key to each candidate so we can keep the top-K uniformly.
        candidate_keys = self._rng.random(candidate_count)

        # Fast path: no existing sample yet.
        if sample_points.size == 0:
            if candidate_count <= max_points:
                return candidate_points, candidate_intensities, candidate_keys
            keep_idx = np.argpartition(candidate_keys, -max_points)[-max_points:]
            return (
                candidate_points[keep_idx],
                candidate_intensities[keep_idx],
                candidate_keys[keep_idx],
            )

        # Combine existing samples with the new candidates.
        combined_keys = np.concatenate([sample_keys, candidate_keys])
        combined_points = np.vstack([sample_points, candidate_points])
        combined_intensities = np.concatenate([sample_intensities, candidate_intensities])

        if combined_keys.size <= max_points:
            return combined_points, combined_intensities, combined_keys

        # Keep the top-K keys to maintain a uniform sample over all seen candidates.
        keep_idx = np.argpartition(combined_keys, -max_points)[-max_points:]
        return (
            combined_points[keep_idx],
            combined_intensities[keep_idx],
            combined_keys[keep_idx],
        )

    # Summary:
    # - Ensure a vector is length 3 by truncating or padding with a fill value.
    # - Input: `self`, `values` (np.ndarray), `fill_value` (float).
    # - Returns: length-3 numpy array.
    def _ensure_length_three(self, values: np.ndarray, fill_value: float) -> np.ndarray:
        # Truncate extra dimensions so downstream math stays 3D.
        if values.size >= 3:
            return values[:3].astype(np.float64, copy=False)

        # Pad missing dimensions so spacing/offset/center always has length 3.
        padded = np.full(3, fill_value, dtype=np.float64)
        padded[: values.size] = values.astype(np.float64, copy=False)
        return padded

    # Summary:
    # - Build a consistent empty result for early exits.
    # - Input: `self`.
    # - Returns: GeometryBuildingResult with empty arrays and None bounds.
    def _empty_result(self) -> GeometryBuildingResult:
        # Keep shapes consistent so UI updates can be simplified.
        empty_points = np.empty((0, 3), dtype=np.float64)
        empty_intensities = np.empty((0,), dtype=np.float32)
        return GeometryBuildingResult(
            points_xyz=empty_points,
            intensities=empty_intensities,
            min_xyz=None,
            max_xyz=None,
        )
