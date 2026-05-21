import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, Iterator, List, Optional

from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.physics.fields import VelocitySnapshot

@dataclass(frozen=True)
class DS1SolenoidalManifest:
    """
    Data contract for DS1: 3D Solenoidal Turbulence Velocity Snapshots.
    
    Attributes:
        dataset_name: Name of the dataset as per repo-purpose.
        grid_dimensions: (129, 129, 129) per VTK sample lines.
        domain_size_L: Cubic domain size (L=1.0).
        file_pattern: Filename structure 'Turb.hydro_w.*.vtk'.
        expected_sampling_rate: Time delta between snapshot indices (e.g., 0.1).
    """
    dataset_name: str = '3D Solenoidal Turbulence Velocity Snapshots'
    grid_dimensions: Tuple[int, int, int] = (129, 129, 129)
    domain_size_L: float = 1.0
    file_pattern: str = 'Turb.hydro_w.*.vtk'
    expected_sampling_rate: float = 0.1


class SolenoidalSnapshotReader:
    """
    High-level reader for 3D solenoidal velocity field snapshots (DS1).
    
    Interprets binary VTK field blocks (scalars/vectors) and maps them to
    domain-specific VelocitySnapshot objects for Lagrangian integration (EXP1).
    """

    def __init__(self, header_parser: VTKHeaderParser, 
                 layout_extractor: VTKStructuredPointExtractor, 
                 field_extractor: VTKFieldExtractor):
        """
        Initialize with low-level VTK parsing components.

        Args:
            header_parser: Handles VTK version, title, and metadata.
            layout_extractor: Extracts DIMENSIONS, ORIGIN, and SPACING.
            field_extractor: Performs binary read of velocity/density data.
        """
        self.header_parser = header_parser
        self.layout_extractor = layout_extractor
        self.field_extractor = field_extractor
        self._manifest = DS1SolenoidalManifest()

    def load_snapshot(self, path: Path) -> VelocitySnapshot:
        """
        Reads a single VTK file and returns a validated VelocitySnapshot.

        Args:
            path: Path to a binary VTK STRUCTURED_POINTS file.

        Returns:
            VelocitySnapshot containing field tensors and timing metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file geometry deviates from domain expectations.
        """
        if not path.exists():
            raise FileNotFoundError(f"VTK snapshot not found: {path}")

        with open(path, 'rb') as f:
            header = self.header_parser.parse(f)
            
            # Reset to beginning before extracting layout to ensure keywords like DIMENSIONS are found
            f.seek(0)
            layout = self.layout_extractor.extract_layout(f)

            # Enforce domain constraints from manifest
            if layout.dimensions != self._manifest.grid_dimensions:
                raise ValueError(
                    f"Grid dimensions {layout.dimensions} do not match "
                    f"manifest expectations {self._manifest.grid_dimensions}"
                )

            # Extract time metadata
            # 1. Try from header title (e.g., "Snapshot at t= 189.030")
            time = None
            title_match = re.search(r"t\s*=\s*([\d\.\-]+)", header.title)
            if title_match:
                try:
                    time = float(title_match.group(1))
                except ValueError:
                    pass
            
            # 2. Fallback to filename (e.g., Turb.hydro_w.18903.vtk)
            if time is None:
                filename_match = re.search(r"w\.(\d+)\.vtk", path.name)
                if filename_match:
                    # Based on DS1 profile: index 18903 matches time 189.03
                    time = float(filename_match.group(1)) * 0.01
                else:
                    time = 0.0

            # Find and extract the velocity field 'v'
            try:
                try:
                    v_offset = self.layout_extractor.locate_field_block(f, 'v')
                except KeyError:
                    # Some datasets might use 'velocity' or 'hydro_w'
                    try:
                        v_offset = self.layout_extractor.locate_field_block(f, 'velocity')
                    except KeyError:
                        try:
                            # Support the specific attribute name used in the turbulence snapshots (DS1)
                            v_offset = self.layout_extractor.locate_field_block(f, 'hydro_w')
                        except KeyError:
                            # Try separate components if VECTORS block not found
                            vx_offset = self.layout_extractor.locate_field_block(f, 'velx')
                            vy_offset = self.layout_extractor.locate_field_block(f, 'vely')
                            vz_offset = self.layout_extractor.locate_field_block(f, 'velz')
                            
                            nx, ny, nz = layout.dimensions
                            nx_c, ny_c, nz_c = nx - 1, ny - 1, nz - 1
                            count = nx_c * ny_c * nz_c
                            
                            vx_flat = self.field_extractor.extract_scalar_field(f, vx_offset, count)
                            vy_flat = self.field_extractor.extract_scalar_field(f, vy_offset, count)
                            vz_flat = self.field_extractor.extract_scalar_field(f, vz_offset, count)
                            
                            # Combine and pad from (nz-1, ny-1, nx-1) to (nz, ny, nx)
                            vx_c = vx_flat.reshape((nz_c, ny_c, nx_c))
                            vy_c = vy_flat.reshape((nz_c, ny_c, nx_c))
                            vz_c = vz_flat.reshape((nz_c, ny_c, nx_c))
                            
                            v_field_c = np.stack([vx_c, vy_c, vz_c], axis=-1)
                            v_field_padded = np.zeros((nz, ny, nx, 3), dtype=v_field_c.dtype)
                            v_field_padded[:nz_c, :ny_c, :nx_c, :] = v_field_c
                            
                            # Periodic padding
                            v_field_padded[nz_c, :, :, :] = v_field_padded[0, :, :, :]
                            v_field_padded[:, ny_c, :, :] = v_field_padded[:, 0, :, :]
                            v_field_padded[:, :, nx_c, :] = v_field_padded[:, :, 0, :]
                            
                            v_field = v_field_padded.transpose(2, 1, 0, 3)
                            
                            return VelocitySnapshot(
                                vx=np.ascontiguousarray(v_field[:, :, :, 0]),
                                vy=np.ascontiguousarray(v_field[:, :, :, 1]),
                                vz=np.ascontiguousarray(v_field[:, :, :, 2]),
                                time=time,
                                grid_dimensions=layout.dimensions,
                                spacing=layout.spacing[0],
                                origin=layout.origin,
                                domain_size_L=self._manifest.domain_size_L
                            )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Could not find velocity field in {path}: {str(e)}")
            
            nx, ny, nz = layout.dimensions
            nx_c, ny_c, nz_c = nx - 1, ny - 1, nz - 1
            count = nx_c * ny_c * nz_c
            v_flat = self.field_extractor.extract_vector_field(f, v_offset, count)
            
            # Pad from (nz-1, ny-1, nx-1, 3) to (nz, ny, nx, 3)
            v_field_c = v_flat.reshape((nz_c, ny_c, nx_c, 3))
            v_field_padded = np.zeros((nz, ny, nx, 3), dtype=v_field_c.dtype)
            v_field_padded[:nz_c, :ny_c, :nx_c, :] = v_field_c
            
            # Periodic padding
            v_field_padded[nz_c, :, :, :] = v_field_padded[0, :, :, :]
            v_field_padded[:, ny_c, :, :] = v_field_padded[:, 0, :, :]
            v_field_padded[:, :, nx_c, :] = v_field_padded[:, :, 0, :]
            
            v_field = v_field_padded.transpose(2, 1, 0, 3)
            
            return VelocitySnapshot(
                vx=np.ascontiguousarray(v_field[:, :, :, 0]),
                vy=np.ascontiguousarray(v_field[:, :, :, 1]),
                vz=np.ascontiguousarray(v_field[:, :, :, 2]),
                time=time,
                grid_dimensions=layout.dimensions,
                spacing=layout.spacing[0],  # Assume uniform spacing
                origin=layout.origin,
                domain_size_L=self._manifest.domain_size_L
            )

    def get_manifest(self) -> DS1SolenoidalManifest:
        """
        Exposes the metadata manifest for the DS1 dataset.

        Returns:
            The DS1 manifest defining grid and domain constraints.
        """
        return self._manifest


class VTKSequenceProcessor:
    """
    Manages the sequential processing of VTK snapshots to minimize memory consumption.
    
    Essential for EXP1 Procedure Phase 2: 'Load consecutive pairs of VTK 
    snapshots from DS1' without loading the entire 200-snapshot series into RAM.
    """

    def __init__(self, snapshot_reader: SolenoidalSnapshotReader):
        """
        Initialize with a snapshot reader instance.

        Args:
            snapshot_reader: Reader capable of parsing individual snapshots.
        """
        self.snapshot_reader = snapshot_reader

    def stream_snapshots(self, paths: Iterable[Path]) -> Iterator[VelocitySnapshot]:
        """
        Yields VelocitySnapshot objects one by one from a provided file list.

        Args:
            paths: Ordered sequence of VTK files to process.

        Yields:
            Hydrated VelocitySnapshot for each path.
        """
        for path in paths:
            yield self.snapshot_reader.load_snapshot(path)

    def stream_snapshot_pairs(self, paths: Iterable[Path]) -> Iterator[Tuple[VelocitySnapshot, VelocitySnapshot]]:
        """
        Yields consecutive pairs of snapshots (t, t+delta_t) for integration.
        
        Supports EXP1 RK4 integration which requires velocity information at 
        overlapping time intervals for interpolation.

        Args:
            paths: Ordered sequence of VTK files to process.

        Yields:
            Tuple of (Previous Snapshot, Current Snapshot).
        """
        it = self.stream_snapshots(paths)
        try:
            prev = next(it)
        except StopIteration:
            return

        for curr in it:
            yield prev, curr
            prev = curr
