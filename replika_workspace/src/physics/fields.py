from dataclasses import dataclass
import numpy as np
from typing import Tuple, Iterator, Dict, Any, List, Optional
from pathlib import Path

@dataclass(frozen=True)
class VelocitySnapshot:
    """
    Container for a 3D solenoidal velocity field snapshot.
    Maps directly to VTK STRUCTURED_POINTS data (DS1).
    """
    vx: np.ndarray  # 3D array of x-velocity components
    vy: np.ndarray  # 3D array of y-velocity components
    vz: np.ndarray  # 3D array of z-velocity components
    time: float     # Simulation time from VTK header
    grid_dimensions: Tuple[int, int, int]  # (129, 129, 129) per DS1 profile
    spacing: float  # Grid cell size (e.g., 0.0078125)
    origin: Tuple[float, float, float] = (-0.5, -0.5, -0.5)
    domain_size_L: float = 1.0

    def __post_init__(self):
        """Basic validation of field shapes against dimensions."""
        expected_shape = self.grid_dimensions
        for field, name in zip([self.vx, self.vy, self.vz], ['vx', 'vy', 'vz']):
            if field.shape != expected_shape:
                raise ValueError(f"Field {name} shape {field.shape} does not match dimensions {expected_shape}")


class GridGeometryManager:
    """
    Manages the 3D spatial grid geometry for solvers and analysis.
    
    Handles the Structured Points container used in DS1 and provides 
    coordinate generation for tracer seeding.
    """

    def generate_uniform_cartesian_points(self, n_points: int, domain_range: Tuple[float, float]) -> np.ndarray:
        """
        Generates random uniform (x, y, z) coordinates within the specified domain.
        Used for 'initial_seeding' in EXP1 and EXP2.

        Args:
            n_points: Number of points to generate (e.g., 8000 tracers).
            domain_range: The [min, max] range for each axis (typically [0, 1] or [-0.5, 0.5]).

        Returns:
            An array of shape (n_points, 3) containing physical coordinates.
        """
        low, high = domain_range
        return np.random.uniform(low, high, (n_points, 3))

    def get_grid_coordinate_arrays(self, dimensions: Tuple[int, int, int], spacing: float, origin: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs the 1D coordinate vectors along each axis for a structured grid.

        Args:
            dimensions: (nx, ny, nz) e.g., (129, 129, 129).
            spacing: dx, dy, dz spacing e.g., 0.0078125.
            origin: (x0, y0, z0) starting point e.g., (-0.5, -0.5, -0.5).

        Returns:
            Tuple of (x_coords, y_coords, z_coords) arrays.
        """
        nx, ny, nz = dimensions
        ox, oy, oz = origin
        
        x_coords = ox + np.arange(nx) * spacing
        y_coords = oy + np.arange(ny) * spacing
        z_coords = oz + np.arange(nz) * spacing
        
        return x_coords, y_coords, z_coords
class FieldMappingService:
    """
    Performs mapping between grid-space indices and physical quantities in the 3D domain.
    """

    def map_vtk_to_physical(self, grid_indices: np.ndarray, snapshot: VelocitySnapshot) -> np.ndarray:
        """
        Transforms integer grid indices (i, j, k) to physical domain coordinates (x, y, z).

        Args:
            grid_indices: Array of shape (N, 3) representing grid indices.
            snapshot: The reference snapshot containing origin and spacing metadata.

        Returns:
            Physical coordinates array of shape (N, 3) mapped into the periodic [0, 1] cube.
        """
        origin = np.array(snapshot.origin)
        physical_coords = origin + grid_indices * snapshot.spacing
        return physical_coords % snapshot.domain_size_L

    def extract_velocity_components(self, snapshot: VelocitySnapshot) -> np.ndarray:
        """
        Combines individual vx, vy, vz components into a single vector field array.

        Returns:
            Array of shape (129, 129, 129, 3) containing full velocity vectors.
        """
        return np.stack([snapshot.vx, snapshot.vy, snapshot.vz], axis=-1)

    def map_mass_density(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Maps raw 'dens' scalar field data from DS1 VTK files to a normalized density field.

        Args:
            raw_data: 3D array of scalar values sampled at structured points.

        Returns:
            3D array of normalized mass density (normalized by mean).
        """
        if raw_data.size == 0:
            return raw_data
            
        mean_val = np.mean(raw_data)
        if mean_val > 0:
            return raw_data / mean_val
        return raw_data
class AnisotropicAlignmentCalculator:
    """
    Calculates vector alignments and projections relative to the large-scale flow field (V_LS).
    
    Supports EXP2 (Anisotropic Dispersion Analysis) requirements:
    - V_LS_hat calculation
    - Parallel and Perpendicular displacement components
    """

    def calculate_vls_unit_vector(self, filtered_velocity: np.ndarray) -> np.ndarray:
        """
        Calculates the unit vector field V_LS_hat based on Eq: V_LS_hat = V_LS / |V_LS|.

        Args:
            filtered_velocity: 3D vector field (vx, vy, vz) filtered using spectral modes n=1-3.

        Returns:
            Unit vector field of the same shape as input.
        """
        # Assume input is (..., 3)
        norms = np.linalg.norm(filtered_velocity, axis=-1, keepdims=True)
        # Avoid division by zero: if norm is 0, result remains 0
        return np.divide(filtered_velocity, norms, out=np.zeros_like(filtered_velocity), where=norms != 0)

    def project_local_velocity(self, local_velocity: np.ndarray, vls_hat: np.ndarray) -> np.ndarray:
        """
        Projects the local velocity onto the large-scale orientation.
        
        Calculation for parallel component v_|| = <v . V_LS_hat>V_LS_hat.

        Args:
            local_velocity: The local velocity vector at a specific position.
            vls_hat: The large-scale unit vector at the same position.

        Returns:
            Vector component parallel to V_LS_hat.
        """
        dot_product = np.sum(local_velocity * vls_hat, axis=-1, keepdims=True)
        return dot_product * vls_hat

    def calculate_transverse_component(self, local_velocity: np.ndarray, vls_hat: np.ndarray) -> np.ndarray:
        """
        Calculates the velocity component perpendicular to the large-scale flow.
        
        Equation: v_perp = v - (v . V_LS_hat)V_LS_hat.

        Args:
            local_velocity: Local velocity vector.
            vls_hat: Large-scale unit direction.

        Returns:
            Perpendicular vector component.
        """
        v_parallel = self.project_local_velocity(local_velocity, vls_hat)
        return local_velocity - v_parallel

    def calculate_msd_anisotropy(self, displacements: np.ndarray, vls_hat: np.ndarray) -> float:
        """
        Calculates lambda(t) = MSD_parallel(t) / MSD_perp(t).
        
        Args:
            displacements: (N, 3) array of tracer displacements delta_x(t).
            vls_hat: (N, 3) array of unit vectors V_LS_hat at tracer locations.
            
        Returns:
            Anisotropy ratio lambda.
        """
        v_parallel = self.project_local_velocity(displacements, vls_hat)
        msd_parallel = np.mean(np.sum(v_parallel**2, axis=-1))
        
        v_perp = displacements - v_parallel
        msd_perp = np.mean(np.sum(v_perp**2, axis=-1))
        
        if msd_perp == 0:
            return 1.0 if msd_parallel == 0 else float('inf')
        return float(msd_parallel / msd_perp)

    def compute_spectral_filter(self, velocity_field: np.ndarray, filter_range: Tuple[int, int] = (1, 3)) -> np.ndarray:
        """
        Filters the velocity field to keep only spectral components within filter_range.
        Usually used to compute V_LS.
        
        Args:
            velocity_field: (NX, NY, NZ, 3) array.
            filter_range: (k_min, k_max) inclusive.
            
        Returns:
            Filtered velocity field of the same shape.
        """
        nx, ny, nz, d = velocity_field.shape
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        kz = np.fft.fftfreq(nz) * nz
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        mask = (k_mag >= filter_range[0]) & (k_mag <= filter_range[1])
        
        filtered_field = np.zeros_like(velocity_field)
        for i in range(d):
            # Compute 3D FFT
            v_fft = np.fft.fftn(velocity_field[..., i])
            # Apply sharp spectral mask
            v_fft_filtered = v_fft * mask
            # Back to physical space
            filtered_field[..., i] = np.real(np.fft.ifftn(v_fft_filtered))
            
        return filtered_field


class SolenoidalFieldProvider:
    """
    Provides access to 3D solenoidal velocity snapshots (DS1) for numerical experiments.
    """

    def get_snapshot_sequence(self, snapshot_paths: Iterator[Path]) -> Iterator[VelocitySnapshot]:
        """
        Yields a stream of validated VelocitySnapshot objects.

        Args:
            snapshot_paths: Iterable of VTK file paths representing the sequence (DS1 residency).

        Yields:
            VelocitySnapshot objects with (vx, vy, vz) components.
        """
        # Local imports to avoid circular dependency with io.vtk_reader
        from src.io.vtk_reader import SolenoidalSnapshotReader
        from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
        
        reader = SolenoidalSnapshotReader(
            header_parser=VTKHeaderParser(),
            layout_extractor=VTKStructuredPointExtractor(),
            field_extractor=VTKFieldExtractor()
        )
        
        for path in snapshot_paths:
            yield reader.load_snapshot(path)

    def get_turbulence_field_metadata(self) -> Dict[str, Any]:
        """
        Returns physical properties of the solenoidal turbulence field (e.g., v_rms, Te).

        Returns:
            Dictionary containing 'v_rms' (0.38) and 'large_eddy_time_te' (2.6).
        """
        return {
            'v_rms': 0.38,
            'large_eddy_time_te': 2.6
        }
