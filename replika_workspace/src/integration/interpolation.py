import numpy as np
from dataclasses import dataclass
from typing import Tuple
from src.physics.fields import VelocitySnapshot

@dataclass(frozen=True)
class InterpolationStencil:
    """
    Container for indices and local fractional offsets used in 3D interpolation.
    
    Attributes:
        base_indices: (N, 3) array of integer grid indices (i, j, k) for the lower-left-bottom corner.
        fractions: (N, 3) array of normalized distances (dx, dy, dz) within the cell [0, 1).
    """
    base_indices: np.ndarray
    fractions: np.ndarray

class TrilinearGridKernel:
    """
    Stateless numerical kernels for 3D trilinear interpolation routines.
    
    Supports EXP1 (Tracer Trajectory Generation) by providing exact mathematical 
    primitives for sampling velocity fields at sub-grid locations.
    """

    def lookup_spatial_indices(self, positions: np.ndarray, origin: Tuple[float, float, float], spacing: float) -> InterpolationStencil:
        """
        Maps continuous 3D coordinates to grid indices and fractional offsets.
        
        Feature: integration/integration/interpolation/kernels/trilinear spatial lookup routine

        Args:
            positions: (N, 3) array of tracer coordinates (x, y, z).
            origin: The (x, y, z) coordinate of the (0, 0, 0) grid point.
            spacing: The uniform distance between grid nodes.

        Returns:
            InterpolationStencil containing base indices and unit-cell fractions.
        """
        relative_positions = (positions - np.array(origin)) / spacing
        base_indices_float = np.floor(relative_positions)
        base_indices = base_indices_float.astype(np.int64)
        fractions = relative_positions - base_indices_float
        
        return InterpolationStencil(base_indices=base_indices, fractions=fractions)

    def compute_fractional_weights(self, fractions: np.ndarray) -> np.ndarray:
        """
        Computes the volume-based weighting factors for the 8 corners of a 3D cell.
        
        Feature: integration/integration/interpolation/kernels/trilinear weight computation

        Equation:
            Weights follow the products of (1 - d) and (d) for each dimension x, y, z.
            e.g., w_000 = (1-dx)(1-dy)(1-dz), w_111 = dx*dy*dz.

        Args:
            fractions: (N, 3) array of fractional offsets (dx, dy, dz).

        Returns:
            (N, 8) array of scalar weights for corners (000, 100, 010, 110, 001, 101, 011, 111).
        """
        dx = fractions[:, 0]
        dy = fractions[:, 1]
        dz = fractions[:, 2]
        
        # Order: (000, 100, 010, 110, 001, 101, 011, 111)
        w000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        w100 = dx * (1.0 - dy) * (1.0 - dz)
        w010 = (1.0 - dx) * dy * (1.0 - dz)
        w110 = dx * dy * (1.0 - dz)
        w001 = (1.0 - dx) * (1.0 - dy) * dz
        w101 = dx * (1.0 - dy) * dz
        w011 = (1.0 - dx) * dy * dz
        w111 = dx * dy * dz
        
        return np.stack([w000, w100, w010, w110, w001, w101, w011, w111], axis=1)

    def calculate_interpolation_coefficients(self, snapshots: np.ndarray, stencil: InterpolationStencil) -> np.ndarray:
        """
        Calculates the vertex values (coefficients) for each interpolation point.
        
        Feature: integration/integration/interpolation/kernels/trilinear grid coefficient calculation

        Handles the gathering of field values from the 8 vertices surrounding each point,
        accounting for periodic boundary wrapping if indices exceed dimensions.

        Args:
            snapshots: The 3D grid data array (e.g., vx, vy, or vz).
            stencil: The stencil containing base indices for lookup.

        Returns:
            (N, 8) array of field values at the surrounding grid nodes.
        """
        base_indices = stencil.base_indices
        dims = np.array(snapshots.shape)
        
        # Corners in the same order as compute_fractional_weights:
        # (000, 100, 010, 110, 001, 101, 011, 111)
        offsets = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        
        N = base_indices.shape[0]
        vertex_values = np.empty((N, 8))
        
        for i in range(8):
            indices = (base_indices + offsets[i]) % dims
            vertex_values[:, i] = snapshots[indices[:, 0], indices[:, 1], indices[:, 2]]
            
        return vertex_values

    def evaluate_trilinear_sum(self, vertex_values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Performs the weighted summation of vertex values to produce the interpolated result.
        
        Feature: integration/integration/interpolation/kernels/trilinear grid evaluation kernel

        Equation:
            V(x,y,z) = V_000(1-xd)(1-yd)(1-zd) + V_100*xd(1-yd)(1-zd) + V_010(1-xd)yd(1-zd) + 
                       V_001(1-xd)(1-yd)zd + V_110*xd*yd(1-zd) + V_101*xd(1-yd)zd + 
                       V_011(1-xd)yd*zd + V_111*xd*yd*zd

        Args:
            vertex_values: (N, 8) array of values at the 8 corners.
            weights: (N, 8) array of pre-calculated trilinear weights.

        Returns:
            (N,) array of interpolated values at the target positions.
        """
        return np.sum(vertex_values * weights, axis=1)
class TrilinearGridInterpolator:
    """
    Orchestrator for evaluating continuous fields from discrete structured grids.
    
    Used in EXP1 to generate the Lagrangian tracer trajectory dataset (DS2) from 
    raw velocity snapshots (DS1).
    """

    def __init__(self):
        """Initializes the interpolator with a standard trilinear kernel."""
        self.kernel = TrilinearGridKernel()

    def evaluate_velocity_field(self, snapshot: VelocitySnapshot, positions: np.ndarray) -> np.ndarray:
        """
        Evaluates the full 3D velocity vector (vx, vy, vz) at given positions.
        
        Feature: integration/integration/interpolation/kernels/trilinear grid evaluation

        Args:
            snapshot: VelocitySnapshot containing vx, vy, vz components and grid metadata.
            positions: (N, 3) array of coordinates in physical space.

        Returns:
            (N, 3) array of interpolated velocity vectors.
        """
        # Handle periodic boundary conditions by wrapping positions to the primary domain
        L = snapshot.domain_size_L
        origin = np.array(snapshot.origin)
        wrapped_positions = origin + np.mod(positions - origin, L)
        
        # Perform trilinear interpolation for each component
        stencil = self.kernel.lookup_spatial_indices(wrapped_positions, snapshot.origin, snapshot.spacing)
        weights = self.kernel.compute_fractional_weights(stencil.fractions)
        
        vx_interp = self.kernel.evaluate_trilinear_sum(
            self.kernel.calculate_interpolation_coefficients(snapshot.vx, stencil), weights
        )
        vy_interp = self.kernel.evaluate_trilinear_sum(
            self.kernel.calculate_interpolation_coefficients(snapshot.vy, stencil), weights
        )
        vz_interp = self.kernel.evaluate_trilinear_sum(
            self.kernel.calculate_interpolation_coefficients(snapshot.vz, stencil), weights
        )
        
        return np.stack([vx_interp, vy_interp, vz_interp], axis=-1)

    def evaluate_on_cartesian_grid(self, field: np.ndarray, dimensions: Tuple[int, int, int], spacing: float, origin: Tuple[float, float, float], positions: np.ndarray) -> np.ndarray:
        """
        General-purpose evaluation for any scalar field defined on a Cartesian grid.
        
        Feature: integration/integration/interpolation/kernels/trilinear cartesian grid evaluation

        Args:
            field: 3D array of values.
            dimensions: (nx, ny, nz) of the grid.
            spacing: The grid cell size.
            origin: The (x, y, z) coordinate of the first grid point.
            positions: (N, 3) array of target coordinates.

        Returns:
            (N,) array of interpolated scalar values.
        """
        # Periodic wrap based on L=1.0 as specified for this repository's experiments
        L = 1.0
        origin_np = np.array(origin)
        wrapped_positions = origin_np + np.mod(positions - origin_np, L)
        
        stencil = self.kernel.lookup_spatial_indices(wrapped_positions, origin, spacing)
        weights = self.kernel.compute_fractional_weights(stencil.fractions)
        
        vertex_values = self.kernel.calculate_interpolation_coefficients(field, stencil)
        return self.kernel.evaluate_trilinear_sum(vertex_values, weights)
