import numpy as np
from dataclasses import dataclass
from typing import Protocol
from src.physics.fields import VelocitySnapshot

@dataclass(frozen=True)
class VelocityGradientTensorField:
    """
    A domain object representing the velocity gradient tensor field (∇v).
    
    Each component is a 3D numpy array corresponding to the spatial grid index.
    The tensor components are defined as T_ij = ∂v_i / ∂x_j.
    
    This field is required for calculating the Q-criterion for EXP3 (Vortex Residence Time Calculation):
    Q = 0.5 * (||Omega||^2 - ||S||^2)
    where Omega = 0.5 * (∇v - (∇v)^T) and S = 0.5 * (∇v + (∇v)^T).
    """
    dvx_dx: np.ndarray
    dvx_dy: np.ndarray
    dvx_dz: np.ndarray
    dvy_dx: np.ndarray
    dvy_dy: np.ndarray
    dvy_dz: np.ndarray
    dvz_dx: np.ndarray
    dvz_dy: np.ndarray
    dvz_dz: np.ndarray

class TensorGradientDiscretizer(Protocol):
    """
    Protocol for numerical discretization schemes used to compute partial 
    derivatives of a 3D field on a structured grid.
    """

    def differentiate(self, field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
        """
        Calculates the partial derivative of a scalar field along a given axis.

        Args:
            field: 3D array representing a velocity component on the 129^3 grid.
            axis: The axis index to differentiate along (0=x, 1=y, 2=z).
            spacing: The physical grid spacing Δx (e.g., 0.0078125).

        Returns:
            np.ndarray: The discretized partial derivative field.
        """
        ...

class CentralDifferenceDerivativeScheme:
    """
    Implements a second-order central difference scheme for spatial derivatives.
    
    Specifically handles periodic boundary conditions for the domain size L=1.0
    and the 129^3 grid points of DS1.
    
    The derivative is calculated as: ∂f/∂x ≈ (f_{i+1} - f_{i-1}) / (2Δx).
    """

    def differentiate(self, field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
        """
        Computes the central difference derivative using periodic wrapping if appropriate.

        Args:
            field: 3D velocity component array.
            axis: Axis index (0, 1, or 2).
            spacing: Physical distance between adjacent grid points.

        Returns:
            np.ndarray: Derivative field holding values for the gradient calculation.
        """
        # Get the dimension of the field along the specified axis
        n = field.shape[axis]
        
        # If the dimension is too small for central difference, return zeros
        if n < 2:
            return np.zeros_like(field)

        # Check for periodicity: last point must match first point for periodic scheme
        first_vals = np.take(field, 0, axis=axis)
        last_vals = np.take(field, -1, axis=axis)
        is_periodic = np.allclose(first_vals, last_vals)

        if is_periodic and n > 2:
            # Generate indices for neighbors with periodic boundary conditions.
            # For a periodic grid with n points where the last point is a duplicate 
            # of the first (field[..., n-1] == field[..., 0]):
            # After i=n-1 (duplicate of 0), the next point is i=1.
            # Before i=0 (duplicate of n-1), the previous point is i=n-2.
            
            idx_plus = np.arange(n)
            idx_plus = np.roll(idx_plus, -1)
            idx_plus[n - 1] = 1 # Neighbor of duplicate endpoint is the second point
            
            idx_minus = np.arange(n)
            idx_minus = np.roll(idx_minus, 1)
            idx_minus[0] = n - 2 # Neighbor of first point is the penultimate point
            
            # Extract components using the periodic indices
            f_plus = np.take(field, idx_plus, axis=axis)
            f_minus = np.take(field, idx_minus, axis=axis)
            
            # Compute central difference (second-order)
            return (f_plus - f_minus) / (2.0 * spacing)
        else:
            # For non-periodic fields or small grids, use np.gradient which handles
            # boundaries using second-order one-sided differences.
            return np.gradient(field, spacing, axis=axis)
class VelocityGradientCalculator:
    """
    Service responsible for the end-to-end velocity gradient calculation for a snapshot.
    
    Applies numerical discretization to compute the full Jacobian ∇v for each point
    in the DS1 velocity snapshots, enabling vortex detection in EXP3.
    """

    def __init__(self, discretizer: TensorGradientDiscretizer):
        """
        Inits with a numerical differentiation policy.

        Args:
            discretizer: Implementation of the gradient discretization logic.
        """
        self.discretizer = discretizer

    def compute_gradient_tensor(self, snapshot: VelocitySnapshot) -> VelocityGradientTensorField:
        """
        Computes the 3x3 velocity gradient tensor field from a snapshot.

        Args:
            snapshot: Input 3D solenoidal velocity field.

        Returns:
            VelocityGradientTensorField: The complete tensor components.

        Raises:
            ValueError: If input snapshot dimensions or spacing are invalid.
        """
        if snapshot.spacing <= 0:
            raise ValueError(f"Invalid spacing: {snapshot.spacing}")

        # Compute partial derivatives for each velocity component
        # vx components: ∂vx/∂x, ∂vx/∂y, ∂vx/∂z
        dvx_dx = self.discretizer.differentiate(snapshot.vx, axis=0, spacing=snapshot.spacing)
        dvx_dy = self.discretizer.differentiate(snapshot.vx, axis=1, spacing=snapshot.spacing)
        dvx_dz = self.discretizer.differentiate(snapshot.vx, axis=2, spacing=snapshot.spacing)

        # vy components: ∂vy/∂x, ∂vy/∂y, ∂vy/∂z
        dvy_dx = self.discretizer.differentiate(snapshot.vy, axis=0, spacing=snapshot.spacing)
        dvy_dy = self.discretizer.differentiate(snapshot.vy, axis=1, spacing=snapshot.spacing)
        dvy_dz = self.discretizer.differentiate(snapshot.vy, axis=2, spacing=snapshot.spacing)

        # vz components: ∂vz/∂x, ∂vz/∂y, ∂vz/∂z
        dvz_dx = self.discretizer.differentiate(snapshot.vz, axis=0, spacing=snapshot.spacing)
        dvz_dy = self.discretizer.differentiate(snapshot.vz, axis=1, spacing=snapshot.spacing)
        dvz_dz = self.discretizer.differentiate(snapshot.vz, axis=2, spacing=snapshot.spacing)

        return VelocityGradientTensorField(
            dvx_dx=dvx_dx, dvx_dy=dvx_dy, dvx_dz=dvx_dz,
            dvy_dx=dvy_dx, dvy_dy=dvy_dy, dvy_dz=dvy_dz,
            dvz_dx=dvz_dx, dvz_dy=dvz_dy, dvz_dz=dvz_dz
        )
