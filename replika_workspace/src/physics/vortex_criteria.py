import numpy as np
from typing import Tuple
from src.physics.tensors import VelocityGradientTensorField

class QCriterionCalculator:
    """
    Computes the Q-criterion scalar field from velocity gradient tensors to identify coherent vortices.
    
    This calculator implements the second invariant of the velocity gradient tensor for 
    incompressible flows as specified in EXP3. Regions where Q > 0 indicate rotation-dominated 
    flow (vortices).
    
    Equation:
        Q = 0.5 * (||Omega||^2 - ||S||^2)
        Where:
        Omega = 0.5 * (∇v - (∇v)^T) is the rotation tensor.
        S = 0.5 * (∇v + (∇v)^T) is the strain-rate tensor.
    """

    def _validate_gradients(self, gradients: VelocityGradientTensorField) -> None:
        """
        Validates that all components of the velocity gradient tensor field are present 
        and have the same shape.
        """
        if gradients is None:
            raise ValueError("VelocityGradientTensorField cannot be None.")

        components = [
            gradients.dvx_dx, gradients.dvx_dy, gradients.dvx_dz,
            gradients.dvy_dx, gradients.dvy_dy, gradients.dvy_dz,
            gradients.dvz_dx, gradients.dvz_dy, gradients.dvz_dz
        ]
        
        for i, c in enumerate(components):
            if c is None:
                raise ValueError(f"Velocity gradient component at index {i} is None.")

        shape = components[0].shape
        for i, c in enumerate(components[1:]):
            if c.shape != shape:
                raise ValueError(f"Inconsistent dimensions in velocity gradient tensor components. "
                                 f"Expected {shape}, but found {c.shape}.")

    def calculate_gradient_invariants(self, gradients: VelocityGradientTensorField) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the squared Frobenius norms of the rotation (Omega) and strain-rate (S) tensors.

        Args:
            gradients: The 9-component velocity gradient tensor field.

        Returns:
            A tuple containing (||Omega||^2, ||S||^2) as 3D numpy arrays.

        Raises:
            ValueError: If input gradient fields have inconsistent dimensions.
        """
        self._validate_gradients(gradients)
        
        # Rotation tensor Omega components (antisymmetric part)
        # Omega_ij = 0.5 * (dv_i/dx_j - dv_j/dx_i)
        w_xy = 0.5 * (gradients.dvx_dy - gradients.dvy_dx)
        w_xz = 0.5 * (gradients.dvx_dz - gradients.dvz_dx)
        w_yz = 0.5 * (gradients.dvy_dz - gradients.dvz_dy)
        
        # ||Omega||^2 = sum(Omega_ij^2)
        # Omega is antisymmetric: Omega_ii = 0 and Omega_ij = -Omega_ji
        # ||Omega||^2 = 2 * (w_xy^2 + w_xz^2 + w_yz^2)
        omega_sq = 2.0 * (w_xy**2 + w_xz**2 + w_yz**2)
        
        # Strain-rate tensor S components (symmetric part)
        # S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)
        s_xx = gradients.dvx_dx
        s_yy = gradients.dvy_dy
        s_zz = gradients.dvz_dz
        s_xy = 0.5 * (gradients.dvx_dy + gradients.dvy_dx)
        s_xz = 0.5 * (gradients.dvx_dz + gradients.dvz_dx)
        s_yz = 0.5 * (gradients.dvy_dz + gradients.dvz_dy)
        
        # ||S||^2 = sum(S_ij^2)
        # S is symmetric: S_ij = S_ji
        # ||S||^2 = s_xx^2 + s_yy^2 + s_zz^2 + 2 * (s_xy^2 + s_xz^2 + s_yz^2)
        s_sq = s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)
        
        return omega_sq, s_sq

    def compute_q_field(self, gradients: VelocityGradientTensorField) -> np.ndarray:
        """
        Computes the Q-criterion scalar field (the second invariant).

        This method implements the core backbone logic for vortex identification required by EXP3.

        Args:
            gradients: The 9-component velocity gradient tensor field derived from a flow snapshot.

        Returns:
            3D numpy array representing the Q-criterion scalar field.

        Raises:
            ValueError: If the gradient tensor field is malformed.
        """
        try:
            omega_sq, s_sq = self.calculate_gradient_invariants(gradients)
        except ValueError as e:
            raise ValueError(f"Malformed gradient tensor field: {str(e)}") from e
            
        return 0.5 * (omega_sq - s_sq)

    def extract_invariant_scalar_field(self, q_field: np.ndarray, threshold: float=0.0) -> np.ndarray:
        """
        Filters the Q-field based on a threshold to isolate coherent structures.

        Args:
            q_field: The raw Q-criterion scalar field.
            threshold: The Q-value threshold for vortex identification (default 0.0 per EXP3).

        Returns:
            A binary 3D array where 1.0 indicates a vortex region.
        """
        return (q_field > threshold).astype(np.float64)
