import numpy as np

class PeriodicBoundaryMapper:
    """
    Handles coordinate transformation and wrapping for periodic simulation domains.

    This component provides the 'periodic domain representation' for the Lagrangian
    tracer simulation defined in EXP1 (Tracer Trajectory Generation). It ensures
    that tracer positions (x, y, z) are consistently mapped into the principal
    domain [0, L) using 'modulo domain mapping' logic to satisfy periodic 
    boundary conditions.

    As specified in the experiment procedure, coordinates are updated after each
    RK4 integration step using the modulo operation:
    
    x_wrapped = x mod L

    where L is the domain size (L=1.0 for the DS1/DS2 datasets).
    """

    def wrap_coordinates(self, positions: np.ndarray, domain_size: float=1.0) -> np.ndarray:
        """
        Performs 'periodic domain wrap' and 'modulo domain coordinate wrapping'.

        Args:
            positions: Array of shape (..., 3) containing tracer coordinates.
            domain_size: The edge length L of the cubic periodic domain. 
                         Defaults to 1.0 as per experiment requirements.

        Returns:
            np.ndarray: Coordinates wrapped into the range [0, domain_size).

        Raises:
            ValueError: If domain_size is non-positive.
        """
        if domain_size <= 0:
            raise ValueError(f"domain_size must be positive, got {domain_size}")
        return self.apply_periodic_modulo_wrap(positions, domain_size)

    def map_to_unit_domain(self, positions: np.ndarray) -> np.ndarray:
        """
        Implements 'periodic wrap modulo 1' for a standardized unit domain.

        A specialized form of 'modulo coordinate mapping' where L is fixed to 1.0.
        This directly supports the validation of DS2 coordinate range checks (0-1).

        Args:
            positions: Array of spatial positions to be mapped.

        Returns:
            np.ndarray: Coordinates mapped to the unit domain [0, 1.0).
        """
        return self.wrap_coordinates(positions, domain_size=1.0)

    def apply_periodic_modulo_wrap(self, values: np.ndarray, modulus: float) -> np.ndarray:
        """
        General interface for 'periodic modulo wrap' operations on arbitrary tensors.

        Used for ensuring any Lagrangian state property adheres to the domain symmetry.

        Args:
            values: Input array (coordinates, periodic indices, etc.).
            modulus: The periodicity interval L.

        Returns:
            np.ndarray: Wrapped values within [0, modulus).
        """
        if modulus <= 0:
            raise ValueError(f"modulus must be positive, got {modulus}")
        # numpy.mod (equivalent to % operator) handles the sign such that
        # the result has the same sign as the divisor (modulus).
        # For positive modulus, the result is in [0, modulus).
        return np.mod(values, modulus)
