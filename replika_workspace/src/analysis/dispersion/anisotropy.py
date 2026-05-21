import numpy as np
from typing import Tuple, Dict
from src.analysis.dispersion.msd import AnisotropicMSDResults

class AnisotropicProjectionEngine:
    """
    Handles geometric decomposition of tracer displacements relative to the large-scale velocity field.

    Coordinates the projection of 3D displacements into parallel (longitudinal) and 
    perpendicular (transverse) components based on the unit vector of the large-scale 
    filtered velocity field V_LS_hat.
    """

    def decompose_displacement_vls(self, displacements: np.ndarray, vls_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects a set of displacement vectors into components parallel and perpendicular 
        to the provided large-scale unit vectors.

        Args:
            displacements: Array of shape (N, 3) representing tracer displacement vectors delta_x(t).
            vls_hat: Array of shape (N, 3) representing large-scale unit vectors V_LS_hat.

        Returns:
            A tuple (parallel_comp, perp_comp) where:
                parallel_comp: (N,) scalar projection onto V_LS_hat: (delta_x . V_LS_hat).
                perp_comp: (N, 3) vector component perpendicular to V_LS_hat: delta_x - (delta_x . V_LS_hat) * V_LS_hat.

        Raises:
            ValueError: If the shapes of displacements and vls_hat do not match.
        """
        if displacements.shape != vls_hat.shape:
            raise ValueError(f"Shape mismatch: displacements {displacements.shape} and vls_hat {vls_hat.shape}")
        
        # Parallel scalar projection: dot product of delta_x and V_LS_hat
        parallel_comp = np.sum(displacements * vls_hat, axis=1)
        
        # Parallel vector component
        parallel_vec = parallel_comp[:, np.newaxis] * vls_hat
        
        # Perpendicular vector component
        perp_comp = displacements - parallel_vec
        
        return parallel_comp, perp_comp


class AnisotropicComponentCalculator:
    """
    Calculates ensemble-averaged Mean Squared Displacement (MSD) for specific anisotropic components.

    Specifically implements the longitudinal (parallel) and transverse (perpendicular) MSDs 
    required for EXP2 to analyze 3D solenoidal turbulence anisotropy.
    """

    def calculate_longitudinal_msd(self, parallel_projections: np.ndarray) -> float:
        """
        Calculates the ensemble-averaged parallel MSD for a single lag time t.
        Equation: MSD_||(t) = <(delta_x(t) . V_LS_hat)^2>.

        Args:
            parallel_projections: Array of scalar projections (delta_x . V_LS_hat) for the ensemble.

        Returns:
            The mean squared longitudinal displacement.
        """
        if parallel_projections.size == 0:
            return 0.0
        return float(np.mean(parallel_projections**2))

    def calculate_transverse_msd(self, perpendicular_vectors: np.ndarray) -> float:
        """
        Calculates the ensemble-averaged perpendicular MSD for a single lag time t.
        Equation: MSD_perp(t) = <|delta_x(t) - (delta_x(t) . V_LS_hat)V_LS_hat|^2>.

        Args:
            perpendicular_vectors: Array of vectors (N, 3) representing components perpendicular to V_LS_hat.

        Returns:
            The mean squared transverse displacement.
        """
        if perpendicular_vectors.size == 0:
            return 0.0
        # Squared magnitude of each perpendicular vector, then mean over ensemble
        sq_magnitudes = np.sum(perpendicular_vectors**2, axis=1)
        return float(np.mean(sq_magnitudes))


class AnisotropyTemporalTracker:
    """
    Tracks and calculates the temporal evolution of anisotropic transport metrics.

    Primary responsibility is the calculation of lambda(t) and the monitoring of 
    longitudinal/transverse transport over the full range of lag times defined in EXP2.
    """

    def calculate_lambda_series(self, msd_parallel: np.ndarray, msd_perp: np.ndarray) -> np.ndarray:
        """
        Computes the anisotropy ratio series lambda(t) over all time lags.
        Equation: lambda(t) = MSD_||(t) / MSD_perp(t).

        Args:
            msd_parallel: Time series array of parallel Mean Squared Displacements.
            msd_perp: Time series array of perpendicular Mean Squared Displacements.

        Returns:
            Array for lambda(t) time series.

        Raises:
            ZeroDivisionError: If any msd_perp value is zero.
        """
        if np.any(msd_perp == 0):
            # To handle t=0 where MSD is naturally zero, we might want to be careful.
            # But the requirement says to raise ZeroDivisionError if ANY msd_perp is zero.
            raise ZeroDivisionError("MSD perpendicular component contains zero values, cannot compute lambda(t).")
            
        return msd_parallel / msd_perp

    def track_transport_components(self, results: AnisotropicMSDResults) -> Dict[str, np.ndarray]:
        """
        Extracts and maps parallel and perpendicular transport series for diagnostic reporting.

        Args:
            results: Aggregated anisotropic MSD data including parallel/perp components.

        Returns:
            A dictionary containing 'longitudinal_transport' and 'transverse_transport' series.
        """
        return {
            "longitudinal_transport": results.msd_parallel,
            "transverse_transport": results.msd_perpendicular
        }

    def evaluate_evolution_against_target(self, lambda_series: np.ndarray, target: float=0.52) -> float:
        """
        Calculates the mean deviation of the observed lambda(t) from the paper's target value.

        Args:
            lambda_series: Observed temporal ratio tracking data.
            target: The theoretical/experimental target ratio (default 0.52 per EXP2).

        Returns:
            Mean absolute error against the target across the stable diffusive regime.
        """
        if lambda_series.size == 0:
            return 0.0
        return float(np.mean(np.abs(lambda_series - target)))
