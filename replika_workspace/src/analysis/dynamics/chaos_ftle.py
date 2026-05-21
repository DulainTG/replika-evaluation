import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Dict
from src.integration.state import TrajectoryDataset
from src.physics.tensors import VelocityGradientTensorField
from src.integration.interpolation import TrilinearGridInterpolator
from src.analysis.dynamics.residence_time import TrappingEvent

class TangentLinearIntegrator:
    """
    Solves the tangent linear equation of motion to track perturbation growth.

    Implements the evolution of a infinitesimal displacement vector delta_x 
    along a Lagrangian trajectory by integrating the equation:
    d(delta_x)/dt = J(x(t), t) * delta_x(t)
    where J is the velocity gradient tensor (Jacobian).
    """

    def integrate_perturbation_growth(self, dataset: TrajectoryDataset, gradients: Sequence[VelocityGradientTensorField], initial_epsilon: float=1e-06) -> np.ndarray:
        """
        Integrates the growth of an initially isotropic perturbation for the ensemble.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            gradients: Sequence of velocity gradient fields corresponding to timestamps.
            initial_epsilon: Magnitude of the initial perturbation vector.

        Returns:
            numpy.ndarray: Magnitude of the perturbation vector at each timestamp for each tracer. 
                           Shape: (n_snapshots, n_tracers).

        Raises:
            ValueError: If the gradient sequence length does not match trajectory timestamps.
        """
        n_snapshots = dataset.positions.shape[0]
        n_tracers = dataset.n_tracers
        
        if len(gradients) != n_snapshots:
            raise ValueError(f"Gradient sequence length ({len(gradients)}) must match trajectory snapshots ({n_snapshots}).")

        # Initial perturbation: pick an arbitrary initial direction (x-axis)
        # for all tracers. In a chaotic system, it will eventually align with 
        # the leading Lyapunov vector.
        delta_x = np.zeros((n_tracers, 3))
        delta_x[:, 0] = initial_epsilon 
        
        magnitudes = np.zeros((n_snapshots, n_tracers))
        magnitudes[0, :] = initial_epsilon
        
        interpolator = TrilinearGridInterpolator()
        
        # Grid parameters expected for DS1
        # Origin and domain size according to project defaults
        origin = (-0.5, -0.5, -0.5)
        L = 1.0
        
        def get_sampled_jacobians(snapshot_idx, positions):
            grad_field = gradients[snapshot_idx]
            nx, ny, nz = grad_field.dvx_dx.shape
            spacing = L / (nx - 1)
            
            # Interpolate all 9 components of the Jacobian tensor field
            jac = np.zeros((n_tracers, 3, 3))
            
            # Component-wise interpolation using the existing trilinear routine
            # Row 1
            jac[:, 0, 0] = interpolator.evaluate_on_cartesian_grid(grad_field.dvx_dx, (nx, ny, nz), spacing, origin, positions)
            jac[:, 0, 1] = interpolator.evaluate_on_cartesian_grid(grad_field.dvx_dy, (nx, ny, nz), spacing, origin, positions)
            jac[:, 0, 2] = interpolator.evaluate_on_cartesian_grid(grad_field.dvx_dz, (nx, ny, nz), spacing, origin, positions)
            # Row 2
            jac[:, 1, 0] = interpolator.evaluate_on_cartesian_grid(grad_field.dvy_dx, (nx, ny, nz), spacing, origin, positions)
            jac[:, 1, 1] = interpolator.evaluate_on_cartesian_grid(grad_field.dvy_dy, (nx, ny, nz), spacing, origin, positions)
            jac[:, 1, 2] = interpolator.evaluate_on_cartesian_grid(grad_field.dvy_dz, (nx, ny, nz), spacing, origin, positions)
            # Row 3
            jac[:, 2, 0] = interpolator.evaluate_on_cartesian_grid(grad_field.dvz_dx, (nx, ny, nz), spacing, origin, positions)
            jac[:, 2, 1] = interpolator.evaluate_on_cartesian_grid(grad_field.dvz_dy, (nx, ny, nz), spacing, origin, positions)
            jac[:, 2, 2] = interpolator.evaluate_on_cartesian_grid(grad_field.dvz_dz, (nx, ny, nz), spacing, origin, positions)
            
            return jac

        # Initialize previous Jacobian at t=0
        J_prev = get_sampled_jacobians(0, dataset.positions[0])
        
        # Integration loop over snapshots
        for i in range(1, n_snapshots):
            dt = dataset.timestamps[i] - dataset.timestamps[i-1]
            if dt <= 0:
                magnitudes[i, :] = magnitudes[i-1, :]
                continue
                
            J_curr = get_sampled_jacobians(i, dataset.positions[i])
            
            # Predict step (Euler)
            k1 = np.einsum('nij,nj->ni', J_prev, delta_x)
            delta_x_pred = delta_x + k1 * dt
            
            # Correct step (Heun's method)
            k2 = np.einsum('nij,nj->ni', J_curr, delta_x_pred)
            delta_x = delta_x + 0.5 * dt * (k1 + k2)
            
            # Record magnitude
            magnitudes[i, :] = np.linalg.norm(delta_x, axis=1)
            
            # Prepare for next iteration
            J_prev = J_curr
            
        return magnitudes

class FTLECalculator:
    """
    Calculates Finite-Time Lyapunov Exponents (FTLE) from perturbation growth data.

    The backbone calculation uses the formula:
    lambda(T) = (1 / T) * ln(||delta_x(T)|| / ||delta_x(0)||)
    where T is the integration time.
    """

    def calculate_long_time_ftle(self, magnitudes: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Computes the FTLE values integrated over the full trajectory duration.

        This supports the EXP5 goal of evaluating if long-time FTLE can distinguish 
        between trapped and free dynamics.

        Args:
            magnitudes: Perturbation magnitudes (snapshots, tracers).
            timestamps: Observation times (snapshots,).

        Returns:
            numpy.ndarray: Final FTLE value for each tracer.
        """
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return np.zeros(magnitudes.shape[1])
        
        # Final magnitude ratio
        growth_ratio = magnitudes[-1, :] / magnitudes[0, :]
        
        # lambda = (1/T) * ln(growth_ratio)
        return (1.0 / duration) * np.log(growth_ratio)

    def measure_instantaneous_exit_ftle(self, magnitudes: np.ndarray, timestamps: np.ndarray, exit_events: List[TrappingEvent]) -> np.ndarray:
        """
        Measures the FTLE at the exact moment of vortex exit.

        Identifies 'vortex exit' events (transition from Q > 0 to Q < 0) and 
        records the instantaneous FTLE integrated since the start of the trapping.

        Args:
            magnitudes: Perturbation magnitudes.
            timestamps: Observation times.
            exit_events: List of identified trapping exit triggers.

        Returns:
            numpy.ndarray: FTLE values at exit for each provided event.
        """
        exit_ftles = []
        for event in exit_events:
            # Locate snapshot indices closest to start and end times of the event
            idx_start = np.argmin(np.abs(timestamps - event.start_time))
            idx_end = np.argmin(np.abs(timestamps - event.end_time))
            
            duration = event.duration
            if duration <= 0:
                exit_ftles.append(0.0)
                continue
            
            mag_at_start = magnitudes[idx_start, event.tracer_index]
            mag_at_end = magnitudes[idx_end, event.tracer_index]
            
            # If magnitude at start is invalid, record 0
            if mag_at_start <= 0:
                exit_ftles.append(0.0)
            else:
                # Calculate FTLE for the trapping interval
                ftle = np.log(mag_at_end / mag_at_start) / duration
                exit_ftles.append(ftle)
                
        return np.array(exit_ftles)
@dataclass(frozen=True)
class ChaosPearsonResult:
    """Container for correlation metrics between chaos and residence dynamics for EXP5."""
    mean_ftle: float
    std_ftle: float
    pearson_r: float
    n_samples: int
class ChaosCorrelationAnalyzer:
    """
    Performs statistical correlation analysis between FTLE and residence characteristics.
    """

    def calculate_pearson_coefficient(self, ftle_values: np.ndarray, residence_metrics: np.ndarray) -> float:
        """
        Calculates the standard Pearson correlation r between two Lagrangian signals.

        Args:
            ftle_values: Array of FTLE measurements.
            residence_metrics: Corresponding residence fractions or durations.

        Returns:
            float: Pearson r coefficient.
        """
        if len(ftle_values) < 2 or len(residence_metrics) < 2:
            return 0.0
        
        # Check for constant arrays (zero variance)
        if np.std(ftle_values) == 0 or np.std(residence_metrics) == 0:
            return 0.0
            
        corr_matrix = np.corrcoef(ftle_values, residence_metrics)
        # corr_matrix is [[1, r], [r, 1]]
        r = corr_matrix[0, 1]
        
        return float(r) if not np.isnan(r) else 0.0

    def analyze_chaos_cohort_stats(self, ftle_values: np.ndarray, residence_fractions: np.ndarray, cohort_masks: Dict[str, np.ndarray]) -> Dict[str, ChaosPearsonResult]:
        """
        Computes FTLE statistics and correlations for specific subsets of the ensemble.

        Supports the EXP5 requirement to compare 'Trapped' and 'Free' cohorts.

        Args:
            ftle_values: Long-time FTLE for tracers.
            residence_fractions: Fraction of time spent in Q > 0 regions.
            cohort_masks: Dictionary mapping cohort names ('Trapped', 'Free') 
                         to boolean arrays.

        Returns:
            Dict[str, ChaosPearsonResult]: Statistics per cohort including mean and Pearson r.
        """
        stats = {}
        for cohort_name, mask in cohort_masks.items():
            # Apply cohort selection (works with both boolean masks and index arrays)
            c_ftle = ftle_values[mask]
            c_res = residence_fractions[mask]
            
            if len(c_ftle) == 0:
                stats[cohort_name] = ChaosPearsonResult(
                    mean_ftle=0.0,
                    std_ftle=0.0,
                    pearson_r=0.0,
                    n_samples=0
                )
                continue
                
            mean_ftle = np.mean(c_ftle)
            std_ftle = np.std(c_ftle)
            pearson_r = self.calculate_pearson_coefficient(c_ftle, c_res)
            
            stats[cohort_name] = ChaosPearsonResult(
                mean_ftle=float(mean_ftle),
                std_ftle=float(std_ftle),
                pearson_r=float(pearson_r),
                n_samples=len(c_ftle)
            )
            
        return stats
