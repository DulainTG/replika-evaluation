from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
from src.integration.state import TrajectoryDataset

@dataclass(frozen=True)
class AnisotropicMSDResults:
    """
    Container for partitioned anisotropic Mean Squared Displacement components.
    
    Attributes:
        lag_times: Array of time intervals (t) from DS2.
        msd_parallel: Parallel component MSD||(t).
        msd_perpendicular: Perpendicular component MSD_perp(t).
        anisotropy_ratio: The ratio lambda(t) = MSD||(t) / MSD_perp(t).
    """
    lag_times: np.ndarray
    msd_parallel: np.ndarray
    msd_perpendicular: np.ndarray
    anisotropy_ratio: np.ndarray

class AnisotropicMSDCalculator:
    """
    Calculator for partitioned Mean Squared Displacement (MSD) relative to 
    the large-scale velocity field (VLS) as defined in Experiment EXP2.
    """

    def _unwrap_trajectories(self, positions: np.ndarray, domain_size: float = 1.0) -> np.ndarray:
        """
        Unwraps periodic trajectories to compute continuous displacements.
        
        This assumes that the particle moves no more than domain_size/2 between
        consecutive snapshots, allowing correction of periodic boundary jumps.
        
        Args:
            positions: Wrapped positions of shape (N_snapshots, N_tracers, 3).
            domain_size: Edge length of the periodic cubic domain.
            
        Returns:
            np.ndarray: Unwrapped positions of shape (N_snapshots, N_tracers, 3).
        """
        if positions.shape[0] < 2:
            return positions.copy()
            
        unwrapped = np.zeros_like(positions)
        unwrapped[0] = positions[0]
        
        # Compute incremental displacements
        diffs = np.diff(positions, axis=0)
        
        # Apply periodic correction: shortest-path assumption
        # Corrected diff = diff - L * round(diff/L)
        diffs = diffs - domain_size * np.round(diffs / domain_size)
        
        unwrapped[1:] = positions[0] + np.cumsum(diffs, axis=0)
        return unwrapped

    def calculate_parallel_component(self, displacements: np.ndarray, vls_hat: np.ndarray) -> np.ndarray:
        """
        Calculates the parallel component of the Mean Squared Displacement.
        
        Equation: MSD||(t) = <(delta_x(t) . V_LS_hat)^2>
        
        Args:
            displacements: Delta positions delta_x(t) of shape (N_snapshots, N_tracers, 3).
            vls_hat: Local unit vector field V_LS_hat of shape (N_snapshots, N_tracers, 3).
            
        Returns:
            Mean squared displacement parallel to the local large-scale flow at each lag time.
        """
        # dot product for each (snapshot, tracer)
        # Using broadcasting if shapes allow, but here they should match (N_snapshots, N_tracers, 3)
        dot_product = np.sum(displacements * vls_hat, axis=-1)
        parallel_sq = dot_product**2
        
        # Average over the tracer ensemble (axis 1)
        return np.mean(parallel_sq, axis=1)

    def calculate_perpendicular_component(self, displacements: np.ndarray, vls_hat: np.ndarray) -> np.ndarray:
        """
        Calculates the perpendicular component of the Mean Squared Displacement.
        
        Equation: MSD_perp(t) = <|delta_x(t) - (delta_x(t) . V_LS_hat)V_LS_hat|^2>
        
        Args:
            displacements: Delta positions delta_x(t) of shape (N_snapshots, N_tracers, 3).
            vls_hat: Local unit vector field V_LS_hat of shape (N_snapshots, N_tracers, 3).
            
        Returns:
            Mean squared displacement perpendicular to the local large-scale flow at each lag time.
        """
        # Parallel projection magnitude
        dot_product = np.sum(displacements * vls_hat, axis=-1)
        
        # Total squared displacement magnitude
        disp_sq = np.sum(displacements**2, axis=-1)
        
        # In 3D, perp^2 = total^2 - parallel^2, since vls_hat is a unit vector
        perp_sq = disp_sq - (dot_product**2)
        
        # Average over the tracer ensemble
        return np.mean(perp_sq, axis=1)

    def compute_anisotropy_stats(self, dataset: TrajectoryDataset, vls_hat_field: np.ndarray) -> AnisotropicMSDResults:
        """
        Aggregates parallel and perpendicular MSD over the tracer ensemble for all lag times.
        
        Supports Experiment EXP2 requirement to measure the ratio lambda(t) and compare 
        against the target value 0.52.
        
        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            vls_hat_field: Unit vectors for the filtered large-scale velocity field 
                at every tracer position and time.
                
        Returns:
            Aggregated anisotropic results including the ratio lambda(t).
        """
        # 1. Prepare continuous trajectories by unwrapping periodic jumps.
        # domain_size_L = 1.0 is the standard for this experiment.
        unwrapped_positions = self._unwrap_trajectories(dataset.positions, domain_size=1.0)
        
        # 2. Compute displacements relative to the start (t=0).
        # Shape: (N_snapshots, N_tracers, 3)
        displacements = unwrapped_positions - unwrapped_positions[0:1]
        
        # 3. Compute MSD parallel and perpendicular components.
        # We project each tracer's achievement delta_x(t) onto its current local V_LS_hat.
        msd_parallel = self.calculate_parallel_component(displacements, vls_hat_field)
        msd_perpendicular = self.calculate_perpendicular_component(displacements, vls_hat_field)
        
        # 4. Calculate anisotropy ratio lambda(t) = MSD||(t) / MSD_perp(t).
        # We handle t=0 (0/0) gracefully.
        with np.errstate(divide='ignore', invalid='ignore'):
            anisotropy_ratio = msd_parallel / msd_perpendicular
            anisotropy_ratio = np.nan_to_num(anisotropy_ratio, nan=0.0)
            
        # 5. Determine lag times from dataset timestamps.
        lag_times = dataset.timestamps - dataset.timestamps[0]
        
        return AnisotropicMSDResults(
            lag_times=lag_times,
            msd_parallel=msd_parallel,
            msd_perpendicular=msd_perpendicular,
            anisotropy_ratio=anisotropy_ratio
        )
class DispersionRegimeAnalyzer:
    """
    Analyzer for identifying Lagrangian transport regimes based on logarithmic 
    scaling of the Mean Squared Displacement (MSD).
    """

    def calculate_local_slope(self, timestamps: np.ndarray, msd_values: np.ndarray) -> np.ndarray:
        """
        Calculates the local logarithmic slope alpha(t) to distinguish 
        between ballistic, superdiffusive, and diffusive regimes (EXP2).
        
        Equation: alpha(t) = d(log MSD) / d(log t)
        
        Characterization:
        - Ballistic regime: alpha(t) approx 2.0
        - Superdiffusive regime: 1.0 < alpha(t) < 2.0
        - Diffusive regime: alpha(t) approx 1.0
        
        Args:
            timestamps: Lagrangian time lags (t).
            msd_values: Calculated mean squared displacements MSD(t).
            
        Returns:
            Array of local slopes alpha(t) corresponding to the time lags.
            
        Raises:
            RuntimeError: If numerical differentiation fails or data is insufficient.
        """
        # Filter for strictly positive values to allow log transformation
        # We need at least two points to compute a gradient
        mask = (timestamps > 0) & (msd_values > 0)
        
        if np.count_nonzero(mask) < 2:
            raise RuntimeError("Insufficient data for numerical differentiation.")
            
        log_t = np.log(timestamps[mask])
        log_msd = np.log(msd_values[mask])
        
        # np.gradient computes central differences for interior points 
        # and one-sided differences for the boundaries.
        slopes = np.gradient(log_msd, log_t)
        
        # We map the results back to the original timestamps shape, using NaN for non-positive values
        alpha = np.full_like(timestamps, np.nan, dtype=float)
        alpha[mask] = slopes
        return alpha

    def identify_regime_windows(self, alpha_series: np.ndarray, timestamps: np.ndarray) -> Dict[str, Sequence[Tuple[float, float]]]:
        """
        Segments the time series into physical transport regimes based on alpha(t) thresholds.
        
        Args:
            alpha_series: Local logarithmic slopes alpha(t).
            timestamps: Corresponding time lags.
            
        Returns:
            Dictionary mapping regime names ('ballistic', 'superdiffusive', 'diffusive') 
            to a sequence of (t_start, t_end) windows where that regime persists.
        """
        # Filter out NaNs to work with valid segments
        valid_mask = ~np.isnan(alpha_series)
        alpha = alpha_series[valid_mask]
        t = timestamps[valid_mask]
        
        if len(t) == 0:
            return {
                "ballistic": [],
                "superdiffusive": [],
                "diffusive": []
            }

        # Define regimes based on logarithmic slope alpha(t)
        # These thresholds are chosen to capture the regimes defined in EXP2.
        # Ballistic: alpha ~ 2.0
        # Superdiffusive: 1.0 < alpha < 2.0
        # Diffusive: alpha ~ 1.0
        regime_masks = {
            "ballistic": alpha >= 1.8,
            "superdiffusive": (alpha > 1.2) & (alpha < 1.8),
            "diffusive": alpha <= 1.2
        }
        
        results = {}
        for regime_name, mask in regime_masks.items():
            results[regime_name] = self._find_contiguous_windows(t, mask)
            
        return results

    def _find_contiguous_windows(self, times: np.ndarray, mask: np.ndarray) -> Sequence[Tuple[float, float]]:
        """Helper to find continuous time intervals where a condition is met."""
        windows = []
        if not np.any(mask):
            return windows
            
        # Standard approach to find contiguous blocks of True values
        # Convert mask to int [0, 1] and find where it transitions
        m = mask.astype(int)
        diff = np.diff(m, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        for s, e in zip(starts, ends):
            # Each window is (t_start, t_end)
            windows.append((float(times[s]), float(times[e])))
            
        return windows
