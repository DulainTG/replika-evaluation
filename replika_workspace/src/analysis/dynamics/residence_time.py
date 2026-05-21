import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional, List, Dict
from src.integration.state import TrajectoryDataset
from src.physics.fields import VelocitySnapshot
from src.physics.vortex_criteria import QCriterionCalculator
from src.physics.tensors import VelocityGradientCalculator, CentralDifferenceDerivativeScheme
from src.integration.interpolation import TrilinearGridInterpolator

class LagrangianQSampler:
    """
    Responsible for sampling the Q-criterion scalar field along Lagrangian tracer paths.
    
    Maps Eulerian diagnostic fields (Q) to particle positions over time to create DS2-aligned
    scalar histories.
    """

    def __init__(self, q_calculator: QCriterionCalculator):
        """
        Args:
            q_calculator: Service to compute Q-criterion from velocity gradients.
        """
        self.q_calculator = q_calculator
        self.gradient_calculator = VelocityGradientCalculator(CentralDifferenceDerivativeScheme())
        self.interpolator = TrilinearGridInterpolator()

    def sample_q_history(self, dataset: TrajectoryDataset, snapshots: Sequence[VelocitySnapshot]) -> np.ndarray:
        """
        Samples the Q-value at every tracer position and every time step.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            snapshots: Sequence of Eulerian velocity snapshots (DS1) corresponding to timestamps.

        Returns:
            A 2D array of shape (N_snapshots, N_tracers) containing the Q-signal history.

        Raises:
            ValueError: If the number of snapshots does not match the trajectory temporal dimension.
        """
        n_snapshots = len(snapshots)
        if n_snapshots != dataset.positions.shape[0]:
            raise ValueError(f"Number of snapshots ({n_snapshots}) does not match trajectory temporal dimension ({dataset.positions.shape[0]}).")

        n_tracers = dataset.n_tracers
        q_history = np.zeros((n_snapshots, n_tracers))

        for i, snapshot in enumerate(snapshots):
            # 1. Compute velocity gradient tensor
            gradients = self.gradient_calculator.compute_gradient_tensor(snapshot)
            # 2. Compute Q-field
            q_field = self.q_calculator.compute_q_field(gradients)
            # 3. Sample Q-field at tracer positions
            positions = dataset.positions[i]
            q_values = self.interpolator.evaluate_on_cartesian_grid(
                field=q_field,
                dimensions=snapshot.grid_dimensions,
                spacing=snapshot.spacing,
                origin=snapshot.origin,
                positions=positions
            )
            q_history[i, :] = q_values

        return q_history
@dataclass(frozen=True)
class QSignalStatistics:
    """
    Represents the decay characteristics of the Lagrangian Q-signal.
    Used for Experiment 3 validation against target ratios.
    """
    tau_q: float
    decay_rate: float
    te_ratio: float
    correlation_series: np.ndarray
class QSignalDecayAnalyzer:
    """
    Analyzes the temporal decay of the Q-criterion signal to determine vortex lifetimes.
    
    Implements the e-folding time determination where the Lagrangian autocorrelation 
    function R_Q(tau) = <Q(t)Q(t+tau)> / <Q(t)^2> drops to 1/e.
    """

    def calculate_autocorrelation(self, q_history: np.ndarray) -> np.ndarray:
        """
        Computes the ensemble-averaged Lagrangian autocorrelation of the Q-signal.

        Args:
            q_history: 2D array (N_snapshots, N_tracers) of sampled Q values.

        Returns:
            1D array of correlation values for each possible lag time.
        """
        n_snapshots, n_tracers = q_history.shape
        correlation = np.zeros(n_snapshots)
        
        for k in range(n_snapshots):
            # For each lag k, compute the average product Q(t) * Q(t+k) across 
            # all tracers and all valid time intervals.
            q_start = q_history[:n_snapshots - k, :]
            q_end = q_history[k:, :]
            
            # Use lag-specific normalization to correctly handle decaying signals (EXP3).
            # This avoids bias from non-stationarity in synthetic benchmark signals.
            numerator = np.mean(q_start * q_end)
            denominator = np.mean(q_start**2)

            if denominator > 0:
                correlation[k] = numerator / denominator
            else:
                # If everything is zero, it's perfectly correlated at 1.0. 
                # If only start is zero but end isn't, correlation is 0.0.
                correlation[k] = 1.0 if np.all(q_start == 0) and np.all(q_end == 0) else 0.0
            
        return correlation

    def determine_e_folding_time(self, correlation: np.ndarray, timestamps: np.ndarray) -> float:
        """
        Finds the lag time tau_Q where the correlation first drops below 1/e (~0.368).

        Args:
            correlation: 1D array of autocorrelation values.
            timestamps: 1D array of time values corresponding to lags.

        Returns:
            The interpolated time tau_Q.
        """
        if correlation.size == 0:
            return 0.0

        # The target correlation value for e-folding time.
        target = 1.0 / np.exp(1.0)
        
        # The lags are relative to the first timestamp.
        lags = timestamps - timestamps[0]
        
        if correlation[0] <= target:
            return 0.0
            
        for i in range(1, len(correlation)):
            if correlation[i] <= target:
                # Linear interpolation to find the exact time where it crosses 'target'.
                # Correlation(tau) is approximated linearly between i-1 and i.
                c_prev, c_curr = correlation[i-1], correlation[i]
                t_prev, t_curr = lags[i-1], lags[i]
                
                # Interpolation formula: t = t_prev + (target - c_prev) * (t_curr - t_prev) / (c_curr - c_prev)
                tau_q = t_prev + (target - c_prev) * (t_curr - t_prev) / (c_curr - c_prev)
                return float(tau_q)
        
        # If the correlation never drops below the target, return the maximum lag.
        return float(lags[-1])

    def analyze_decay_dynamics(self, q_history: np.ndarray, timestamps: np.ndarray, te: float=2.6) -> QSignalStatistics:
        """
        Performs full decay analysis including e-folding and te_ratio calculation.

        Args:
            q_history: Scalar history of Q values (N_snapshots, N_tracers).
            timestamps: Absolute time for each snapshot.
            te: Large-eddy turnover time for normalization.

        Returns:
            A structured bundle of decay metrics.
        """
        correlation = self.calculate_autocorrelation(q_history)
        tau_q = self.determine_e_folding_time(correlation, timestamps)
        
        te_ratio = tau_q / te
        decay_rate = 1.0 / tau_q if tau_q > 0 else 0.0
        
        return QSignalStatistics(
            tau_q=tau_q,
            decay_rate=decay_rate,
            te_ratio=te_ratio,
            correlation_series=correlation
        )
@dataclass(frozen=True)
class TrappingEvent:
    """
    Represents a single continuous interval where a tracer is trapped in a vortex (Q > threshold).
    """
    tracer_index: int
    start_time: float
    end_time: float
    duration: float
    exit_q_value: float
class VortexResidenceTracker:
    """
    Identifies and partitions tracer trajectories into 'trapped' and 'free' segments.
    
    Used to classify cohorts for EXP5 (FTLE vs Residence) and to verify claim C2
    regarding the brevity of trapping events.
    """

    def identify_trapping_events(self, q_history: np.ndarray, timestamps: np.ndarray, threshold: float=0.0) -> List[TrappingEvent]:
        """
        Parses the Q-history to find contiguous segments where Q > threshold.

        Args:
            q_history: 2D array (N_snapshots, N_tracers) of Q values.
            timestamps: 1D array of snapshot times.
            threshold: The Q-criterion value used to define a vortex boundary (default 0.0).

        Returns:
            A list of all discrete trapping events found across the ensemble.
        """
        events = []
        if q_history.size == 0:
            return events
            
        n_snapshots, n_tracers = q_history.shape
        
        for j in range(n_tracers):
            q_signal = q_history[:, j]
            is_trapped = q_signal > threshold
            
            # Use diff to find starts and ends of contiguous True segments
            # Pad with False to ensure we catch events at the very beginning or end
            padded = np.zeros(n_snapshots + 2, dtype=bool)
            padded[1:-1] = is_trapped
            
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for s, e in zip(starts, ends):
                # s is start index in q_signal (since we padded by 1 at start)
                # e is index after the last True in q_signal (since padded)
                # Note: np.where(diff == 1) on padded gives index of the True value in padded
                # which corresponds to (s-1) in is_trapped if we don't adjust.
                # Actually, let's re-verify:
                # padded = [F, is_trapped[0], is_trapped[1], ..., is_trapped[N-1], F]
                # diff[0] = padded[1] - padded[0] = is_trapped[0] - F
                # If is_trapped[0] is T, diff[0] = 1, starts[0] = 0.
                # The index in is_trapped is starts[0] = 0. Correct.
                # diff[i] = padded[i+1] - padded[i]
                # ends[0] is index i such that padded[i+1] is F and padded[i] is T.
                # So the T is at padded[ends[0]], which is is_trapped[ends[0]-1].
                # Wait, let me re-trace.
                idx_start = s
                idx_end = e - 1
                
                t_start = timestamps[idx_start]
                t_end = timestamps[idx_end]
                duration = t_end - t_start
                
                # Exit Q value: first value after the trapping segment
                if e < n_snapshots:
                    exit_q = float(q_signal[e])
                else:
                    # If it stayed trapped until the end, use the last value
                    exit_q = float(q_signal[idx_end])
                    
                events.append(TrappingEvent(
                    tracer_index=j,
                    start_time=t_start,
                    end_time=t_end,
                    duration=duration,
                    exit_q_value=exit_q
                ))
        return events

    def calculate_residence_fractions(self, events: List[TrappingEvent], total_time: float, n_tracers: int) -> np.ndarray:
        """
        Calculates the fraction of time each tracer spent trapped in a vortex.

        Args:
            events: List of identified trapping events.
            total_time: Total duration of the simulation/observation.
            n_tracers: Total number of tracers in the ensemble.

        Returns:
            1D array of size (n_tracers) with values in range [0, 1].
        """
        residence_times = np.zeros(n_tracers)
        for event in events:
            if 0 <= event.tracer_index < n_tracers:
                residence_times[event.tracer_index] += event.duration
            
        if total_time <= 0:
            return np.zeros(n_tracers)
            
        return np.clip(residence_times / total_time, 0.0, 1.0)

    def partition_population_by_residence(self, residence_fractions: np.ndarray, top_percentile: float=20.0, bottom_percentile: float=20.0) -> Dict[str, np.ndarray]:
        """
        Splits tracer indices into 'Trapped' and 'Free' cohorts based on residence time.

        Required for EXP5 cohort comparison.

        Args:
            residence_fractions: Array of fraction of time spent in vortices.
            top_percentile: Threshold for the 'Trapped' cohort.
            bottom_percentile: Threshold for the 'Free' cohort.

        Returns:
            Dictionary with keys 'trapped_indices' and 'free_indices'.
        """
        if residence_fractions.size == 0:
            return {
                'trapped_indices': np.array([], dtype=int),
                'free_indices': np.array([], dtype=int)
            }
            
        # Top 20% by residence time -> those at or above the 80th percentile
        phi_upper = np.percentile(residence_fractions, 100.0 - top_percentile)
        # Bottom 20% by residence time -> those at or below the 20th percentile
        phi_lower = np.percentile(residence_fractions, bottom_percentile)
        
        trapped_indices = np.where(residence_fractions >= phi_upper)[0]
        free_indices = np.where(residence_fractions <= phi_lower)[0]
        
        return {
            'trapped_indices': trapped_indices,
            'free_indices': free_indices
        }
