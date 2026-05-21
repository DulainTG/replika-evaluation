import numpy as np
import math
from typing import Dict
from src.integration.state import TrajectoryDataset

class EnsembleAveragingService:
    """
    Orchestrates the aggregation and averaging of tracer data across the ensemble.

    Handles temporal averaging loops required for stable statistics in EXP2 and EXP4.
    """

    def calculate_ensemble_mean(self, values: np.ndarray) -> np.ndarray:
        """
        Averages tracer properties across the entire ensemble at each time step.

        Args:
            values: Array of shape (n_snapshots, n_tracers) or (n_snapshots, n_tracers, Dim).

        Returns:
            np.ndarray: Ensemble-averaged values per snapshot.
        """
        return np.mean(values, axis=1)

    def aggregate_by_time_lag(self, dataset: TrajectoryDataset) -> Dict[float, np.ndarray]:
        """
        Groups tracer displacements by their elapsed time lag from the start of the trajectory.

        Args:
            dataset: The generated trajectory dataset (DS2).

        Returns:
            Dict mapping each lag time to an array of all tracer property samples at that lag.
        """
        pos = dataset.positions
        n_snapshots, n_tracers, _ = pos.shape
        
        if n_snapshots == 0:
            return {}

        # 1. Calculate unwrapped displacements from start (t0) using consecutive image differences
        # This assumes tracers do not cross more than half the domain [0, 1.0) between snapshots.
        diffs = np.diff(pos, axis=0)
        # Apply periodic boundary correction (minimum image convention) assuming L=1.0
        diffs = diffs - np.round(diffs)
        
        unwrapped_disp = np.zeros_like(pos)
        unwrapped_disp[1:] = np.cumsum(diffs, axis=0)
        
        # 2. Group by lag time
        lags = dataset.timestamps - dataset.timestamps[0]
        
        return {float(lag): unwrapped_disp[i] for i, lag in enumerate(lags)}
class VelocityAutocorrelationCalculator:
    """
    Calculates the Lagrangian velocity autocorrelation series.

    Required for EXP2 to analyze tracer memory.
    The Velocity Autocorrelation Function (VACF) is defined as:
    VACF(t) = < v(t_0) . v(t_0 + t) >
    """

    def calculate_vacf_series(self, dataset: TrajectoryDataset) -> np.ndarray:
        """
        Generates the full VACF sequence for the provided trajectories.

        Args:
            dataset: Dataset containing tracer velocities and timestamps.

        Returns:
            np.ndarray: Array containing the VACF values for every available lag time.
        """
        v = dataset.velocities
        n_snapshots, _, _ = v.shape
        
        if n_snapshots == 0:
            return np.array([])
            
        vacf = np.zeros(n_snapshots)
        
        for k in range(n_snapshots):
            # Calculate VACF for lag index k by averaging over all possible start times t_i
            # and all tracers in the ensemble.
            v_start = v[:n_snapshots - k]
            v_end = v[k:]
            
            # Lagrangian VACF(k) = < v(t_i) . v(t_i + k) >
            # Average over tracers (axis 1) and time intervals (axis 0)
            dot_product = np.sum(v_start * v_end, axis=-1)
            vacf[k] = np.mean(dot_product)
            
        return vacf
class HillTailEstimator:
    """
    Implementation of the Hill estimator for tail index and power law identification.

    Used in EXP4 (Displacement PDF Evolution) to verify the absence of heavy tails
    in Lagrangian displacements.

    The Hill estimator xi is defined for the upper k order statistics X_(1) <= ... <= X_(n) as:
    xi = (1/k) * Sum_{i=1 to k} [ ln(X_(n-i+1)) - ln(X_(n-k)) ]
    The tail index alpha_L is then given by alpha_L = 1/xi.
    """

    def estimate_tail_index(self, displacement_samples: np.ndarray, n_tail_points: int) -> float:
        """
        Calculates the Hill tail index (alpha_L) for the given samples and threshold.

        Args:
            displacement_samples: Array of observed displacements.
            n_tail_points: Number of upper order statistics (k) to include in the estimate.

        Returns:
            float: The estimated tail index alpha_L.

        Raises:
            ValueError: If n_tail_points is invalid for the sample size.
        """
        n = len(displacement_samples)
        if n_tail_points >= n or n_tail_points <= 0:
            raise ValueError(f"n_tail_points {n_tail_points} is invalid for sample size {n}")

        # Sort samples to find the largest values
        sorted_samples = np.sort(displacement_samples)
        
        k = n_tail_points
        # The k largest observations: X_{(n-k+1)}, ..., X_{(n)}
        tail = sorted_samples[-k:]
        # The (k+1)-th largest observation: X_{(n-k)}
        threshold = sorted_samples[-k-1]

        if threshold <= 0:
            # Hill estimator requires positive values for the log.
            # Usually applied to absolute displacements or values in the tail.
            # We filter for positive values if the threshold is non-positive.
            mask = tail > 0
            if not np.any(mask):
                return 0.0
            tail = tail[mask]
            # Since threshold is still <= 0 or not filtered, we have a problem.
            # In practice, for power law tails, we expect positive values.
            # If threshold is <= 0, we'll use a very small positive value to avoid error,
            # but ideally the input should be positive.
            threshold = 1e-10

        xi = np.mean(np.log(tail) - np.log(threshold))
        
        if xi <= 0:
            return float('inf')
            
        return float(1.0 / xi)

    def estimate_power_law_parameters(self, displacement_samples: np.ndarray) -> float:
        """
        Estimates parameters required to describe the power law behavior of the distribution tail.

        Args:
            displacement_samples: Raw displacement data from Lagrangian trajectories.

        Returns:
            float: The power law index corresponding to the right-tail behavior.
        """
        n = len(displacement_samples)
        if n < 20:
            if n < 2: return 0.0
            k = 1
        else:
            k = int(0.1 * n)
        return self.estimate_tail_index(displacement_samples, k)
class DistributionMomentAnalyzer:
    """
    Analyzes statistical moments focusing on the shape of displacement distributions.

    Supports EXP4 claims regarding platykurtic behavior at late times (t=5.0, 9.0).

    Excess kurtosis (kappa) is defined as:
    kappa = ( <(x - mu)^4> / sigma^4 ) - 3
    where mu is the mean and sigma is the standard deviation.
    Negative values (kappa < 0) indicate platykurtic behavior due to finite-domain effects.
    """

    def calculate_excess_kurtosis(self, samples: np.ndarray) -> float:
        """
        Computes the excess kurtosis and flatness for the dataset.

        Args:
            samples: Array of displacement or tracer property values.

        Returns:
            float: The excess kurtosis value.
        """
        if len(samples) == 0:
            return 0.0
        mu = np.mean(samples)
        sigma = np.std(samples)
        if sigma == 0:
            return 0.0
        
        fourth_moment = np.mean((samples - mu)**4)
        kappa = (fourth_moment / (sigma**4)) - 3
        return float(kappa)

    def verify_platykurtic_behavior(self, samples: np.ndarray, threshold: float=0.0) -> bool:
        """
        Verifies if the distribution displays late-time platykurtic behavior.

        Args:
            samples: Data points from late-time Lagrangian distributions.
            threshold: Optional negative threshold for strict verification.

        Returns:
            bool: True if the distribution is verified as platykurtic.
        """
        kappa = self.calculate_excess_kurtosis(samples)
        return kappa < threshold
class NormalityTester:
    """
    Performs Kolmogorov-Smirnov (KS) tests to verify the Gaussianity of displacements.

    Used in EXP4 to evaluate if normalized displacements at t=2.0 follow N(0, 1).
    """

    def perform_ks_test_against_gaussian(self, samples: np.ndarray) -> float:
        """
        Runs the KS test against a standard normal distribution.

        Args:
            samples: Normalized displacement data.

        Returns:
            float: The p-value of the test.
        """
        if len(samples) == 0:
            return 1.0
            
        # 1. Sort samples
        x = np.sort(samples)
        n = len(x)
        
        # 2. Compute normal CDF for each x: F(x) = 0.5 * (1 + erf(x / sqrt(2)))
        def norm_cdf(val):
            return 0.5 * (1 + math.erf(val / math.sqrt(2.0)))
        
        cdf_vals = np.array([norm_cdf(v) for v in x])
        
        # 3. Compute KS statistic D_n
        j = np.arange(1, n + 1)
        d_plus = np.max(j / n - cdf_vals)
        d_minus = np.max(cdf_vals - (j - 1) / n)
        d_n = max(d_plus, d_minus)
        
        # 4. Approximate p-value using the Kolmogorov distribution
        # For large n, p-value \approx Q(d_n * sqrt(n))
        l = d_n * math.sqrt(n)
        
        if l < 0.2:
            return 1.0
            
        p = 0.0
        for k in range(1, 101):
            term = 2 * ((-1)**(k-1)) * math.exp(-2 * (k**2) * (l**2))
            p += term
            if abs(term) < 1e-10:
                break
        return max(0.0, min(1.0, p))

    def compare_distributions(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """
        Compares two empirical distributions using the two-sample KS test.

        Args:
            data_a: First set of displacement samples.
            data_b: Second set of displacement samples.

        Returns:
            float: Maximum distance between the two empirical distribution functions.
        """
        n1 = len(data_a)
        n2 = len(data_b)
        if n1 == 0 or n2 == 0:
            return 1.0
            
        x1 = np.sort(data_a)
        x2 = np.sort(data_b)
        
        # Combine all sample points for evaluation of empirical CDFs
        data_all = np.sort(np.unique(np.concatenate([x1, x2])))
        
        cdf1 = np.searchsorted(x1, data_all, side='right') / n1
        cdf2 = np.searchsorted(x2, data_all, side='right') / n2
        
        d_nm = np.max(np.abs(cdf1 - cdf2))
        return float(d_nm)
