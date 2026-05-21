import numpy as np
from typing import Tuple, Dict
from src.integration.state import TrajectoryDataset

class QuintileCohortSelector:
    """
    Identifies tracer indices belonging to the top and bottom 20% quintiles of a metric distribution.
    
    Used specifically in EXP5 (FTLE and Chaos Analysis) to define 'Trapped' and 'Free' cohorts 
    based on long-term Q-residence statistics.
    """

    def select_extreme_quintiles(self, metric_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the population into the bottom 20% and top 20% based on the provided metric.

        Args:
            metric_values: Scalar metric for each tracer (e.g., residence fraction).

        Returns:
            Tuple of (bottom_20_indices, top_20_indices).

        Raises:
            ValueError: If the input array does not contain enough tracers to form quintiles.
        """
        n = len(metric_values)
        if n < 5:
            raise ValueError("Not enough tracers to form quintiles (need at least 5).")

        n_20 = n // 5
        sorted_indices = np.argsort(metric_values)
        
        bottom_20_indices = sorted_indices[:n_20]
        top_20_indices = sorted_indices[-n_20:]

        return bottom_20_indices, top_20_indices
class ResidenceTimeThresholder:
    """
    Partitions tracers into cohorts using specific duration or fraction thresholds.
    """

    def partition_by_duration_threshold(self, durations: np.ndarray, threshold_te_units: float, te: float=2.6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partitions tracers based on whether their residence time tau_Q exceeds a threshold.

        Args:
            durations: Array of residence times tau_Q for the ensemble.
            threshold_te_units: Threshold in units of Te (e.g., 0.07).
            te: Large-eddy turnover time (Te = 2.6 per PhysicsConfig).

        Returns:
            Tuple of (indices_below, indices_above).
        """
        threshold_value = threshold_te_units * te
        indices_below = np.where(durations <= threshold_value)[0]
        indices_above = np.where(durations > threshold_value)[0]
        return indices_below, indices_above

    def partition_by_fraction_threshold(self, fractions: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Partitions tracers based on the fraction of time spent in vortices (Q > 0).

        Args:
            fractions: Array of residence fractions (0.0 to 1.0).
            threshold: Fraction threshold value (e.g., 0.1).

        Returns:
            Tuple of (indices_below, indices_above).
        """
        indices_below = np.where(fractions <= threshold)[0]
        indices_above = np.where(fractions > threshold)[0]
        return indices_below, indices_above
class TrajectoryCohortGrouper:
    """
    Groups Lagrangian trajectories into Trapped and Free populations for comparative evaluation.
    
    This interface facilitates the calculation of cohort-specific statistics like mean FTLE
    and displacement PDFs as required by Experiment EXP5.
    """

    def group_trajectories(self, dataset: TrajectoryDataset, cohort_indices: Dict[str, np.ndarray]) -> Dict[str, TrajectoryDataset]:
        """
        Creates sub-datasets for each named cohort from the master trajectory dataset.

        Args:
            dataset: The complete DS2 trajectory dataset.
            cohort_indices: Mapping of cohort names (e.g., 'Trapped', 'Free') to tracer indices.

        Returns:
            A dictionary mapping cohort names to their respective TrajectoryDataset subsets.

        Raises:
            IndexError: If cohort indices are invalid for the provided dataset.
        """
        grouped_datasets = {}
        for cohort_name, indices in cohort_indices.items():
            if len(indices) > 0:
                if np.any(indices < 0) or np.any(indices >= dataset.n_tracers):
                    raise IndexError(f"Cohort indices for '{cohort_name}' are out of bounds for the dataset.")
            
            # Subset the positions and velocities for the given tracer indices
            # positions shape: (N_snapshots, N_tracers, 3)
            # velocities shape: (N_snapshots, N_tracers, 3)
            subset_positions = dataset.positions[:, indices, :]
            subset_velocities = dataset.velocities[:, indices, :]

            # Create a new TrajectoryDataset for this cohort
            grouped_datasets[cohort_name] = TrajectoryDataset(
                positions=subset_positions,
                velocities=subset_velocities,
                timestamps=dataset.timestamps.copy(),
                n_tracers=len(indices)
            )

        return grouped_datasets
