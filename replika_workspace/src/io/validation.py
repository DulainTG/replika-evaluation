import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.integration.state import TrajectoryDataset
from src.experiments.config.parameters import PhysicsConfig

@dataclass(frozen=True)
class IntegrationConsistencyReport:
    """
    Container for Lagrangian tracer validation results as required by EXP1.
    
    Attributes:
        mean_squared_velocity: The ensemble average of the squares of tracer velocities 
            sum(|v|^2)/N across all time steps.
        coordinate_min: The minimum coordinate value across all dimensions and tracers.
        coordinate_max: The maximum coordinate value across all dimensions and tracers.
        is_within_bounds: True if all coordinates fall within [0, domain_size_L].
        is_temporally_consistent: True if timestamps are strictly monotonically increasing.
    """
    mean_squared_velocity: float
    coordinate_min: float
    coordinate_max: float
    is_within_bounds: bool
    is_temporally_consistent: bool

# For backward compatibility with existing usage in src/io/result_writers.py and tests
IntegrationMetrics = IntegrationConsistencyReport

class IntegrationConsistencyValidator:
    """
    Validator for ensuring Lagrangian tracer trajectories (DS2) remain physically 
    and numerically consistent during EXP1 simulation.

    This validator performs the 'mean squared velocity' and 'coordinate range check' 
    prescribed in the EXP1 content contract to verify RK4 integration and 
    periodic boundary adherence (L=1.0).
    """

    def __init__(self, config: Optional[PhysicsConfig] = None):
        """
        Initialize the validator with physics configurations.

        Args:
            config: Optional configuration object containing domain_size_L and other physics constants.
        """
        self.config = config

    def validate_trajectory_consistency(self, dataset: TrajectoryDataset) -> IntegrationConsistencyReport:
        """
        Evaluates the TrajectoryDataset against expected physical bounds and temporal logic.

        Calculates the mean squared velocity <|v|^2> and checks coordinate ranges 
        against the periodic domain boundaries defined in PhysicsConfig.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2) to validate.

        Returns:
            IntegrationConsistencyReport summarizing metrics and boolean integrity checks.

        Raises:
            ValueError: If the dataset is empty or contains NaN/Inf values, or if config is missing.
        """
        if dataset.positions.size == 0:
            raise ValueError("Trajectory dataset is empty.")
        
        if self.config is None:
            raise ValueError("PhysicsConfig is required for validation but was not provided.")
        
        if not np.all(np.isfinite(dataset.positions)):
            raise ValueError("Trajectory dataset contains NaN or Inf position values.")
        
        if not np.all(np.isfinite(dataset.velocities)):
            raise ValueError("Trajectory dataset contains NaN or Inf velocity values.")

        # 1. Mean Squared Velocity: (1/N) * sum(|v|^2)
        # v has shape (N_snapshots, N_tracers, 3)
        v_sq_mag = np.sum(dataset.velocities**2, axis=-1)
        mean_squared_velocity = float(np.mean(v_sq_mag))

        # 2. Coordinate Range Check
        coordinate_min = float(np.min(dataset.positions))
        coordinate_max = float(np.max(dataset.positions))
        is_within_bounds = bool(coordinate_min >= 0.0 and coordinate_max <= self.config.domain_size_L)

        # 3. Monotonicity Check
        if len(dataset.timestamps) < 2:
            is_temporally_consistent = True
        else:
            is_temporally_consistent = bool(np.all(np.diff(dataset.timestamps) > 0))

        return IntegrationConsistencyReport(
            mean_squared_velocity=mean_squared_velocity,
            coordinate_min=coordinate_min,
            coordinate_max=coordinate_max,
            is_within_bounds=is_within_bounds,
            is_temporally_consistent=is_temporally_consistent
        )

    def validate_trajectories(self, trajectories: TrajectoryDataset, config: PhysicsConfig) -> IntegrationMetrics:
        """
        Backward compatibility wrapper for validate_trajectories.
        Used by existing tests and potentially other modules.
        """
        self.config = config
        return self.validate_trajectory_consistency(trajectories)
