from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Protocol, Iterable, List, Tuple
from src.physics.fields import VelocitySnapshot
from src.experiments.config.parameters import PhysicsConfig
from src.integration.seeding import TracerSeedingStrategy
from src.integration.interpolation import TrilinearGridInterpolator
from src.integration.boundaries import PeriodicBoundaryMapper
from src.integration.solvers import RK4IntegrationKernel, RK4SubstepManager

@dataclass(frozen=True)
class TrajectoryDataset:
    """
    DS2: Lagrangian tracer trajectory dataset.
    Contains coordinates and local velocities sampled at every time step.
    """
    positions: np.ndarray  # Shape: (N_snapshots, N_tracers, 3)
    velocities: np.ndarray # Shape: (N_snapshots, N_tracers, 3)
    timestamps: np.ndarray # Shape: (N_snapshots,)
    n_tracers: int

    def __post_init__(self):
        """Verify consistency of the trajectory ensemble."""
        if self.positions.shape != self.velocities.shape:
            raise ValueError("Position and velocity arrays must have identical shapes.")
        if self.positions.shape[0] != len(self.timestamps):
            raise ValueError("Temporal dimension of trajectories must match timestamps.")
        if self.positions.shape[1] != self.n_tracers:
            raise ValueError("Tracer count in data does not match explicit n_tracers count.")


class PassiveParticleIntegrator(Protocol):
    """
    Solves the Lagrangian evolution equation for passive tracers.

    The integrator tracks an ensemble of particles as they are advected by 
    a velocity field without exerting back-reaction on the flow.

    The governing equation for the trajectory x(t) is:
    dx/dt = v(x(t), t)

    where v is the velocity field provided by snapshots (DS1).
    """

    def advance_state(self, current_positions: np.ndarray, snapshot_start: VelocitySnapshot, snapshot_end: VelocitySnapshot) -> np.ndarray:
        """
        Advances the particle positions from the time of snapshot_start to 
        snapshot_end using numerical integration (e.g., RK4).

        Args:
            current_positions: Array of shape (n_tracers, 3) representing coordinates in [0, L].
            snapshot_start: The velocity field at the beginning of the interval.
            snapshot_end: The velocity field at the end of the interval.

        Returns:
            np.ndarray: Updated positions of shape (n_tracers, 3) after advancing state.

        Raises:
            ValueError: If position coordinates fall outside the domain bounds.
        """
        ...


class RK4Integrator:
    """
    A concrete implementation of the PassiveParticleIntegrator using 4th-order Runge-Kutta.
    """
    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.interpolator = TrilinearGridInterpolator()
        self.boundary_mapper = PeriodicBoundaryMapper()
        self.kernel = RK4IntegrationKernel(self.interpolator, self.boundary_mapper)
        self.manager = RK4SubstepManager()

    def advance_state(self, current_positions: np.ndarray, snapshot_start: VelocitySnapshot, snapshot_end: VelocitySnapshot) -> np.ndarray:
        # Check domain bounds
        L = self.config.domain_size_L
        if np.any(current_positions < 0) or np.any(current_positions >= L):
            raise ValueError("Position coordinates fall outside the domain bounds.")

        n_substeps = self.config.rk4_substeps_per_snapshot
        t_start = snapshot_start.time
        t_end = snapshot_end.time
        
        dt = self.manager.calculate_dt(t_start, t_end, n_substeps)
        timestamps = self.manager.get_step_timestamps(t_start, dt, n_substeps)
        
        positions = current_positions.copy()
        for t_curr in timestamps:
            positions = self.kernel.execute_substep(
                positions, 
                snapshot_start, 
                snapshot_end, 
                dt, 
                t_start, 
                t_end, 
                t_curr
            )
        return positions


class PassiveTracerTrajectoryGenerator:
    """
    Orchestrates the generation of the Lagrangian tracer trajectory dataset (DS2).

    This generator implements the procedure for EXP1: Tracer Trajectory Generation.
    It initializes tracers using a seeding strategy and integrates them through 
    a sequence of velocity snapshots (DS1) to produce the trajectory output.

    The resulting DS2 includes coordinates (x, y, z) and local velocity v at every 
    time step to enable downstream dispersion and vortex trapping analysis.
    """

    def __init__(self, config: PhysicsConfig, seeding_strategy: TracerSeedingStrategy):
        """
        Initialize the generator with physical constants and seeding rules.

        Args:
            config: Constants such as n_tracers (8000) and domain_size_L (1.0).
            seeding_strategy: Protocol for initializing tracer coordinates.
        """
        self.config = config
        self.seeding_strategy = seeding_strategy
        self.integrator = RK4Integrator(config)

    def generate_from_snapshots(self, snapshots: Iterable[VelocitySnapshot]) -> TrajectoryDataset:
        """
        Generates the full DS2 trajectory dataset from a sequence of 3D snapshots.

        Procedure (EXP1):
        1. Seed n_tracers (8,000) at random uniform coordinates.
        2. Advance positions between snapshot pairs using RK4 with 10 substeps.
        3. Use trilinear interpolation to sample local velocities.
        4. Apply periodic boundaries (modulo 1.0) at each step.
        5. Record positions, velocities, and timestamps.

        Args:
            snapshots: An iterator providing DS1 VelocitySnapshot objects.

        Returns:
            TrajectoryDataset: The generated DS2 dataset containing the ensemble history.

        Raises:
            RuntimeError: If snapshots are insufficient for the configured total_snapshots (200).
        """
        # 1. Seed n_tracers (8,000) at random uniform coordinates.
        current_positions = self.seeding_strategy.seed_tracers(self.config)
        
        positions_history = []
        velocities_history = []
        timestamps = []
        
        prev_snapshot: Optional[VelocitySnapshot] = None
        snapshot_count = 0
        expected_snapshots = 200
        
        for snapshot in snapshots:
            if prev_snapshot is not None:
                # 2. Advance positions between snapshot pairs using RK4 with 10 substeps.
                # 3 & 4 are handled within the integrator.
                current_positions = self.integrator.advance_state(
                    current_positions, prev_snapshot, snapshot
                )
            
            # 5. Record positions, velocities, and timestamps.
            # Record state at the time of the current snapshot.
            positions_history.append(current_positions.copy())
            
            # Sample local velocity at current positions.
            current_velocities = self.integrator.interpolator.evaluate_velocity_field(
                snapshot, current_positions
            )
            velocities_history.append(current_velocities)
            timestamps.append(snapshot.time)
            
            prev_snapshot = snapshot
            snapshot_count += 1
            if snapshot_count == expected_snapshots:
                break

        if snapshot_count < expected_snapshots:
            raise RuntimeError(f"Snapshots are insufficient for the configured total_snapshots ({expected_snapshots}). Got {snapshot_count}.")

        return TrajectoryDataset(
            positions=np.stack(positions_history),
            velocities=np.stack(velocities_history),
            timestamps=np.array(timestamps),
            n_tracers=self.config.n_tracers
        )
