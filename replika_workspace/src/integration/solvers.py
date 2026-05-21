import numpy as np
from typing import Tuple, Sequence
from src.physics.fields import VelocitySnapshot
from src.integration.interpolation import TrilinearGridInterpolator
from src.integration.boundaries import PeriodicBoundaryMapper
from src.experiments.config.parameters import PhysicsConfig

class RK4SubstepManager:
    """
    Manages temporal resolution and substep iteration counts for the 4th-order Runge-Kutta solver.
    Aligns with EXP1 parameters: substeps_per_snapshot=10.
    """

    def calculate_dt(self, t_start: float, t_end: float, n_substeps: int) -> float:
        """
        Computes the fixed time step (dt) for integration between snapshots.

        Args:
            t_start: Timestamp of the initial snapshot.
            t_end: Timestamp of the destination snapshot.
            n_substeps: Number of RK4 substeps to perform (e.g., 10).

        Returns:
            The temporal resolution (dt) for each substep.
        """
        if n_substeps <= 0:
            return 0.0
        return (t_end - t_start) / n_substeps

    def get_step_timestamps(self, t_start: float, dt: float, n_substeps: int) -> Tuple[float, ...]:
        """
        Generates the sequence of timestamps for each substep in an interval.

        Args:
            t_start: Base timestamp.
            dt: Calculated time step size.
            n_substeps: Total number of steps.

        Returns:
            Tuple of timestamps for integration stages.
        """
        return tuple(t_start + i * dt for i in range(n_substeps))

class RK4IntegrationKernel:
    """
    Core implementation of the 4th-order Runge-Kutta multi-stage update.
    Solves dx/dt = v(x, t) using the weighted average of four stages.
    """

    def __init__(self, interpolator: TrilinearGridInterpolator, boundary_mapper: PeriodicBoundaryMapper):
        """
        Initialize kernel with grid interpolation and periodic mapping logic.
        """
        self.interpolator = interpolator
        self.boundary_mapper = boundary_mapper

    def _get_velocity(self, positions: np.ndarray, v_field_0: VelocitySnapshot, v_field_1: VelocitySnapshot, t_snap_0: float, t_snap_1: float, t_current: float) -> np.ndarray:
        """
        Computes velocity at given positions and time using bilinear time interpolation
        and trilinear spatial interpolation.
        """
        v0 = self.interpolator.evaluate_velocity_field(v_field_0, positions)
        v1 = self.interpolator.evaluate_velocity_field(v_field_1, positions)
        
        # Time interpolation
        dt_snap = t_snap_1 - t_snap_0
        if dt_snap == 0:
            return v0
        
        weight_1 = (t_current - t_snap_0) / dt_snap
        weight_0 = 1.0 - weight_1
        
        return weight_0 * v0 + weight_1 * v1

    def execute_substep(self, positions: np.ndarray, v_field_0: VelocitySnapshot, v_field_1: VelocitySnapshot, dt: float, t_snap_0: float, t_snap_1: float, t_current: float) -> np.ndarray:
        """
        Performs a single 4th-order Runge-Kutta step for an ensemble of tracers.
        
        Equations:
        k1 = v(x_n, t_n)
        k2 = v(x_n + 0.5*dt*k1, t_n + 0.5*dt)
        k3 = v(x_n + 0.5*dt*k2, t_n + 0.5*dt)
        k4 = v(x_n + dt*k3, t_n + dt)
        x_{n+1} = x_n + (dt/6) * (k1 + 2k2 + 2k3 + k4) modulo domain_size_L

        Args:
            positions: Current tracer positions (N, 3).
            v_field_0: Velocity snapshot at start of interval.
            v_field_1: Velocity snapshot at end of interval.
            dt: Substep time increment.
            t_snap_0: Global time of v_field_0.
            t_snap_1: Global time of v_field_1.
            t_current: Local time within the snapshot interval.

        Returns:
            Updated positions (N, 3) wrapped to periodic boundaries.
        """
        # k1 stage
        k1 = self._get_velocity(positions, v_field_0, v_field_1, t_snap_0, t_snap_1, t_current)
        
        # k2 stage
        x2 = self.boundary_mapper.wrap_coordinates(positions + 0.5 * dt * k1)
        k2 = self._get_velocity(x2, v_field_0, v_field_1, t_snap_0, t_snap_1, t_current + 0.5 * dt)
        
        # k3 stage
        x3 = self.boundary_mapper.wrap_coordinates(positions + 0.5 * dt * k2)
        k3 = self._get_velocity(x3, v_field_0, v_field_1, t_snap_0, t_snap_1, t_current + 0.5 * dt)
        
        # k4 stage
        x4 = self.boundary_mapper.wrap_coordinates(positions + dt * k3)
        k4 = self._get_velocity(x4, v_field_0, v_field_1, t_snap_0, t_snap_1, t_current + dt)
        
        # Final update
        new_positions = positions + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.boundary_mapper.wrap_coordinates(new_positions)

class RungeKuttaTracerSolver:
    """
    High-level backbone for the Tracer Trajectory Generation experiment (EXP1).
    Manages the multi-step integration over a sequence of snapshots to construct DS2.
    """

    def __init__(self, kernel: RK4IntegrationKernel, manager: RK4SubstepManager, config: PhysicsConfig):
        """
        Initializes the solver with integration logic and paper-specific parameters.
        """
        self.kernel = kernel
        self.manager = manager
        self.config = config

    def solve_trajectories(self, initial_positions: np.ndarray, snapshots: Sequence[VelocitySnapshot]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates an ensemble of tracers across a full snapshot sequence.

        Args:
            initial_positions: Cartesian coordinates (N=8000, 3) seeded uniformly.
            snapshots: Sequence of 100-200 VelocitySnapshot (DS1) objects.

        Returns:
            Tuple of (positions_history, velocity_history) with shape (T, N, 3).
        """
        n_snapshots = len(snapshots)
        if n_snapshots == 0:
            return np.empty((0, 0, 3)), np.empty((0, 0, 3))
            
        n_tracers = initial_positions.shape[0]
        n_substeps = self.config.rk4_substeps_per_snapshot
        
        pos_history = np.empty((n_snapshots, n_tracers, 3))
        vel_history = np.empty((n_snapshots, n_tracers, 3))
        
        current_positions = initial_positions.copy()
        
        for i in range(n_snapshots):
            # Record state at current snapshot i
            pos_history[i] = current_positions
            vel_history[i] = self.kernel.interpolator.evaluate_velocity_field(snapshots[i], current_positions)
            
            # Integrate to the next snapshot if possible
            if i < n_snapshots - 1:
                snap_start = snapshots[i]
                snap_end = snapshots[i+1]
                
                t_start = snap_start.time
                t_end = snap_end.time
                
                dt = self.manager.calculate_dt(t_start, t_end, n_substeps)
                timestamps = self.manager.get_step_timestamps(t_start, dt, n_substeps)
                
                for t_curr in timestamps:
                    current_positions = self.kernel.execute_substep(
                        current_positions, 
                        snap_start, 
                        snap_end, 
                        dt, 
                        t_start, 
                        t_end, 
                        t_curr
                    )
                    
        return pos_history, vel_history
