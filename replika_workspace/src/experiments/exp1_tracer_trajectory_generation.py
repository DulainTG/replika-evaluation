import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from src.experiments.orchestration import Experiment, ExperimentResult
from src.experiments.config.parameters import PhysicsConfig
from src.physics.fields import VelocitySnapshot
from src.io.vtk_reader import SolenoidalSnapshotReader
from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.integration.state import TrajectoryDataset
from src.integration.seeding import UniformRandomSeeder
from src.integration.boundaries import PeriodicBoundaryMapper
from src.integration.solvers import RK4IntegrationKernel, RK4SubstepManager
from src.integration.interpolation import TrilinearGridInterpolator

class SnapshotPairManager:
    """
    Manages the sequential loading and temporal state of VTK velocity snapshot pairs (DS1).
    
    Ensures that the integration loop has access to snapshots at time t and t+dt,
    handling the file indexing and memory transitions between steps.
    """

    def __init__(self, data_directory: Path, snapshot_indices: List[int]):
        """
        Initialize with the directory containing VTK files and the targeted index sequence.
        
        Args:
            data_directory: Path to the raw DS1 VTK files.
            snapshot_indices: List of numerical indices (e.g., [18903, 18913, ...]) to load.
        """
        self.data_directory = Path(data_directory)
        self.snapshot_indices = snapshot_indices
        self.reader = SolenoidalSnapshotReader(
            header_parser=VTKHeaderParser(),
            layout_extractor=VTKStructuredPointExtractor(),
            field_extractor=VTKFieldExtractor()
        )

    def get_load_sequence(self) -> List[Tuple[int, int]]:
        """
        Generates the sequence of pairs to be integrated.
        
        Returns:
            List of (index_t, index_t_plus_1) defining the temporal steps.
        """
        return [(self.snapshot_indices[i], self.snapshot_indices[i+1]) 
                for i in range(len(self.snapshot_indices) - 1)]

    def load_pair(self, t_index: int, next_index: int) -> Tuple[VelocitySnapshot, VelocitySnapshot]:
        """
        Loads a specific pair of VTK files into memory as VelocitySnapshot objects.
        
        Args:
            t_index: Current snapshot file index.
            next_index: Consecutive snapshot file index.
            
        Returns:
            A tuple of (S_t, S_{t+dt}) snapshots ready for interpolation.
            
        Raises:
            FileNotFoundError: If a snapshot index cannot be mapped to a local file.
        """
        path_t = self.data_directory / f"Turb.hydro_w.{t_index}.vtk"
        path_next = self.data_directory / f"Turb.hydro_w.{next_index}.vtk"
        
        if not path_t.exists():
             raise FileNotFoundError(f"Snapshot file not found: {path_t}")
        if not path_next.exists():
             raise FileNotFoundError(f"Snapshot file not found: {path_next}")

        return self.reader.load_snapshot(path_t), self.reader.load_snapshot(path_next)

    def slide_window(self, previous_next_snapshot: VelocitySnapshot, next_index: int) -> Tuple[VelocitySnapshot, VelocitySnapshot]:
        """
        Optimizes loading by reusing the second snapshot of a pair as the first of the next pair.
        
        Args:
            previous_next_snapshot: The snapshot formerly at t+dt.
            next_index: The new target index for t+2dt.
            
        Returns:
            The new consecutive pair.
        """
        path_next = self.data_directory / f"Turb.hydro_w.{next_index}.vtk"
        if not path_next.exists():
            raise FileNotFoundError(f"Snapshot file not found: {path_next}")
        
        new_next_snapshot = self.reader.load_snapshot(path_next)
        return previous_next_snapshot, new_next_snapshot

class TracerTrajectoryGenerator(Experiment):
    """
    Implementation of EXP1: Tracer Trajectory Generation.
    
    Generates Lagrangian tracer trajectories (DS2) from raw velocity snapshots (DS1) 
    using a 4th-order Runge-Kutta (RK4) integrator with 10 substeps per snapshot pair.
    """

    def __init__(self, config: PhysicsConfig, pair_manager: SnapshotPairManager):
        """
        Args:
            config: Physical/numerical constants (n_tracers=8000, L=1.0).
            pair_manager: Provider for DS1 snapshot pairs.
        """
        self.config = config
        self.pair_manager = pair_manager
        self.seeder = UniformRandomSeeder(seed=42)
        self.interpolator = TrilinearGridInterpolator()
        self.boundary_mapper = PeriodicBoundaryMapper()
        self.kernel = RK4IntegrationKernel(self.interpolator, self.boundary_mapper)
        self.step_manager = RK4SubstepManager()

    def execute(self) -> ExperimentResult:
        """
        Executes the full trajectory generation workflow.
        
        1. Random uniform seeding of 8,000 tracers in [0, 1].
        2. Iterative loop over consecutive snapshot pairs.
        3. RK4 integration dx/dt = v(x(t), t) with trilinear interpolation.
        4. Periodic boundary correction: x = x mod L.
        5. Construction of TrajectoryDataset (DS2).
        6. Verification of consistency metrics.
        
        Returns:
            ExperimentResult containing DS2 metadata and validation metrics (JSON).
        """
        # 1. Random uniform seeding
        current_positions = self.seeder.seed_tracers(self.config)
        
        indices = self.pair_manager.snapshot_indices
        if not indices:
             return ExperimentResult(
                experiment_id="EXP1",
                success=False,
                metrics={},
                artifacts={},
                errors=["No snapshot indices provided."]
            )
             
        # Load first snapshot for initial state recording
        first_path = self.pair_manager.data_directory / f"Turb.hydro_w.{indices[0]}.vtk"
        if not first_path.exists():
            return ExperimentResult(
                experiment_id="EXP1",
                success=False,
                metrics={},
                artifacts={},
                errors=[f"First snapshot not found: {first_path}"]
            )
        
        prev_snap = self.pair_manager.reader.load_snapshot(first_path)
        
        positions_history = [current_positions.copy()]
        velocities_history = [self.interpolator.evaluate_velocity_field(prev_snap, current_positions)]
        timestamps = [prev_snap.time]
        
        # 2. Iterative loop over consecutive snapshot pairs.
        for i in range(len(indices) - 1):
            next_idx = indices[i+1]
            try:
                # Use slide window for efficiency
                _, curr_snap = self.pair_manager.slide_window(prev_snap, next_idx)
            except Exception as e:
                return ExperimentResult(
                    experiment_id="EXP1",
                    success=False,
                    metrics={},
                    artifacts={},
                    errors=[f"Failed to load snapshot {next_idx}: {str(e)}"]
                )
            
            # 3 & 4. RK4 integration and Periodic boundary correction
            current_positions = self.integrate_pair_interval(prev_snap, curr_snap, current_positions)
            
            # Record state
            positions_history.append(current_positions.copy())
            velocities_history.append(self.interpolator.evaluate_velocity_field(curr_snap, current_positions))
            timestamps.append(curr_snap.time)
            
            prev_snap = curr_snap
            
        # 5. Construction of TrajectoryDataset (DS2).
        try:
            dataset = TrajectoryDataset(
                positions=np.stack(positions_history),
                velocities=np.stack(velocities_history),
                timestamps=np.array(timestamps),
                n_tracers=self.config.n_tracers
            )
        except Exception as e:
            return ExperimentResult(
                experiment_id="EXP1",
                success=False,
                metrics={},
                artifacts={},
                errors=[f"Failed to construct TrajectoryDataset: {str(e)}"]
            )
        
        # 6. Verification of consistency metrics.
        v_sq = np.sum(dataset.velocities**2, axis=-1)
        mean_v_sq = np.mean(v_sq)
        
        metrics = {
            "mean_squared_velocity": float(mean_v_sq),
            "coordinate_ranges": {
                "min": dataset.positions.min(axis=(0, 1)).flatten().tolist(),
                "max": dataset.positions.max(axis=(0, 1)).flatten().tolist()
            },
            "timestamps": {
                "start": float(dataset.timestamps[0]),
                "end": float(dataset.timestamps[-1]),
                "count": len(dataset.timestamps)
            },
            "n_tracers": int(dataset.n_tracers)
        }
        
        return ExperimentResult(
            experiment_id="EXP1",
            success=True,
            metrics=metrics,
            artifacts={}
        )

    def integrate_pair_interval(self, s_start: VelocitySnapshot, s_end: VelocitySnapshot, initial_positions: np.ndarray) -> np.ndarray:
        """
        Performs integration across a single snapshot interval using RK4.
        
        Accommodates the requirement of 10 substeps per snapshot, sampling
        velocity via trilinear interpolation across the temporal interval [t, t+dt].
        
        Args:
            s_start: Flow field at the start of the interval.
            s_end: Flow field at the end of the interval.
            initial_positions: Shape (N_tracers, 3) coordinates.
            
        Returns:
            Updated positions after the interval integration.
        """
        n_substeps = self.config.rk4_substeps_per_snapshot
        t_start = s_start.time
        t_end = s_end.time
        
        dt = self.step_manager.calculate_dt(t_start, t_end, n_substeps)
        timestamps = self.step_manager.get_step_timestamps(t_start, dt, n_substeps)
        
        positions = initial_positions.copy()
        for t_curr in timestamps:
            positions = self.kernel.execute_substep(
                positions, 
                s_start, 
                s_end, 
                dt, 
                t_start, 
                t_end, 
                t_curr
            )
        return positions

    def _apply_periodic_boundaries(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Applies modulo 1.0 to ensure tracers remain within the cubic L=1 domain.
        
        Args:
            coordinates: Raw coordinates potentially outside [0, 1].
            
        Returns:
            Coordinates wrapped into the periodic domain.
        """
        return self.boundary_mapper.wrap_coordinates(coordinates, self.config.domain_size_L)

