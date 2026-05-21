import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, List

from src.experiments.config.parameters import PhysicsConfig
from src.integration.state import TrajectoryDataset
from src.physics.fields import VelocitySnapshot
from src.experiments.orchestration import Experiment, ExperimentResult
from src.analysis.dynamics.residence_time import LagrangianQSampler, QSignalDecayAnalyzer
from src.physics.vortex_criteria import QCriterionCalculator
from src.io.vtk_reader import SolenoidalSnapshotReader, VTKSequenceProcessor
from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.io.persistence import TrajectoryHDF5Store
from src.io.result_writers import VortexTimescaleCSVWriter, VortexTimescaleMetrics

@dataclass(frozen=True)
class VortexResidenceReport:
    """
    Output contract for EXP3 as defined in the repo-purpose report.
    Contains the calculated timescales for the ensemble-wide vortex trapping analysis.
    """
    tau_q: float
    te: float
    ratio_tau_te: float


class QSignalAutocorrelationWorkflow:
    """
    Orchestrates the calculation of Lagrangian Q-signal decay dynamics.
    
    This workflow extracts the Q-value along tracer paths and determines
    the characteristic e-folding time tau_Q.

    Equations:
        Autocorrelation: C_Q(t) = <Q(t_0)Q(t_0 + t)> / <Q(t_0)^2>
        Timescale: C_Q(tau_Q) = 1/e

    Args:
        config: Physics configuration for numerical thresholds.
    """

    def __init__(self, config: PhysicsConfig):
        self.config = config
        self.sampler = LagrangianQSampler(QCriterionCalculator())
        self.analyzer = QSignalDecayAnalyzer()

    def calculate_ensemble_timescale(self, dataset: TrajectoryDataset, snapshots: Sequence[VelocitySnapshot]) -> float:
        """
        Processes the full ensemble to find the 1/e decay time of the Q-signal.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            snapshots: The ordered sequence of 3D velocity snapshots (DS1).

        Returns:
            float: The ensemble-averaged residence timescale tau_Q.

        Raises:
            RuntimeError: If the autocorrelation never reaches the 1/e threshold.
        """
        if not snapshots:
            raise ValueError("No snapshots provided for timescale calculation.")
        
        # Sample Q-criterion values along the trajectories
        q_history = self.sampler.sample_q_history(dataset, snapshots)
        
        # Analyze the decay of the Q-signal autocorrelation
        stats = self.analyzer.analyze_decay_dynamics(
            q_history=q_history,
            timestamps=dataset.timestamps,
            te=self.config.large_eddy_time_te
        )
        
        # Verify if tau_Q was actually found (not just the last timestamp)
        # Note: determine_e_folding_time returns last lag if not found.
        # Check against correlation series to see if it actually dropped below 1/e.
        if stats.correlation_series[-1] > (1.0 / np.exp(1.0)) and stats.tau_q >= (dataset.timestamps[-1] - dataset.timestamps[0]):
             raise RuntimeError("The Q-signal autocorrelation never reaches the 1/e threshold within the available time range.")

        return stats.tau_q


class VortexResidenceTimeExperiment(Experiment):
    """
    Implementation of EXP3: Vortex Residence Time Calculation.
    
    This experiment aims to reproduce Claim C2: 'Vortex trapping events are brief, 
    lasting only about 7% of a large-eddy turnover time'.
    
    Required Outputs:
        - Artifact: CSV table with columns [tau_Q, Te, ratio_tau_Te].
        - Metrics: Validation of the 0.077 ratio target.

    The experiment follows this procedure:
    1. Load velocity snapshots (DS1) and tracer trajectories (DS2).
    2. Compute Q-criterion field for snapshots using the equations:
       Q = 0.5 * (||Ω||^2 - ||S||^2)
       Ω = 0.5 * (∇v - (∇v)^T)
       S = 0.5 * (∇v + (∇v)^T)
    3. Perform Q-signal autocorrelation analysis to find tau_Q (1/e decay).
    4. Evaluate the ratio tau_Q / Te against the target 0.077.

    Args:
        config: Physics configuration containing Te (2.6) and target (0.077).
        data_dir: Path to directory containing raw VTK snapshots (DS1).
        trajectory_path: Path to the DS2 trajectory file generated in EXP1.
        output_dir: Path to save result CSV and plots.
    """

    def __init__(self, config: PhysicsConfig, data_dir: Path, trajectory_path: Path, output_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.trajectory_path = Path(trajectory_path)
        self.output_dir = Path(output_dir)
        
        self.workflow = QSignalAutocorrelationWorkflow(config)
        self.trajectory_store = TrajectoryHDF5Store()
        
        # Setup VTK snapshot reader
        self.snapshot_reader = SolenoidalSnapshotReader(
            VTKHeaderParser(),
            VTKStructuredPointExtractor(),
            VTKFieldExtractor()
        )
        self.sequence_processor = VTKSequenceProcessor(self.snapshot_reader)

    def execute(self) -> ExperimentResult:
        """
        Executes the residence time workflow and saves a CSV report.

        Returns:
            ExperimentResult: Contains success status and paths to the generated CSV report.

        Raises:
            FileNotFoundError: If input snapshots or trajectories are missing.
            ValueError: If trajectory data is inconsistent with snapshot count.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {self.data_dir}")
        if not self.trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectory_path}")

        # 1. Collect and sort VTK snapshot files
        vtk_paths = sorted(list(self.data_dir.glob("Turb.hydro_w.*.vtk")))
        if not vtk_paths:
            raise FileNotFoundError(f"No VTK snapshots found in {self.data_dir}")

        # 2. Load tracer trajectories (DS2)
        dataset = self.trajectory_store.load(self.trajectory_path)
        
        # 3. Load velocity snapshots (DS1)
        # We need snapshots corresponding to the trajectory timestamps.
        snapshots = []
        for t in dataset.timestamps:
            # Map time t (absolute) to VTK index
            # Based on DS1: index 18903 matches time 189.03
            idx = int(round(t / 0.01))
            vtk_path = self.data_dir / f"Turb.hydro_w.{idx}.vtk"
            
            if not vtk_path.exists():
                return ExperimentResult(
                    experiment_id="EXP3",
                    success=False,
                    metrics={},
                    artifacts={},
                    errors=[f"VTK snapshot not found at {vtk_path} for time {t}"]
                )
            snapshots.append(self.snapshot_reader.load_snapshot(vtk_path))

        # 4. Perform autocorrelation analysis to find tau_Q
        try:
            tau_q = self.workflow.calculate_ensemble_timescale(dataset, snapshots)
        except Exception as e:
             return ExperimentResult(
                experiment_id="EXP3",
                success=False,
                metrics={},
                artifacts={},
                errors=[f"Workflow execution failed: {str(e)}"]
            )

        # 5. Evaluate against target and generate report
        te = self.config.large_eddy_time_te
        ratio = tau_q / te
        
        # Save CSV artifact
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "vortex_residence_timescale.csv"
        
        writer = VortexTimescaleCSVWriter()
        metrics = VortexTimescaleMetrics(tau_q=tau_q, t_e=te, ratio_tau_te=ratio)
        writer.write_summary(csv_path, metrics)

        return ExperimentResult(
            experiment_id="EXP3",
            success=True,
            metrics={
                "tau_q": float(tau_q),
                "te": float(te),
                "ratio_tau_te": float(ratio),
                "target_ratio": self.config.residence_time_ratio_target,
                "error_relative": abs(ratio - self.config.residence_time_ratio_target) / self.config.residence_time_ratio_target
            },
            artifacts={
                "residence_time_table": str(csv_path)
            }
        )

