from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
from pathlib import Path

from src.experiments.config.parameters import PhysicsConfig
from src.integration.state import TrajectoryDataset
from src.physics.fields import VelocitySnapshot
from src.analysis.dynamics.residence_time import (
    TrappingEvent, 
    LagrangianQSampler, 
    VortexResidenceTracker
)
from src.analysis.dynamics.chaos_ftle import (
    TangentLinearIntegrator,
    FTLECalculator,
    ChaosCorrelationAnalyzer
)
from src.physics.tensors import VelocityGradientCalculator, CentralDifferenceDerivativeScheme
from src.experiments.orchestration import Experiment, ExperimentResult
from src.io.vtk_reader import SolenoidalSnapshotReader, VTKSequenceProcessor
from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.io.persistence import TrajectoryHDF5Store
from src.physics.vortex_criteria import QCriterionCalculator

@dataclass(frozen=True)
class CohortChaosStats:
    """
    Statistical summary of chaotic properties for a tracer cohort.
    Matches the content contract for the EXP5 markdown table.
    """
    mean_ftle: float
    std_ftle: float
    pearson_r: float
    n_samples: int

class ChaosCorrelationWorkflow:
    """
    Workflow for calculating the relationship between chaotic stretching (FTLE) 
    and Lagrangian residence fractions in coherent vortices.
    
    Analyzes Claim C4: Long-time Finite-Time Lyapunov Exponents (FTLE) are 
    ineffective at distinguishing between the dynamics of tracers trapped in 
    vortices and those moving freely.
    """

    def __init__(self, config: PhysicsConfig):
        """
        Initialize the workflow with experiment parameters.
        
        Args:
            config: PhysicsConfig containing experimental thresholds and constants.
        """
        self.config = config
        self.gradient_calculator = VelocityGradientCalculator(CentralDifferenceDerivativeScheme())
        self.integrator = TangentLinearIntegrator()
        self.ftle_calculator = FTLECalculator()
        self.correlation_analyzer = ChaosCorrelationAnalyzer()

    def calculate_perturbation_magnitudes(self, dataset: TrajectoryDataset, snapshots: Sequence[VelocitySnapshot]) -> np.ndarray:
        """
        Calculates the evolution of perturbation magnitudes along the trajectories.
        Required for both long-time and exit-event FTLE analysis.
        """
        # 1. Compute velocity gradients from snapshots
        gradients = [self.gradient_calculator.compute_gradient_tensor(s) for s in snapshots]
        
        # 2. Integrate perturbation growth
        # Use default initial_epsilon 1e-6 as per TangentLinearIntegrator
        return self.integrator.integrate_perturbation_growth(dataset, gradients)

    def calculate_long_time_ftle(self, dataset: TrajectoryDataset, snapshots: Sequence[VelocitySnapshot]) -> np.ndarray:
        """
        Calculate the long-time FTLE for each tracer by integrating the growth 
        of a small perturbation vector along the path.

        Args:
            dataset: TrajectoryDataset (DS2) containing coordinates and velocities.
            snapshots: Sequence of VelocitySnapshot (DS1) for gradient calculation.

        Returns:
            np.ndarray: 1D array of scalar FTLE values of length n_tracers.
        """
        magnitudes = self.calculate_perturbation_magnitudes(dataset, snapshots)
        
        # 3. Calculate final long-time FTLE
        return self.ftle_calculator.calculate_long_time_ftle(magnitudes, dataset.timestamps)

    def compute_residence_correlation(self, ftle_values: np.ndarray, residence_fractions: np.ndarray) -> float:
        """
        Calculate the Pearson correlation coefficient between FTLE and residence fraction.

        Args:
            ftle_values: Long-time FTLE values for the ensemble.
            residence_fractions: Fraction of path-time spent in Q > 0 regions.

        Returns:
            float: Pearson correlation coefficient (r).
        """
        return self.correlation_analyzer.calculate_pearson_coefficient(ftle_values, residence_fractions)

class CohortComparisonWorkflow:
    """
    Workflow to compare chaotic behavior across specific tracer populations.
    Categorizes tracers into 'Trapped' and 'Free' cohorts based on residence metrics.
    """

    def __init__(self, config: PhysicsConfig):
        """
        Initialize the comparison workflow.
        
        Args:
            config: PhysicsConfig containing experimental thresholds.
        """
        self.config = config
        self.ftle_calculator = FTLECalculator()
        self.correlation_analyzer = ChaosCorrelationAnalyzer()

    def categorize_cohorts(self, residence_fractions: np.ndarray, top_threshold: float=20.0, bottom_threshold: float=20.0) -> Dict[str, np.ndarray]:
        """
        Partitions the ensemble into cohorts based on Q-residence percentiles.

        Args:
            residence_fractions: Array of residence fractions for the ensemble.
            top_threshold: Percentile for 'Trapped' cohort (default 20.0, representing top 20%).
            bottom_threshold: Percentile for 'Free' cohort (default 20.0, representing bottom 20%).

        Returns:
            Dict[str, np.ndarray]: Mapping from cohort labels ('Trapped', 'Free') 
                                   to arrays of tracer indices.
        """
        # Top 20% means above 80th percentile
        upper_percentile = 100.0 - top_threshold
        # Bottom 20% means below 20th percentile
        lower_percentile = bottom_threshold
        
        thresh_upper = np.percentile(residence_fractions, upper_percentile)
        thresh_lower = np.percentile(residence_fractions, lower_percentile)
        
        trapped_indices = np.where(residence_fractions >= thresh_upper)[0]
        free_indices = np.where(residence_fractions <= thresh_lower)[0]
        
        return {
            "Trapped": trapped_indices,
            "Free": free_indices
        }

    def extract_exit_event_ftle(self, ftle_timeseries: np.ndarray, timestamps: np.ndarray, events: List[TrappingEvent]) -> np.ndarray:
        """
        Identify 'vortex exit' events and record the instantaneous FTLE 
        at the timestamp of transition (Q > 0 to Q < 0).

        Args:
            ftle_timeseries: (n_snapshots, n_tracers) array of perturbation magnitudes.
            timestamps: (n_snapshots,) array of observation times.
            events: List of identified trapping exit events.

        Returns:
            np.ndarray: Array of FTLE values recorded at the moment of exit (transition).
        """
        # We assume ftle_timeseries contains perturbation magnitudes as calculated by TangentLinearIntegrator.
        return self.ftle_calculator.measure_instantaneous_exit_ftle(ftle_timeseries, timestamps, events)

    def calculate_cohort_stats(self, ftle_values: np.ndarray, residence_fractions: np.ndarray, cohort_indices: Dict[str, np.ndarray]) -> Dict[str, CohortChaosStats]:
        """
        Computes summary statistics and correlations for each cohort.
        
        Args:
            ftle_values: Array of FTLE values for the ensemble.
            residence_fractions: Array of residence fractions.
            cohort_indices: Mapping from cohort name to indices.
            
        Returns:
            Dict[str, CohortChaosStats]: Mapping from cohort name to statistics.
        """
        intermediate_results = self.correlation_analyzer.analyze_chaos_cohort_stats(
            ftle_values, residence_fractions, cohort_indices
        )
        
        stats = {}
        for cohort, res in intermediate_results.items():
            stats[cohort] = CohortChaosStats(
                mean_ftle=res.mean_ftle,
                std_ftle=res.std_ftle,
                pearson_r=res.pearson_r,
                n_samples=res.n_samples
            )
        return stats
class FTLEChaosAnalysisExperiment(Experiment):
    """
    Implementation of Experiment 5: FTLE and Chaos Analysis.
    
    Required Output: Markdown table with Mean FTLE, Std Dev, and Pearson r 
    for Trapped, Free, and Exit cohorts.
    """

    def __init__(self, config: PhysicsConfig, data_dir: Path, trajectory_path: Path, output_dir: Path):
        self.config = config
        self.data_dir = Path(data_dir)
        self.trajectory_path = Path(trajectory_path)
        self.output_dir = Path(output_dir)
        
        self.correlation_workflow = ChaosCorrelationWorkflow(config)
        self.comparison_workflow = CohortComparisonWorkflow(config)
        self.trajectory_store = TrajectoryHDF5Store()
        self.snapshot_reader = SolenoidalSnapshotReader(
            VTKHeaderParser(),
            VTKStructuredPointExtractor(),
            VTKFieldExtractor()
        )
        self.sequence_processor = VTKSequenceProcessor(self.snapshot_reader)
        self.q_sampler = LagrangianQSampler(QCriterionCalculator())
        self.residence_tracker = VortexResidenceTracker()

    def execute(self) -> ExperimentResult:
        """
        Executes the EXP5 sequence:
        1. Calculate long-time FTLE for all tracers.
        2. Identify Trapped vs Free cohorts (top/bottom 20% residency).
        3. Compute statistics and Pearson correlation.
        4. Identify exit events and their instantaneous FTLE.
        5. Generate the markdown comparison table.
        
        Returns:
            ExperimentResult: Container with stats and artifact path.
        """
        try:
            # 1. Load trajectories (DS2)
            if not self.trajectory_path.exists():
                return ExperimentResult(
                    experiment_id="EXP5",
                    success=False,
                    metrics={},
                    artifacts={},
                    errors=[f"Trajectory file not found: {self.trajectory_path}"]
                )
            dataset = self.trajectory_store.load(self.trajectory_path)
            
            # 2. Load snapshots (DS1)
            # We need snapshots corresponding to the trajectory timestamps.
            snapshots = []
            for t in dataset.timestamps:
                # Map time t (absolute) to VTK index
                # Based on DS1: index 18903 matches time 189.03
                idx = int(round(t / 0.01))
                vtk_path = self.data_dir / f"Turb.hydro_w.{idx}.vtk"
                
                if not vtk_path.exists():
                    return ExperimentResult(
                        experiment_id="EXP5",
                        success=False,
                        metrics={},
                        artifacts={},
                        errors=[f"VTK snapshot not found at {vtk_path} for time {t}"]
                    )
                snapshots.append(self.snapshot_reader.load_snapshot(vtk_path))

            # 3. Calculate perturbation magnitudes
            # We need these for both long-time FTLE and exit-event FTLE
            magnitudes = self.correlation_workflow.calculate_perturbation_magnitudes(dataset, snapshots)
            
            # 4. Calculate long-time FTLE
            long_time_ftle = self.correlation_workflow.ftle_calculator.calculate_long_time_ftle(
                magnitudes, dataset.timestamps
            )

            # 5. Identify residence metrics and cohorts
            q_history = self.q_sampler.sample_q_history(dataset, snapshots)
            threshold = self.config.q_criterion_threshold
            events = self.residence_tracker.identify_trapping_events(q_history, dataset.timestamps, threshold=threshold)
            
            total_time = dataset.timestamps[-1] - dataset.timestamps[0]
            residence_fractions = self.residence_tracker.calculate_residence_fractions(
                events, total_time, dataset.n_tracers
            )
            
            cohort_indices = self.comparison_workflow.categorize_cohorts(residence_fractions)
            cohort_stats = self.comparison_workflow.calculate_cohort_stats(
                long_time_ftle, residence_fractions, cohort_indices
            )
            
            # 6. Extract exit event stats
            exit_ftles = self.comparison_workflow.extract_exit_event_ftle(magnitudes, dataset.timestamps, events)
            # Residence metrics for exit events: we'll use duration / total_time
            exit_res_fractions = np.array([e.duration / total_time for e in events]) if events else np.array([])
            
            pearson_r_exit = self.correlation_workflow.compute_residence_correlation(exit_ftles, exit_res_fractions)
            
            exit_stats = CohortChaosStats(
                mean_ftle=float(np.mean(exit_ftles)) if len(exit_ftles) > 0 else 0.0,
                std_ftle=float(np.std(exit_ftles)) if len(exit_ftles) > 0 else 0.0,
                pearson_r=float(pearson_r_exit),
                n_samples=len(exit_ftles)
            )

            # 7. Generate markdown table
            table_md = self._generate_markdown_table(
                cohort_stats["Trapped"],
                cohort_stats["Free"],
                exit_stats
            )
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.output_dir / "ftle_chaos_analysis.md"
            with open(artifact_path, "w") as f:
                f.write("# Experiment 5: FTLE and Chaos Analysis\n\n")
                f.write(table_md)
                
            # 8. Collect metrics for the result
            metrics = {
                "trapped_mean_ftle": cohort_stats["Trapped"].mean_ftle,
                "trapped_pearson_r": cohort_stats["Trapped"].pearson_r,
                "free_mean_ftle": cohort_stats["Free"].mean_ftle,
                "free_pearson_r": cohort_stats["Free"].pearson_r,
                "exit_mean_ftle": exit_stats.mean_ftle,
                "exit_pearson_r": exit_stats.pearson_r,
                "n_exit_events": len(events)
            }

            return ExperimentResult(
                experiment_id="EXP5",
                success=True,
                metrics=metrics,
                artifacts={"chaos_analysis_table": str(artifact_path)}
            )

        except Exception as e:
            return ExperimentResult(
                experiment_id="EXP5",
                success=False,
                metrics={},
                artifacts={},
                errors=[str(e)]
            )

    def _generate_markdown_table(self, trapped_stats: CohortChaosStats, free_stats: CohortChaosStats, exit_stats: CohortChaosStats) -> str:
        """
        Creates the markdown table artifact for EXP5 report.
        """
        table = "| Cohort | Mean FTLE | Std Dev | Pearson r |\n"
        table += "| :--- | :---: | :---: | :---: |\n"
        table += f"| Trapped Tracers | {trapped_stats.mean_ftle:.4f} | {trapped_stats.std_ftle:.4f} | {trapped_stats.pearson_r:.4f} |\n"
        table += f"| Free Tracers | {free_stats.mean_ftle:.4f} | {free_stats.std_ftle:.4f} | {free_stats.pearson_r:.4f} |\n"
        table += f"| Exit Events | {exit_stats.mean_ftle:.4f} | {exit_stats.std_ftle:.4f} | {exit_stats.pearson_r:.4f} |\n"
        return table
