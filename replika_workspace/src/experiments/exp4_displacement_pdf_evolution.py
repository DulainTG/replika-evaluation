import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Sequence, Protocol, Dict
from pathlib import Path

from src.integration.state import TrajectoryDataset
from src.analysis.statistics.pdf_evolution import NormalizedPDFGenerator, DisplacementPDFResult, DisplacementDistributionScaler
from src.analysis.statistics.metrics import HillTailEstimator, DistributionMomentAnalyzer, NormalityTester
from src.experiments.orchestration import Experiment, ExperimentResult
from src.experiments.config.parameters import PhysicsConfig
from src.io.persistence import TrajectoryHDF5Store


class DisplacementHistogramWorkflow:
    """
    Workflow for generating normalized displacement histograms for individual time lags.
    
    This workflow extracts single-component displacements (delta_x) and scales them 
    by the ensemble standard deviation as required by EXP4 procedure.
    """

    def __init__(self, pdf_generator: NormalizedPDFGenerator):
        """
        Initialize with the core PDF generation service.
        
        Args:
            pdf_generator: Service for scaling and binning distributions.
        """
        self.pdf_generator = pdf_generator

    def generate_normalized_histogram(self, dataset: TrajectoryDataset, lag_time: float, n_bins: int=100) -> DisplacementPDFResult:
        """
        Generates a normalized PDF for a specific lag time.
        
        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            lag_time: The specific time lag (tau) to analyze.
            n_bins: Number of bins for the histogram.
            
        Returns:
            DisplacementPDFResult containing densities and scaled bin centers.
            
        Raises:
            ValueError: If the lag_time is not present in the dataset timestamps.
        """
        # Find the index of the lag_time in the dataset timestamps
        # timestamps start from t0. lag_time is tau. So we look for t0 + lag_time.
        t0 = dataset.timestamps[0]
        target_time = t0 + lag_time
        
        # Use find_nearest or exact match? Given snapshots are discrete, find nearest.
        # But usually we expect the lag_time to match one of the snapshot times.
        idx = np.argmin(np.abs(dataset.timestamps - target_time))
        
        # Check if the closest timestamp is close enough (e.g. within 1e-6)
        if not np.isclose(dataset.timestamps[idx], target_time, atol=1e-6):
            raise ValueError(f"Lag time {lag_time} not found in dataset timestamps. Closest: {dataset.timestamps[idx] - t0}")
            
        return self.pdf_generator.generate_lag_pdf(dataset, idx, n_bins=n_bins)
@dataclass(frozen=True)
class EXP4EvolutionMetrics:
    """
    Container for statistical diagnostics required by EXP4 for claim C3.
    """
    ks_p_value_t2: float
    excess_kurtosis_t5: float
    excess_kurtosis_t9: float
    hill_alpha_t5: float
    hill_alpha_t9: float
class PDFEvolutionWorkflow:
    """
    Coordinates the progression of displacement PDFs across multiple time lags.
    """

    def __init__(self, histogram_workflow: DisplacementHistogramWorkflow, 
                 normality_tester: NormalityTester,
                 moment_analyzer: DistributionMomentAnalyzer,
                 tail_estimator: HillTailEstimator):
        """
        Initialize with required statistical services.
        """
        self.histogram_workflow = histogram_workflow
        self.normality_tester = normality_tester
        self.moment_analyzer = moment_analyzer
        self.tail_estimator = tail_estimator

    def evaluate_progression(self, dataset: TrajectoryDataset, time_lags: Sequence[float]) -> List[DisplacementPDFResult]:
        """
        Computes the sequence of PDFs for the specified time lag checkpoints.
        
        The checkpoints correspond to key phases: early (0.5, 1.0), 
        intermediate (2.0), and late (5.0, 9.0).
        
        Args:
            dataset: DS2 trajectory data.
            time_lags: List of lag times e.g., [0.5, 1.0, 2.0, 5.0, 9.0].
            
        Returns:
            List of result objects ordered by lag time.
        """
        results = []
        for lag in time_lags:
            results.append(self.histogram_workflow.generate_normalized_histogram(dataset, lag))
        return results

    def _get_displacements_at_lag(self, dataset: TrajectoryDataset, lag_time: float) -> np.ndarray:
        t0 = dataset.timestamps[0]
        target_time = t0 + lag_time
        idx = np.argmin(np.abs(dataset.timestamps - target_time))
        if not np.isclose(dataset.timestamps[idx], target_time, atol=1e-6):
             raise ValueError(f"Lag time {lag_time} not found in dataset timestamps.")
        
        # We use x-component displacement (index 0)
        return dataset.positions[idx, :, 0] - dataset.positions[0, :, 0]

    def calculate_diagnostic_suite(self, dataset: TrajectoryDataset) -> EXP4EvolutionMetrics:
        """
        Calculates the specific statistical metrics required to verify Claim C3.
        
        Performs:
        - KS test at t=2.0 (Gaussian check).
        - Excess kurtosis at t=5.0 and t=9.0 (Platykurtic check).
        - Hill estimator at t=5.0 and t=9.0 (Tail power-law check).
        
        Args:
            dataset: DS2 trajectory data.
            
        Returns:
            EXP4EvolutionMetrics containing p-values, kurtosis, and tail indices.
        """
        # KS Test at t=2.0
        disp_t2 = self._get_displacements_at_lag(dataset, 2.0)
        # For KS test against N(0,1), we need to normalize disp_t2
        scaler = DisplacementDistributionScaler()
        scaled_t2 = scaler.scale_to_unit_variance(disp_t2)
        ks_p = self.normality_tester.perform_ks_test_against_gaussian(scaled_t2)

        # Excess Kurtosis at t=5.0 and t=9.0
        disp_t5 = self._get_displacements_at_lag(dataset, 5.0)
        disp_t9 = self._get_displacements_at_lag(dataset, 9.0)
        
        kurt_t5 = self.moment_analyzer.calculate_excess_kurtosis(disp_t5)
        kurt_t9 = self.moment_analyzer.calculate_excess_kurtosis(disp_t9)

        # Hill alpha at t=5.0 and t=9.0
        # Use only positive displacements for tail estimation as per Hill estimator implementation
        alpha_t5 = self.tail_estimator.estimate_power_law_parameters(np.abs(disp_t5))
        alpha_t9 = self.tail_estimator.estimate_power_law_parameters(np.abs(disp_t9))

        return EXP4EvolutionMetrics(
            ks_p_value_t2=ks_p,
            excess_kurtosis_t5=kurt_t5,
            excess_kurtosis_t9=kurt_t9,
            hill_alpha_t5=alpha_t5,
            hill_alpha_t9=alpha_t9
        )
class DisplacementPDFExperiment(Experiment):
    """
    EXP4: Displacement PDF Evolution Experiment.
    
    Analyzes displacement distributions at specified time lags to reproduce 
    Claim C3: distributions are nearly Gaussian at intermediate times and 
    become platykurtic at late times due to finite-domain effects, 
    without heavy tails.
    
    Required Output:
        - PNG plot of normalized displacement PDFs at lags [0.5, 1.0, 2.0, 5.0, 9.0]
          overlaid with a standard Gaussian curve (mu=0, sigma=1).
    """

    def __init__(self, config: PhysicsConfig, trajectory_path: Path, output_dir: Path):
        """
        Initialize the experiment with configuration and data paths.
        
        Args:
            config: Physical parameters (large_eddy_time_te, domain_size_L).
            trajectory_path: Path to the DS2 Lagrangian trajectory dataset.
            output_dir: Directory to save the PDF plots and metrics.
        """
        self.config = config
        self.trajectory_path = Path(trajectory_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Services and Workflows
        scaler = DisplacementDistributionScaler()
        pdf_gen = NormalizedPDFGenerator(scaler)
        hist_wf = DisplacementHistogramWorkflow(pdf_gen)
        
        norm_tester = NormalityTester()
        moment_analyzer = DistributionMomentAnalyzer()
        tail_est = HillTailEstimator()
        
        self.workflow = PDFEvolutionWorkflow(
            histogram_workflow=hist_wf,
            normality_tester=norm_tester,
            moment_analyzer=moment_analyzer,
            tail_estimator=tail_est
        )

    def execute(self) -> ExperimentResult:
        """
        Executes the PDF evolution workflow and saves performance diagnostics.
        
        1. Loads DS2 trajectories.
        2. Generates normalized histograms for lags defined in PhysicsConfig.
        3. Performs KS test and kurtosis analysis.
        4. Generates the PDF evolution plot.
        
        Returns:
            ExperimentResult containing success status and artifact paths.
        """
        # 1. Load DS2 trajectories
        store = TrajectoryHDF5Store()
        try:
            dataset = store.load(self.trajectory_path)
        except Exception as e:
            return ExperimentResult(
                experiment_id="EXP4",
                success=False,
                metrics={},
                artifacts={},
                errors=[str(e)]
            )

        # 2. Generate normalized histograms
        # Lags are [0.5, 1.0, 2.0, 5.0, 9.0] as per requirement
        time_lags = [0.5, 1.0, 2.0, 5.0, 9.0]
        try:
            pdf_results = self.workflow.evaluate_progression(dataset, time_lags)

            # 3. Calculate diagnostic suite
            metrics = self.workflow.calculate_diagnostic_suite(dataset)

            # 4. Generate Plot
            plot_path = self.output_dir / "displacement_pdf_evolution.png"
            self._generate_plot(pdf_results, plot_path)

            # Optional: Save metrics
            metrics_path = self.output_dir / "evolution_metrics.json"
            self._save_metrics(metrics, metrics_path)

            return ExperimentResult(
                experiment_id="EXP4",
                success=True,
                metrics={
                    "ks_p_value_t2": metrics.ks_p_value_t2,
                    "excess_kurtosis_t5": metrics.excess_kurtosis_t5,
                    "excess_kurtosis_t9": metrics.excess_kurtosis_t9,
                    "hill_alpha_t5": metrics.hill_alpha_t5,
                    "hill_alpha_t9": metrics.hill_alpha_t9
                },
                artifacts={
                    "png_plot": str(plot_path),
                    "metrics_json": str(metrics_path)
                }
            )
        except Exception as e:
             return ExperimentResult(
                experiment_id="EXP4",
                success=False,
                metrics={},
                artifacts={},
                errors=[str(e)]
            )

    def _generate_plot(self, results: List[DisplacementPDFResult], plot_path: Path):
        """Generates the required PDF evolution plot."""
        plt.figure(figsize=(10, 7))
        
        # Color palette for different lags
        colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
        
        for res, color in zip(results, colors):
            plt.plot(res.bin_centers, res.densities, label=f't = {res.lag_time:.1f}', 
                     linewidth=2, color=color)
            
        # Standard Gaussian for reference
        x_ref = np.linspace(-5, 5, 200)
        pdf_ref = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_ref**2)
        plt.plot(x_ref, pdf_ref, 'k--', label='Normal(0,1)', linewidth=2, alpha=0.8)
        
        plt.yscale('log')
        plt.ylim(1e-4, 1.0)
        plt.xlim(-5, 5)
        plt.xlabel(r'Scaled Displacement $\delta x / \sigma$')
        plt.ylabel(r'Probability Density $P(\delta x / \sigma)$')
        plt.title('Lagrangian Displacement PDF Evolution (DS2)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def _save_metrics(self, metrics: EXP4EvolutionMetrics, metrics_path: Path):
        """Saves metrics to JSON for records."""
        import json
        with open(metrics_path, 'w') as f:
            json.dump({
                "ks_p_value_t2": metrics.ks_p_value_t2,
                "excess_kurtosis_t5": metrics.excess_kurtosis_t5,
                "excess_kurtosis_t9": metrics.excess_kurtosis_t9,
                "hill_alpha_t5": metrics.hill_alpha_t5,
                "hill_alpha_t9": metrics.hill_alpha_t9
            }, f, indent=4)
