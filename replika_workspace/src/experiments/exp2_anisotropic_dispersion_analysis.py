import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any

from src.experiments.orchestration import Experiment, ExperimentResult
from src.experiments.config.parameters import PhysicsConfig, VTKSnapshotParameters
from src.integration.state import TrajectoryDataset
from src.analysis.dispersion.msd import AnisotropicMSDResults, AnisotropicMSDCalculator
from src.io.persistence import TrajectoryHDF5Store
from src.io.vtk_reader import SolenoidalSnapshotReader
from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.physics.spectral import ThreeDimensionalFFTProcessor, SharpSpectralFilter, LargeScaleFieldExtractor
from src.integration.interpolation import TrilinearGridInterpolator

class TimeLagAggregationProtocol:
    """
    Orchestrates the protocol for aggregating displacement components over discrete time lags.

    This protocol implements the ensemble averaging process required to compute 
    Mean Squared Displacement (MSD) components across the tracer population for 
    each available lag time t in [0.0, 9.0].

    Responsibilities include computing displacement vectors \delta x(t), aggregating 
    parallel projections (\delta x(t) \cdot V_LS_hat) and perpendicular rejections 
    per the equations:
    - MSD||(t) = <(delta_x(t) . V_LS_hat)^2>
    - MSD_perp(t) = <|delta_x(t) - (delta_x(t) . V_LS_hat)V_LS_hat|^2>
    """

    def aggregate_ensemble_displacements(self, dataset: TrajectoryDataset, vls_hat_field: np.ndarray) -> AnisotropicMSDResults:
        """
        Perform the ensemble-wide aggregation of displacements into parallel and perpendicular MSD.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            vls_hat_field: Unit vector field of the large-scale velocity (V_LS_hat) 
                           calculated for each tracer position and time.

        Returns:
            AnisotropicMSDResults containing lag_times, parallel MSD, perpendicular MSD, 
            and the resulting lambda(t) ratio.
        """
        calculator = AnisotropicMSDCalculator()
        return calculator.compute_anisotropy_stats(dataset, vls_hat_field)


class AnisotropicDispersionExperiment(Experiment):
    """
    Implementation of EXP2: Anisotropic Dispersion Analysis.

    This workflow tracks the temporal evolution of the anisotropy ratio lambda(t) 
    relative to the large-scale velocity field V_LS. The goal is to reproduce 
    the persistent transverse-dominant anisotropy (lambda < 1.0) observed in the paper.

    Workflow steps:
    1. Filter raw velocity snapshots (DS1) for large-scale modes (n=1 to 3).
    2. Calculate V_LS_hat along the trajectories (DS2).
    3. Compute lambda(t) = MSD||(t) / MSD_perp(t) using the TimeLagAggregationProtocol.
    4. Evaluate the evolution against the target ratio of 0.52.
    5. Generate the required PNG plot of lambda(t).

    Required Output:
        - lambda_evolution.png: Plot with X-axis=lag time t, Y-axis=lambda(t),
          including a horizontal line at 0.52.
    """

    def __init__(self, config: PhysicsConfig, data_dir: Path, output_dir: Path):
        """
        Initialize the experiment with physical parameters and I/O paths.
        
        Args:
            config: Source of truth for physical constants and target ratios (e.g., target 0.52).
            data_dir: Directory containing DS1 (VTK snapshots) and DS2 (trajectories).
            output_dir: Directory for plot artifacts.
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reader = SolenoidalSnapshotReader(
            header_parser=VTKHeaderParser(),
            layout_extractor=VTKStructuredPointExtractor(),
            field_extractor=VTKFieldExtractor()
        )
        self.interpolator = TrilinearGridInterpolator()
        self.fft_processor = ThreeDimensionalFFTProcessor()
        self.spectral_filter = SharpSpectralFilter()
        self.vls_extractor = LargeScaleFieldExtractor(self.fft_processor, self.spectral_filter)
        self.aggregation_protocol = TimeLagAggregationProtocol()

    def execute(self) -> ExperimentResult:
        """
        Executes the full anisotropy analysis workflow.

        Returns:
            ExperimentResult containing success status, lambda(t) metrics, and 
            paths to the temporal evolution plots.
        """
        try:
            # 1. Load DS2 (Trajectories)
            # Experiment requires DS2 to be present.
            ds2_path = self.data_dir / "trajectories.h5"
            if not ds2_path.exists():
                # Fallback to search for any .h5 file if trajectories.h5 doesn't exist
                h5_files = list(self.data_dir.glob("*.h5"))
                if h5_files:
                    ds2_path = h5_files[0]
                else:
                    return ExperimentResult(
                        experiment_id="EXP2",
                        success=False,
                        metrics={},
                        artifacts={},
                        errors=[f"Trajectory dataset not found at {ds2_path} and no .h5 files in {self.data_dir}"]
                    )
            
            loader = TrajectoryHDF5Store()
            dataset = loader.load(ds2_path)
            
            # 2. Derive V_LS_hat along trajectories
            vls_hat_series = []
            
            # Iterate through all snapshots in the trajectory dataset
            for i, t in enumerate(dataset.timestamps):
                # Map time t (absolute) to VTK index
                # Based on DS1: index 18903 matches time 189.03
                idx = int(round(t / 0.01))
                vtk_path = self.data_dir / f"Turb.hydro_w.{idx}.vtk"
                
                if not vtk_path.exists():
                    return ExperimentResult(
                        experiment_id="EXP2",
                        success=False,
                        metrics={},
                        artifacts={},
                        errors=[f"VTK snapshot not found at {vtk_path} for time {t}"]
                    )
                
                # Load and filter snapshot
                snapshot = self.reader.load_snapshot(vtk_path)
                v_ls_snapshot = self.vls_extractor.extract_vls_from_snapshot(
                    snapshot, spectral_range=self.config.spectral_filter_range
                )
                
                # Interpolate V_LS at tracer positions at current time
                positions_at_t = dataset.positions[i]
                v_ls_at_tracers = self.interpolator.evaluate_velocity_field(v_ls_snapshot, positions_at_t)
                
                # Compute unit vectors V_LS_hat
                v_ls_mag = np.linalg.norm(v_ls_at_tracers, axis=-1, keepdims=True)
                # Use np.divide with where to handle zero magnitude (though unlikely in turbulence)
                v_ls_hat = np.divide(v_ls_at_tracers, v_ls_mag, out=np.zeros_like(v_ls_at_tracers), where=v_ls_mag > 0)
                
                vls_hat_series.append(v_ls_hat)
            
            vls_hat_field = np.stack(vls_hat_series) # (N_snapshots, N_tracers, 3)
            
            # 3. Compute anisotropic dispersion metrics
            results = self.aggregation_protocol.aggregate_ensemble_displacements(dataset, vls_hat_field)
            
            # 4. Evaluation and plotting
            plot_path = self._track_lambda_evolution(results.lag_times.tolist(), results.anisotropy_ratio.tolist())
            
            # Metrics: Persistent lambda < 1.0 (approaching 0.52)
            # According to paper, lambda(t) saturates at late times.
            # We take mean over the second half of the lag time range.
            mid_idx = len(results.lag_times) // 2
            mean_lambda_late = np.mean(results.anisotropy_ratio[mid_idx:]) if len(results.lag_times) > 0 else 0.0
            
            metrics = {
                "mean_lambda_late": float(mean_lambda_late),
                "target_lambda": self.config.anisotropy_target_lambda,
                "max_lambda": float(np.max(results.anisotropy_ratio)) if len(results.anisotropy_ratio) > 0 else 0.0,
                "min_lambda": float(np.min(results.anisotropy_ratio)) if len(results.anisotropy_ratio) > 0 else 0.0
            }
            
            return ExperimentResult(
                experiment_id="EXP2",
                success=True,
                metrics=metrics,
                artifacts={"lambda_evolution_plot": plot_path}
            )
            
        except Exception as e:
            return ExperimentResult(
                experiment_id="EXP2",
                success=False,
                metrics={},
                artifacts={},
                errors=[str(e)]
            )

    def _track_lambda_evolution(self, lag_times: List[float], lambda_series: List[float]) -> str:
        """
        Internal utility to generate and save the temporal evolution plot.

        Args:
            lag_times: List of time lags from 0 to 9.0.
            lambda_series: Corresponding anisotropy ratio values.

        Returns:
            Path to the saved PNG plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(lag_times, lambda_series, 'b-', linewidth=2, label=r'Measured $\lambda(t)$')
        plt.axhline(y=self.config.anisotropy_target_lambda, color='r', linestyle='--', 
                    label=f'Target Ratio ({self.config.anisotropy_target_lambda})')
        
        plt.xlabel('Lag Time $t$', fontsize=12)
        plt.ylabel(r'Anisotropy Ratio $\lambda(t) = MSD_{||} / MSD_{\perp}$', fontsize=12)
        plt.title('EXP2: Temporal Evolution of Anisotropy Ratio', fontsize=14)
        plt.ylim(0, 1.2)
        if lag_times:
            plt.xlim(0, max(lag_times))
        else:
            plt.xlim(0, 9.0)
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plot_path = self.output_dir / "lambda_evolution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
