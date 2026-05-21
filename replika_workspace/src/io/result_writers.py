import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping

from src.io.validation import IntegrationConsistencyReport

class EnsembleConsistencyJSONWriter:
    """
    Writer for ensemble-wide validation metrics in JSON format.
    
    Supports EXP1 (Tracer Trajectory Generation) by persisting consistency checks.
    The output includes mean squared velocity, coordinate range bounds (0-1), and 
    timestamp verification as required by the EXP1 content contract.
    """

    def write_statistics(self, path: Path, metrics: IntegrationConsistencyReport) -> None:
        """
        Serializes integration metrics to a JSON file.
        
        Args:
            path: Destination file path.
            metrics: Validated consistency metrics from the Lagrangian integration ensemble.
            
        Raises:
            OSError: If the file cannot be written.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(asdict(metrics), f, indent=4)
        except Exception as e:
            raise OSError(f"Failed to write statistics to {path}: {str(e)}")

@dataclass(frozen=True)
class VortexTimescaleMetrics:
    """
    DTO for vortex residence timescales as defined in EXP3.
    
    Attributes:
        tau_q: Lag time where the Q-signal autocorrelation drops to 1/e.
               Based on Q = 0.5 * (||Omega||^2 - ||S||^2).
        t_e: Large-eddy turnover time (Te = 2.6).
        ratio_tau_te: Normalized residence time (Target: 0.077).
    """
    tau_q: float
    t_e: float
    ratio_tau_te: float

class VortexTimescaleCSVWriter:
    """
    Writer for Lagrangian and vortex timescale summaries in CSV format.
    
    Handles the output contract for EXP3 (Vortex Residence Time Calculation),
    exporting tau_Q, Te, and their ratio to verify trapping event duration claims.
    """

    def write_summary(self, path: Path, metrics: VortexTimescaleMetrics) -> None:
        """
        Writes timescale metrics to a structured CSV file.
        
        Args:
            path: Destination file path.
            metrics: Computed timescales and ratios for the ensemble.
            
        Raises:
            OSError: If the file cannot be written.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['tau_q', metrics.tau_q])
                writer.writerow(['t_e', metrics.t_e])
                writer.writerow(['ratio_tau_te', metrics.ratio_tau_te])
        except Exception as e:
            raise OSError(f"Failed to write summary to {path}: {str(e)}")

@dataclass(frozen=True)
class FTLECohortStatistics:
    """
    Statistical measures for Finite-Time Lyapunov Exponents within a specific tracer cohort.
    """
    mean_ftle: float
    std_dev: float
    pearson_r: float

class FTLECohortMarkdownWriter:
    """
    Writer for FTLE statistics comparison in Markdown table format.
    
    Supports EXP5 (FTLE and Chaos Analysis) by generating the comparison 
    report for 'Trapped', 'Free', and 'Exit Events' cohorts.
    """

    def write_comparison_table(self, path: Path, cohort_data: Mapping[str, FTLECohortStatistics]) -> None:
        """
        Generates a Markdown table comparing FTLE metrics across cohorts.
        
        Args:
            path: Destination file path.
            cohort_data: Mapping from cohort names (e.g., 'Trapped', 'Free') 
                         to their corresponding statistics.
            
        Raises:
            OSError: If the file cannot be written.
            KeyError: If required cohorts are missing from the mapping.
        """
        required_cohorts = ['Trapped', 'Free', 'Exit Events']
        for cohort in required_cohorts:
            if cohort not in cohort_data:
                raise KeyError(f"Missing required cohort data: {cohort}")
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write("| Cohort | Mean FTLE | Std Dev | Pearson R |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for cohort in required_cohorts:
                    stats = cohort_data[cohort]
                    f.write(f"| {cohort} | {stats.mean_ftle:.6f} | {stats.std_dev:.6f} | {stats.pearson_r:.6f} |\n")
        except KeyError:
            raise
        except Exception as e:
            raise OSError(f"Failed to write comparison table to {path}: {str(e)}")
