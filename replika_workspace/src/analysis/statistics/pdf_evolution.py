import numpy as np
from dataclasses import dataclass
from typing import Tuple
from src.integration.state import TrajectoryDataset

class DisplacementDistributionScaler:
    """
    Responsible for scaling Lagrangian displacement distributions to unit variance.
    
    This facilitates comparison across different time lags and against the 
    standard Gaussian distribution (mu=0, sigma=1) as required by EXP4.
    """

    def scale_to_unit_variance(self, displacements: np.ndarray) -> np.ndarray:
        """
        Scales raw displacements by the ensemble standard deviation.
        
        The transformation follows the equation: 
        delta_x_scaled = (delta_x - <delta_x>) / sqrt(<(delta_x - <delta_x>)^2>)
        where the denominator is the standard deviation (sigma).

        Args:
            displacements: 1D array of single-component displacements for the ensemble.

        Returns:
            np.ndarray: Displacements shifted to zero mean and scaled to unit variance.

        Raises:
            ValueError: If the standard deviation is zero or the input is empty.
        """
        if displacements.size == 0:
            raise ValueError("Displacements array is empty.")
        
        sigma = np.std(displacements)
        if sigma == 0:
            raise ValueError("Standard deviation of displacements is zero; cannot scale to unit variance.")
        
        mean = np.mean(displacements)
        return (displacements - mean) / sigma

@dataclass(frozen=True)
class DisplacementPDFResult:
    """
    Container for a computed Probability Density Function of displacements.
    """
    lag_time: float
    bin_centers: np.ndarray
    densities: np.ndarray
    standard_deviation: float
    n_samples: int

class NormalizedPDFGenerator:
    """
    Generates normalized displacement PDFs from trajectory data for specific time lags.
    
    Supports EXP4 (Displacement PDF Evolution) requirements for lags: 
    [0.5, 1.0, 2.0, 5.0, 9.0].
    """

    def __init__(self, scaler: DisplacementDistributionScaler):
        """
        Args:
            scaler: The service used to perform unit variance scaling.
        """
        self.scaler = scaler

    def generate_lag_pdf(self, dataset: TrajectoryDataset, lag_index: int, n_bins: int=100, range_sigma: Tuple[float, float]=(-5.0, 5.0)) -> DisplacementPDFResult:
        """
        Processes a specific time lag to produce a normalized displacement PDF.

        Procedure (EXP4):
        1. Extract single-component displacements (delta_x) for the given lag.
        2. Normalize by the ensemble standard deviation (sigma).
        3. Compute a histogram count and normalize to density.

        Args:
            dataset: The Lagrangian tracer trajectory dataset (DS2).
            lag_index: The index in the temporal dimension corresponding to the target lag.
            n_bins: Number of bins for the histogram.
            range_sigma: The range in units of standard deviation for the bins.

        Returns:
            DisplacementPDFResult: The computed PDF and metadata.

        Raises:
            IndexError: If the lag_index is out of range of the dataset.
        """
        if lag_index < 0 or lag_index >= dataset.positions.shape[0]:
            raise IndexError(f"lag_index {lag_index} is out of range for the dataset.")

        # 1. Extract single-component displacements (delta_x) for the given lag.
        # Following conventions, we use the x-component (index 0).
        # Displacement is x(t) - x(0) within the periodic domain [0, L).
        delta_x = dataset.positions[lag_index, :, 0] - dataset.positions[0, :, 0]
        
        # Capture raw sigma for the result container
        sigma = np.std(delta_x)
        
        # 2. Normalize by the ensemble standard deviation (sigma).
        # This shifts the distribution to zero mean and unit variance.
        # If sigma is zero, the scaler will raise a ValueError.
        scaled_delta_x = self.scaler.scale_to_unit_variance(delta_x)
        
        # 3. Compute a histogram count and normalize to density.
        densities, bin_edges = np.histogram(
            scaled_delta_x,
            bins=n_bins,
            range=range_sigma,
            density=True
        )
        
        # Convert edges to centers for plotting and analysis
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        
        # Calculate lag time from timestamps
        lag_time = dataset.timestamps[lag_index] - dataset.timestamps[0]
        
        return DisplacementPDFResult(
            lag_time=float(lag_time),
            bin_centers=bin_centers,
            densities=densities,
            standard_deviation=float(sigma),
            n_samples=len(delta_x)
        )
