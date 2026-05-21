from dataclasses import dataclass
from typing import Tuple, Sequence

@dataclass(frozen=True)
class PhysicsConfig:
    """
    Immutable physical and numerical constants defined in the paper.
    Provides a single source of truth for all experimental modules.
    """
    n_tracers: int = 8000
    domain_size_L: float = 1.0
    large_eddy_time_te: float = 2.6
    v_rms: float = 0.38
    rk4_substeps_per_snapshot: int = 10
    q_criterion_threshold: float = 0.0
    spectral_filter_range: Tuple[int, int] = (1, 3)
    
    # Analytical targets/limits
    anisotropy_target_lambda: float = 0.52
    residence_time_ratio_target: float = 0.077
    geometric_saturation_limit: float = 0.16666666666666666 # L^2 / 6


@dataclass(frozen=True)
class TemporalSnapshotParameters:
    """
    Configuration for the temporal discretization and indexing of velocity snapshots (DS1).
    These parameters support EXP1 (Tracer Trajectory Generation).

    Attributes:
        total_snapshots (int): Total number of sequential snapshots to process. Default is 200.
        snapshot_index_increment (int): The numerical step in file indexing (e.g., 10 for files 18903, 18913).
        integration_substeps (int): Number of numerical integration steps (RK4) performed between snapshots.
        delta_t_snapshot (float): Physical time interval between subsequent VTK snapshots.
    """
    total_snapshots: int = 200
    snapshot_index_increment: int = 10
    integration_substeps: int = 10
    delta_t_snapshot: float = 0.1
@dataclass(frozen=True)
class SpectralFilterParameters:
    """
    Configuration for the sharp Fourier spectral filter used in EXP2.
    Used to derive the large-scale velocity field V_LS from DS1.

    Attributes:
        mode_range (Tuple[int, int]): The range of wavenumbers (n) included in the low-pass filter.
            According to the paper, this is typically n = 1 to 3.
        filter_type (str): The mathematical form of the filter (defaults to 'sharp').
    """
    mode_range: Tuple[int, int] = (1, 3)
    filter_type: str = 'sharp'
@dataclass(frozen=True)
class DispersionAnalysisParameters:
    """
    Parameters for calculating anisotropic dispersion metrics (EXP2) and PDF evolution (EXP4).

    The analysis uses the equations:
    MSD_parallel(t) = <(delta_x(t) . V_LS_hat)^2>
    MSD_perp(t) = <|delta_x(t) - (delta_x(t) . V_LS_hat)V_LS_hat|^2>
    lambda(t) = MSD_parallel(t) / MSD_perp(t)

    Attributes:
        lag_time_range (Sequence[float]): The range of time lags [0, 9.0] for dispersion calculation.
        pdf_lag_checkpoints (Sequence[float]): Specific time lags for displacement PDF generation [0.5, 1.0, 2.0, 5.0, 9.0].
        anisotropy_target_ratio (float): The persistent transverse-dominant ratio target (0.52).
        geometric_saturation_limit (float): Upper bound for squared displacement in periodic L=1 domain (L^2/6 approx 0.1667).
    """
    lag_time_range: Sequence[float] = (0.0, 9.0)
    pdf_lag_checkpoints: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 9.0)
    anisotropy_target_ratio: float = 0.52
    geometric_saturation_limit: float = 0.16666666666666666
@dataclass(frozen=True)
class SolenoidalTurbulenceParameters:
    """
    Physical constants characterizing 3D solenoidal turbulence in a periodic domain.

    Supports the Lagrangian integration logic: dx/dt = v(x(t), t).

    Attributes:
        domain_size_L (float): Length of the periodic cubic domain. Default is 1.0.
        large_eddy_time_te (float): Large-eddy turnover time (Te = 2.6).
        v_rms (float): Root-mean-square velocity of the flow (0.38).
    """
    domain_size_L: float = 1.0
    large_eddy_time_te: float = 2.6
    v_rms: float = 0.38
@dataclass(frozen=True)
class VTKSnapshotParameters:
    """
    Structural specification for reading DS1 VTK files (STRUCTURED_POINTS).

    Attributes:
        grid_dimensions (Tuple[int, int, int]): Number of points in (X, Y, Z). Defaults to (129, 129, 129).
        grid_spacing (float): Uniform cell size (defaults to 0.0078125).
        grid_origin (Tuple[float, float, float]): Coordinate origin (defaults to -0.5, -0.5, -0.5).
        vtk_version (str): Expected VTK DataFile version (e.g., '2.0').
        velocity_attribute_name (str): The name of the vector field in the VTK file.
    """
    grid_dimensions: Tuple[int, int, int] = (129, 129, 129)
    grid_spacing: float = 0.0078125
    grid_origin: Tuple[float, float, float] = (-0.5, -0.5, -0.5)
    vtk_version: str = '2.0'
    velocity_attribute_name: str = 'hydro_w'
