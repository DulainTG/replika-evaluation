import pytest
import numpy as np
from typing import Dict
from src.integration.state import TrajectoryDataset
from src.analysis.cohorts import (
    QuintileCohortSelector,
    ResidenceTimeThresholder,
    TrajectoryCohortGrouper
)
from src.analysis.statistics.metrics import (
    EnsembleAveragingService,
    VelocityAutocorrelationCalculator,
    HillTailEstimator
)
from src.analysis.statistics.pdf_evolution import (
    DisplacementDistributionScaler,
    NormalizedPDFGenerator,
    DisplacementPDFResult
)
from src.analysis.dynamics.residence_time import (
    LagrangianQSampler,
    QSignalDecayAnalyzer,
    VortexResidenceTracker,
    TrappingEvent
)
from src.analysis.dynamics.chaos_ftle import (
    TangentLinearIntegrator,
    FTLECalculator,
    ChaosCorrelationAnalyzer
)
from src.analysis.dispersion.msd import AnisotropicMSDCalculator
from src.analysis.dispersion.anisotropy import (
    AnisotropicProjectionEngine,
    AnisotropicComponentCalculator,
    AnisotropyTemporalTracker
)
from src.physics.fields import VelocitySnapshot
from src.physics.vortex_criteria import QCriterionCalculator
from src.physics.tensors import VelocityGradientTensorField

@pytest.fixture
def sample_trajectory_dataset():
    # 5 snapshots, 10 tracers, 3 dimensions
    n_snapshots = 5
    n_tracers = 10
    positions = np.random.rand(n_snapshots, n_tracers, 3)
    velocities = np.random.rand(n_snapshots, n_tracers, 3)
    timestamps = np.linspace(0, 1, n_snapshots)
    return TrajectoryDataset(positions, velocities, timestamps, n_tracers)

class TestCohortAnalysisIntegration:
    def test_cohort_selection_and_grouping(self, sample_trajectory_dataset):
        # 1. Provide some residence metrics
        residence_fractions = np.linspace(0, 1, sample_trajectory_dataset.n_tracers)
        
        # 2. Select cohorts (quintiles)
        selector = QuintileCohortSelector()
        bottom_20, top_20 = selector.select_extreme_quintiles(residence_fractions)
        
        assert len(bottom_20) == 2
        assert len(top_20) == 2
        
        # 3. Group trajectories
        grouper = TrajectoryCohortGrouper()
        cohort_indices = {'Free': bottom_20, 'Trapped': top_20}
        grouped = grouper.group_trajectories(sample_trajectory_dataset, cohort_indices)
        
        assert len(grouped) == 2
        assert 'Free' in grouped
        assert 'Trapped' in grouped
        assert grouped['Free'].n_tracers == 2
        assert grouped['Trapped'].n_tracers == 2
        assert grouped['Free'].positions.shape == (5, 2, 3)
        assert np.array_equal(grouped['Free'].timestamps, sample_trajectory_dataset.timestamps)

    def test_residence_thresholding_integration(self):
        thresholder = ResidenceTimeThresholder()
        durations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Based on Te = 2.6, 0.1 * 2.6 = 0.26
        below, above = thresholder.partition_by_duration_threshold(durations, threshold_te_units=0.1, te=2.6)
        
        # durations <= 0.26 are [0.1, 0.2] indices [0, 1]
        assert np.array_equal(below, [0, 1])
        # durations > 0.26 are [0.3, 0.4, 0.5] indices [2, 3, 4]
        assert np.array_equal(above, [2, 3, 4])
        
        fractions = np.array([0.05, 0.15, 0.25])
        below_f, above_f = thresholder.partition_by_fraction_threshold(fractions, threshold=0.1)
        assert np.array_equal(below_f, [0])
        assert np.array_equal(above_f, [1, 2])

class TestDisplacementPDFIntegration:
    def test_pdf_generation_flow(self, sample_trajectory_dataset):
        # 1. Initialize services
        scaler = DisplacementDistributionScaler()
        pdf_gen = NormalizedPDFGenerator(scaler)
        
        # 2. Test scaling directly
        displacements = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaled = scaler.scale_to_unit_variance(displacements)
        assert np.isclose(np.mean(scaled), 0.0)
        assert np.isclose(np.std(scaled), 1.0)
        
        # 3. Generate PDF for a lag
        # For our sample dataset, lag_index 2 is at t=0.5
        res = pdf_gen.generate_lag_pdf(sample_trajectory_dataset, lag_index=2, n_bins=10)
        
        assert isinstance(res, DisplacementPDFResult)
        assert res.lag_time == 0.5
        assert len(res.bin_centers) == 10
        assert len(res.densities) == 10
        assert res.n_samples == 10
        # Densities should sum up to something close to 1 when integrated (approx)
        # sum(densities * dx) ~ 1
        dx = res.bin_centers[1] - res.bin_centers[0]
        integral = np.sum(res.densities) * dx
        assert np.isclose(integral, 1.0, atol=0.1)

    def test_pdf_generation_error_cases(self, sample_trajectory_dataset):
        scaler = DisplacementDistributionScaler()
        pdf_gen = NormalizedPDFGenerator(scaler)
        
        # Invalid lag index
        with pytest.raises(IndexError):
            pdf_gen.generate_lag_pdf(sample_trajectory_dataset, lag_index=10)
            
        # Zero variance (all displacements the same)
        positions = np.zeros((5, 10, 3))
        velocities = np.zeros((5, 10, 3))
        timestamps = np.linspace(0, 1, 5)
        bad_dataset = TrajectoryDataset(positions, velocities, timestamps, 10)
        
        with pytest.raises(ValueError, match="Standard deviation of displacements is zero"):
            pdf_gen.generate_lag_pdf(bad_dataset, lag_index=2)

class TestAnisotropyIntegration:
    def test_anisotropy_calculation_flow(self, sample_trajectory_dataset):
        n_snapshots, n_tracers, _ = sample_trajectory_dataset.positions.shape
        
        # 1. Create a dummy V_LS_hat field (all pointing in x-direction)
        vls_hat_field = np.zeros_like(sample_trajectory_dataset.positions)
        vls_hat_field[:, :, 0] = 1.0
        
        # 2. Use AnisotropicMSDCalculator
        app_msd_calc = AnisotropicMSDCalculator()
        res = app_msd_calc.compute_anisotropy_stats(sample_trajectory_dataset, vls_hat_field)
        
        assert len(res.lag_times) == n_snapshots
        assert len(res.msd_parallel) == n_snapshots
        assert len(res.msd_perpendicular) == n_snapshots
        assert len(res.anisotropy_ratio) == n_snapshots
        
        # 3. Use individual components for further analysis
        proj_engine = AnisotropicProjectionEngine()
        comp_calc = AnisotropicComponentCalculator()
        tracker = AnisotropyTemporalTracker()
        
        # Test projection engine
        # delta_x = [1, 1, 1], vls_hat = [1, 0, 0] -> parallel = 1, perp_vec = [0, 1, 1]
        disp = np.array([[1.0, 1.0, 1.0]])
        vls = np.array([[1.0, 0.0, 0.0]])
        parallel, perp = proj_engine.decompose_displacement_vls(disp, vls)
        assert parallel[0] == 1.0
        assert np.array_equal(perp[0], [0.0, 1.0, 1.0])
        
        # Test overall tracker integration
        ratio_series = tracker.calculate_lambda_series(res.msd_parallel[1:], res.msd_perpendicular[1:])
        assert np.array_equal(ratio_series, res.anisotropy_ratio[1:])
class TestDynamicsIntegration:
    def test_dynamics_analysis_workflow(self, sample_trajectory_dataset):
        # 1. Setup minimal grid and snapshots
        grid_dim = (5, 5, 5)
        spacing = 1.0 / (grid_dim[0] - 1)
        origin = (-0.5, -0.5, -0.5)
        
        snapshots = []
        for t in sample_trajectory_dataset.timestamps:
            # Create a simple rotating velocity field: v = (-y, x, 0)
            x = np.linspace(-0.5, 0.5, grid_dim[0])
            y = np.linspace(-0.5, 0.5, grid_dim[1])
            z = np.linspace(-0.5, 0.5, grid_dim[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            vx = -Y
            vy = X
            vz = np.zeros_like(Z)
            
            snapshots.append(VelocitySnapshot(vx, vy, vz, t, grid_dim, spacing, origin))
            
        # 2. Sample Q history
        q_calc = QCriterionCalculator()
        q_sampler = LagrangianQSampler(q_calc)
        q_history = q_sampler.sample_q_history(sample_trajectory_dataset, snapshots)
        
        assert q_history.shape == (5, 10) # 5 snapshots, 10 tracers
        
        # In a rotating field v=(-y, x, 0), the gradient is [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
        # Omega = [[0, -1, 0], [1, 0, 0], [0, 0, 0]], S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # ||Omega||^2 = (-1)^2 + 1^2 = 2. ||S||^2 = 0.
        # Q = 0.5 * (2 - 0) = 1.0.
        # Since it's a solid body rotation, Q should be positive everywhere.
        assert np.all(q_history > 0)
        
        # 3. Analyze residence time
        tracker = VortexResidenceTracker()
        events = tracker.identify_trapping_events(q_history, sample_trajectory_dataset.timestamps, threshold=0.5)
        
        # All tracers are in Q > 0.5 for the whole duration
        assert len(events) == 10
        for event in events:
            assert event.duration == sample_trajectory_dataset.timestamps[-1] - sample_trajectory_dataset.timestamps[0]
            
        res_fractions = tracker.calculate_residence_fractions(events, total_time=1.0, n_tracers=10)
        assert np.all(res_fractions == 1.0)
        
        # 4. Integrate perturbation growth for FTLE
        integrator = TangentLinearIntegrator()
        # Need gradients fields for TangentLinearIntegrator
        from src.physics.tensors import VelocityGradientCalculator, CentralDifferenceDerivativeScheme
        grad_calc = VelocityGradientCalculator(CentralDifferenceDerivativeScheme())
        gradient_fields = [grad_calc.compute_gradient_tensor(s) for s in snapshots]
        
        magnitudes = integrator.integrate_perturbation_growth(sample_trajectory_dataset, gradient_fields)
        assert magnitudes.shape == (5, 10)
        
        # 5. Calculate FTLE
        ftle_calc = FTLECalculator()
        long_time_ftle = ftle_calc.calculate_long_time_ftle(magnitudes, sample_trajectory_dataset.timestamps)
        assert len(long_time_ftle) == 10
        
        # 6. Correlate
        corr_analyzer = ChaosCorrelationAnalyzer()
        cohort_masks = {'Trapped': np.ones(10, dtype=bool), 'Free': np.zeros(10, dtype=bool)}
        stats = corr_analyzer.analyze_chaos_cohort_stats(long_time_ftle, res_fractions, cohort_masks)
        
        assert 'Trapped' in stats
        assert stats['Trapped'].n_samples == 10
        assert 'Free' in stats
        assert stats['Free'].n_samples == 0
    def test_q_signal_decay_integration(self):
        # 1. Create a synthetic Q signal that decays exponentially
        n_snapshots = 100
        n_tracers = 10
        timestamps = np.linspace(0, 10, n_snapshots)
        # Q(t) = exp(-t)
        q_history = np.zeros((n_snapshots, n_tracers))
        for i, t in enumerate(timestamps):
            q_history[i, :] = np.exp(-t)
            
        # 2. Analyze decay
        analyzer = QSignalDecayAnalyzer()
        stats = analyzer.analyze_decay_dynamics(q_history, timestamps, te=2.6)
        
        # for Q(t) = exp(-t), R_Q(tau) = <exp(-t)exp(-(t+tau))> / <exp(-2t)> = exp(-tau)
        # exp(-tau_Q) = 1/e -> tau_Q = 1.0
        assert np.isclose(stats.tau_q, 1.0, atol=0.1)
        assert np.isclose(stats.te_ratio, 1.0 / 2.6, atol=0.05)
        
    def test_vacf_integration(self, sample_trajectory_dataset):
        calc = VelocityAutocorrelationCalculator()
        vacf = calc.calculate_vacf_series(sample_trajectory_dataset)
        
        assert len(vacf) == 5
        # VACF(0) should be the mean squared velocity (ensemble averaged)
        expected_vacf_0 = np.mean(np.sum(sample_trajectory_dataset.velocities**2, axis=-1))
        # Wait, VelocityAutocorrelationCalculator averages over possible starts too.
        # For k=0, v_start = v[0:5], v_end = v[0:5].
        # It's np.mean(np.sum(v * v, axis=-1))
        assert np.isclose(vacf[0], expected_vacf_0)

class TestStatisticsIntegration:
    def test_ensemble_averaging_flow(self, sample_trajectory_dataset):
        service = EnsembleAveragingService()
        
        # Test ensemble mean
        data = np.ones((5, 10, 3))
        mean = service.calculate_ensemble_mean(data)
        assert mean.shape == (5, 3)
        assert np.all(mean == 1.0)
        
        # Test aggregation by lag time
        lags_data = service.aggregate_by_time_lag(sample_trajectory_dataset)
        assert len(lags_data) == 5
        for lag in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert lag in lags_data
            assert lags_data[lag].shape == (10, 3)

    def test_hill_tail_estimator_integration(self):
        estimator = HillTailEstimator()
        
        # Create a Pareto distribution tail: P(X > x) = (x/xm)^-alpha
        # X = xm * (1-U)^(-1/alpha)
        xm = 1.0
        alpha = 2.5
        samples = xm * (np.random.pareto(alpha, 1000) + 1)
        
        estimated_alpha = estimator.estimate_tail_index(samples, n_tail_points=100)
        # Hill estimator should be reasonably close to alpha
        assert 1.5 < estimated_alpha < 4.0



