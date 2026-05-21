import pytest
import numpy as np
from src.physics.fields import (
    VelocitySnapshot, 
    GridGeometryManager, 
    FieldMappingService, 
    AnisotropicAlignmentCalculator,
    SolenoidalFieldProvider
)
from src.physics.spectral import (
    ThreeDimensionalFFTProcessor, 
    SharpSpectralFilter, 
    LargeScaleFieldExtractor
)
from src.physics.tensors import (
    CentralDifferenceDerivativeScheme, 
    VelocityGradientCalculator,
    VelocityGradientTensorField
)
from src.physics.vortex_criteria import QCriterionCalculator

class TestPhysicsIntegration:
    """
    Integration tests for the physics module.
    Focuses on the interaction between components:
    - Spectral processing and field extraction.
    - Tensor calculations and vortex criteria.
    - Alignment calculations and mapping services.
    """

    @pytest.fixture
    def grid_v_dims(self):
        # Using 33 to have 32 intervals and match the 129-style grid (N+1)
        return (33, 33, 33)

    @pytest.fixture
    def spacing(self):
        # L=1.0, N_intervals=32
        return 1.0 / 32.0

    @pytest.fixture
    def synthetic_snapshot(self, grid_v_dims, spacing):
        nx, ny, nz = grid_v_dims
        # Create a grid including the periodic duplicate at the end
        x = np.linspace(0, 1, nx, endpoint=True)
        y = np.linspace(0, 1, ny, endpoint=True)
        z = np.linspace(0, 1, nz, endpoint=True)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Create a field with a known k=2 wavenumber
        # Note: for a grid with a duplicate point, a pure sine wave is periodic if
        # we have an integer number of cycles over the N intervals.
        # k=2 over L=1 means 2 cycles.
        vx = np.sin(2 * np.pi * 2 * X)
        vy = np.zeros_like(X)
        vz = np.zeros_like(X)

        return VelocitySnapshot(
            vx=vx, vy=vy, vz=vz,
            time=0.0,
            grid_dimensions=grid_v_dims,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            domain_size_L=1.0
        )

    def test_spectral_filtering_integration(self, synthetic_snapshot):
        """
        Tests the integration of FFTProcessor, SpectralFilter and LargeScaleFieldExtractor
        to isolate specific wavenumbers in a VelocitySnapshot.
        """
        fft_proc = ThreeDimensionalFFTProcessor()
        spec_filt = SharpSpectralFilter()
        extractor = LargeScaleFieldExtractor(fft_proc, spec_filt)

        # 1. Test passing the signal (k=2 is within [1, 3])
        vls_passed = extractor.extract_vls_from_snapshot(synthetic_snapshot, spectral_range=(1, 3))
        
        # The FFT on a grid with a duplicate point won't be perfectly a single spike
        # because the FFT treats all 33 points as unique. 
        # However, it should still largely preserve the signal or at least keep it non-zero.
        # Given how the SharpSpectralFilter is implemented (using fftfreq), 
        # it will select modes based on the total N=33.
        
        # Signal k=2 over L=1 on N=33 grid.
        # Let's verify that it's not zeroed out.
        assert np.max(np.abs(vls_passed.vx)) > 0.5
        
        # 2. Test blocking the signal (k=10 to 15 is well away from k=2)
        vls_blocked = extractor.extract_vls_from_snapshot(synthetic_snapshot, spectral_range=(10, 15))
        np.testing.assert_allclose(vls_blocked.vx, np.zeros_like(vls_blocked.vx), atol=1e-1) 
        # atol=0.1 because of spectral leakage due to the duplicate point and sharp filter

    def test_vortex_identification_pipeline(self, grid_v_dims, spacing):
        """
        Tests the full pipeline from VelocitySnapshot to Q-criterion field.
        Snapshot -> VelocityGradientCalculator -> QCriterionCalculator.
        """
        nx, ny, nz = grid_v_dims
        x = np.linspace(-0.5, 0.5, nx, endpoint=True)
        y = np.linspace(-0.5, 0.5, ny, endpoint=True)
        z = np.linspace(-0.5, 0.5, nz, endpoint=True)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Simple rankine-like vortex or just a rotational field
        # vx = -sin(2pi * y), vy = sin(2pi * x)
        vx = -np.sin(2 * np.pi * Y)
        vy = np.sin(2 * np.pi * X)
        vz = np.zeros_like(X)

        snapshot = VelocitySnapshot(
            vx=vx, vy=vy, vz=vz,
            time=1.0,
            grid_dimensions=grid_v_dims,
            spacing=spacing,
            origin=(-0.5, -0.5, -0.5),
            domain_size_L=1.0
        )

        discretizer = CentralDifferenceDerivativeScheme()
        grad_calc = VelocityGradientCalculator(discretizer)
        q_calc = QCriterionCalculator()

        gradients = grad_calc.compute_gradient_tensor(snapshot)
        q_field = q_calc.compute_q_field(gradients)

        # Q should be positive in the center where rotation dominates
        center_idx = nx // 2
        assert q_field[center_idx, center_idx, center_idx] > 0

        # Verify binary extraction
        vortex_mask = q_calc.extract_invariant_scalar_field(q_field, threshold=0.0)
        assert vortex_mask[center_idx, center_idx, center_idx] == 1.0

    def test_anisotropic_alignment_integration(self, synthetic_snapshot):
        """
        Tests AnisotropicAlignmentCalculator using a filtered large-scale field.
        """
        align_calc = AnisotropicAlignmentCalculator()

        # Test component calculation
        vls_field = np.stack([synthetic_snapshot.vx, synthetic_snapshot.vy, synthetic_snapshot.vz], axis=-1)
        vls_hat = align_calc.calculate_vls_unit_vector(vls_field)

        # Local velocity has some component in all directions
        # v_local = 2 * v_vls (parallel) + 1 * [0, 1, 0] (transverse at some points)
        v_local = 2.0 * vls_field + np.array([0.0, 1.0, 0.0])

        v_para = align_calc.project_local_velocity(v_local, vls_hat)
        v_perp = align_calc.calculate_transverse_component(v_local, vls_hat)

        # Check vector addition: v_para + v_perp = v_local
        np.testing.assert_allclose(v_para + v_perp, v_local, atol=1e-10)

        # MSD Anisotropy ratio λ
        # lambda = MSD_para / MSD_perp
        # Using mock displacements and VLS_hats
        N = 100
        mock_vls_hats = np.zeros((N, 3))
        mock_vls_hats[:, 0] = 1.0 # All V_LS aligned with X
        
        mock_displacements = np.zeros((N, 3))
        mock_displacements[:, 0] = 3.0 # Parallel component = 3
        mock_displacements[:, 1] = 2.0 # Perpendicular component = 2
        
        # MSD_para = mean(3^2) = 9
        # MSD_perp = mean(2^2 + 0^2) = 4
        # Expected lambda = 9 / 4 = 2.25
        
        lambda_val = align_calc.calculate_msd_anisotropy(mock_displacements, mock_vls_hats)
        assert pytest.approx(lambda_val) == 2.25

    def test_grid_mapping_consistency(self, synthetic_snapshot):
        """
        Integrates GridGeometryManager and FieldMappingService.
        """
        geo_manager = GridGeometryManager()
        mapping_service = FieldMappingService()

        # 1. Coordinates construction integration
        x_coords, y_coords, z_coords = geo_manager.get_grid_coordinate_arrays(
            synthetic_snapshot.grid_dimensions, 
            synthetic_snapshot.spacing, 
            synthetic_snapshot.origin
        )
        
        # 2. Extract grid indices for the whole grid
        ni, nj, nk = synthetic_snapshot.grid_dimensions
        # Just test a few points
        indices = np.array([
            [0, 0, 0],
            [ni - 1, nj - 1, nk - 1]
        ])
        
        physical_coords = mapping_service.map_vtk_to_physical(indices, synthetic_snapshot)
        
        # Point [0,0,0] should be origin
        np.testing.assert_allclose(physical_coords[0], synthetic_snapshot.origin, atol=1e-10)
        
        # Point [ni-1, nj-1, nk-1] should be origin + (N-1)*spacing
        expected_last = np.array(synthetic_snapshot.origin) + (np.array(synthetic_snapshot.grid_dimensions) - 1) * synthetic_snapshot.spacing
        # Wrapped into [0, 1]
        expected_last = expected_last % synthetic_snapshot.domain_size_L
        np.testing.assert_allclose(physical_coords[1], expected_last, atol=1e-10)

    def test_full_physics_flow_example(self, synthetic_snapshot):
        """
        Tests a composite flow representing a mini-experiment:
        1. Extract large-scale field.
        2. Compute gradient and vortices of the original field.
        3. Compute alignment of vortices (gradient) relative to large-scale field.
        """
        fft_proc = ThreeDimensionalFFTProcessor()
        spec_filt = SharpSpectralFilter()
        extractor = LargeScaleFieldExtractor(fft_proc, spec_filt)
        
        discretizer = CentralDifferenceDerivativeScheme()
        grad_calc = VelocityGradientCalculator(discretizer)
        q_calc = QCriterionCalculator()
        
        align_calc = AnisotropicAlignmentCalculator()

        # Step 1: V_LS
        vls_snapshot = extractor.extract_vls_from_snapshot(synthetic_snapshot, spectral_range=(1, 3))
        vls_field = np.stack([vls_snapshot.vx, vls_snapshot.vy, vls_snapshot.vz], axis=-1)
        vls_hat = align_calc.calculate_vls_unit_vector(vls_field)

        # Step 2: Gradients and Q-field
        gradients = grad_calc.compute_gradient_tensor(synthetic_snapshot)
        q_field = q_calc.compute_q_field(gradients)
        
        # Step 3: Interaction - for example, project something related to gradients onto V_LS
        # (Though not a standard physical quantity, we test the API integration)
        dv_dx = np.stack([gradients.dvx_dx, gradients.dvy_dx, gradients.dvz_dx], axis=-1)
        projected_grad_x = align_calc.project_local_velocity(dv_dx, vls_hat)
        
        assert projected_grad_x.shape == dv_dx.shape
        assert q_field.shape == synthetic_snapshot.grid_dimensions

    def test_physics_error_cases(self, synthetic_snapshot):
        """
        Covers failure modes and boundary conditions.
        """
        q_calc = QCriterionCalculator()
        mapping_service = FieldMappingService()
        
        # 1. Test Q-calc with None input
        with pytest.raises(ValueError, match="VelocityGradientTensorField cannot be None"):
            q_calc.compute_q_field(None)

        # 2. Test Q-calc with malformed tensor components (None component)
        grads = VelocityGradientTensorField(
            dvx_dx=None, dvx_dy=None, dvx_dz=None,
            dvy_dx=None, dvy_dy=None, dvy_dz=None,
            dvz_dx=None, dvz_dy=None, dvz_dz=None
        )
        with pytest.raises(ValueError):
            q_calc.compute_q_field(grads)

        # 3. Mass density normalization with all zeros
        zeros = np.zeros((10, 10, 10))
        normalized = mapping_service.map_mass_density(zeros)
        np.testing.assert_array_equal(normalized, zeros)

        # 4. Central difference on tiny field (n < 2)
        discretizer = CentralDifferenceDerivativeScheme()
        tiny_field = np.array([[[1.0]]]) # 1x1x1
        deriv = discretizer.differentiate(tiny_field, axis=0, spacing=0.1)
        assert np.all(deriv == 0)

    def test_additional_physics_apis(self):
        """
        Covers remaining public APIs in the physics module.
        """
        geo_manager = GridGeometryManager()
        mapping_service = FieldMappingService()
        field_provider = SolenoidalFieldProvider()

        # 1. GridGeometryManager.generate_uniform_cartesian_points
        n_points = 100
        points = geo_manager.generate_uniform_cartesian_points(n_points, domain_range=(0.0, 1.0))
        assert points.shape == (n_points, 3)
        assert np.all(points >= 0.0) and np.all(points <= 1.0)

        # 2. FieldMappingService.extract_velocity_components
        data = np.ones((5, 5, 5))
        snap = VelocitySnapshot(
            vx=data, vy=data*2, vz=data*3,
            time=0.0, grid_dimensions=(5, 5, 5), spacing=0.1, origin=(0,0,0)
        )
        vec_field = mapping_service.extract_velocity_components(snap)
        assert vec_field.shape == (5, 5, 5, 3)
        np.testing.assert_array_equal(vec_field[0,0,0], [1.0, 2.0, 3.0])

        # 3. SolenoidalFieldProvider.get_turbulence_field_metadata
        metadata = field_provider.get_turbulence_field_metadata()
        assert metadata['v_rms'] == 0.38
        assert metadata['large_eddy_time_te'] == 2.6

    def test_alignment_calculator_spectral_filter_integration(self, synthetic_snapshot):
        """
        Tests the spectral filter implementation inside AnisotropicAlignmentCalculator.
        """
        align_calc = AnisotropicAlignmentCalculator()
        v_field = np.stack([synthetic_snapshot.vx, synthetic_snapshot.vy, synthetic_snapshot.vz], axis=-1)
        
        # Test passing the signal
        filtered = align_calc.compute_spectral_filter(v_field, filter_range=(1, 3))
        assert filtered.shape == v_field.shape
        assert np.max(np.abs(filtered)) > 0.5
        
        # Test blocking the signal
        filtered_blocked = align_calc.compute_spectral_filter(v_field, filter_range=(10, 15))
        np.testing.assert_allclose(filtered_blocked, 0.0, atol=1e-1)

