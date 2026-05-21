import pytest
import numpy as np
from src.physics.fields import VelocitySnapshot
from src.experiments.config.parameters import PhysicsConfig
from src.integration.state import (
    PassiveTracerTrajectoryGenerator,
    TrajectoryDataset,
    RK4Integrator
)
from src.integration.seeding import UniformRandomSeeder
from src.integration.interpolation import TrilinearGridInterpolator
from src.integration.boundaries import PeriodicBoundaryMapper
from src.integration.solvers import RK4SubstepManager, RK4IntegrationKernel, RungeKuttaTracerSolver

class TestIntegrationIntegration:
    """
    Integration tests for the integration module.
    Focuses on interactions between seeding, interpolation, boundaries, and solvers.
    """

    @pytest.fixture
    def physics_config(self):
        return PhysicsConfig(
            n_tracers=100,
            domain_size_L=1.0,
            rk4_substeps_per_snapshot=2
        )

    @pytest.fixture
    def grid_v_dims(self):
        return (10, 10, 10)

    @pytest.fixture
    def spacing(self, grid_v_dims):
        return 1.0 / (grid_v_dims[0] - 1)

    def create_synthetic_snapshot(self, time, dims, spacing, velocity_val=0.1):
        nx, ny, nz = dims
        # Constant velocity field for simplicity in testing integration
        vx = np.full(dims, velocity_val)
        vy = np.zeros(dims)
        vz = np.zeros(dims)
        return VelocitySnapshot(
            vx=vx, vy=vy, vz=vz,
            time=time,
            grid_dimensions=dims,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            domain_size_L=1.0
        )

    def test_full_trajectory_generation_pipeline(self, physics_config, grid_v_dims, spacing):
        """
        Tests the end-to-end flow of generating trajectories from snapshots.
        Integrates:
        - UniformRandomSeeder
        - PassiveTracerTrajectoryGenerator
        - RK4Integrator
        - TrilinearGridInterpolator
        - PeriodicBoundaryMapper
        - RK4SubstepManager
        - RK4IntegrationKernel
        """
        seeder = UniformRandomSeeder(seed=42)
        generator = PassiveTracerTrajectoryGenerator(physics_config, seeder)

        # Override the expected_snapshots for testing
        import unittest.mock
        with unittest.mock.patch('src.integration.state.PassiveTracerTrajectoryGenerator.__getattribute__', side_effect=None) as mock_attr:
             # Actually it's easier to just provide the expected number of snapshots
             # and check if it handles it.
             pass

        # Let's adjust physics_config for the test
        # We need to Provide enough snapshots.
        # But PassiveTracerTrajectoryGenerator has a hardcoded expected_snapshots = 200.
        # I should check if I can modify that or if I need to provide 200 snapshots.

        # Wait, I see this in src/integration/state.py:
        # expected_snapshots = 200
        # ...
        # if snapshot_count == expected_snapshots:
        #     break
        # ...
        # if snapshot_count < expected_snapshots:
        #     raise RuntimeError(f"Snapshots are insufficient for the configured total_snapshots ({expected_snapshots}). Got {snapshot_count}.")

        # This makes it hard to test with few snapshots. I should probably monkeypatch it.
        
        # Actually, let's look at the source again.
        # It's a local variable inside generate_from_snapshots.
        
        # If I can't change it, I'll have to provide 200 snapshots, but each can be small.
        # But the requirement says each test should be < 10s. 200 snapshots with 100 tracers might be slow.
        
        n_snapshots = 200
        snapshots = [self.create_synthetic_snapshot(float(i), grid_v_dims, spacing) for i in range(n_snapshots)]

        dataset = generator.generate_from_snapshots(snapshots)

        assert isinstance(dataset, TrajectoryDataset)
        assert dataset.n_tracers == physics_config.n_tracers
        assert dataset.positions.shape == (n_snapshots, physics_config.n_tracers, 3)
        assert dataset.velocities.shape == (n_snapshots, physics_config.n_tracers, 3)
        assert dataset.timestamps.shape == (n_snapshots,)
        
        # Check that tracers moved. 
        # Velocity is (0.1, 0, 0). dt between snapshots is 1.0. 
        # Total time is 199.0.
        # Displacement should be around 19.9. Modulo 1.0 it should be back to somewhere.
        # Given constant velocity and RK4, it should be exact.
        
        initial_pos = dataset.positions[0]
        final_pos = dataset.positions[-1]
        expected_displacement = (n_snapshots - 1) * 0.1
        expected_final_pos_x = (initial_pos[:, 0] + expected_displacement) % 1.0
        
        np.testing.assert_allclose(final_pos[:, 0], expected_final_pos_x, atol=1e-5)
        np.testing.assert_allclose(final_pos[:, 1], initial_pos[:, 1], atol=1e-5)
        np.testing.assert_allclose(final_pos[:, 2], initial_pos[:, 2], atol=1e-5)

    def test_rk4_integrator_advance_state_periodic(self, physics_config, grid_v_dims, spacing):
        """
        Tests RK4Integrator's ability to advance state across periodic boundaries.
        Integrates:
        - RK4Integrator
        - TrilinearGridInterpolator
        - PeriodicBoundaryMapper
        - RK4SubstepManager
        - RK4IntegrationKernel
        """
        integrator = RK4Integrator(physics_config)
        
        # Position near the right boundary
        current_positions = np.array([
            [0.95, 0.5, 0.5]
        ])
        
        # Velocity that will push it across the boundary
        # Snapshot 0 at t=0, Snapshot 1 at t=1.0
        # Velocity = 0.1 -> displacement = 0.1
        # New position = 0.95 + 0.1 = 1.05 -> wrapped to 0.05
        snap0 = self.create_synthetic_snapshot(0.0, grid_v_dims, spacing, velocity_val=0.1)
        snap1 = self.create_synthetic_snapshot(1.0, grid_v_dims, spacing, velocity_val=0.1)
        
        new_positions = integrator.advance_state(current_positions, snap0, snap1)
        
        expected_pos = np.array([[0.05, 0.5, 0.5]])
        np.testing.assert_allclose(new_positions, expected_pos, atol=1e-7)

    def test_integrator_out_of_bounds_error(self, physics_config, grid_v_dims, spacing):
        """
        Tests that RK4Integrator raises ValueError for out-of-bounds positions before starting.
        """
        integrator = RK4Integrator(physics_config)
        
        # Position out of bounds
        current_positions = np.array([
            [1.5, 0.5, 0.5]
        ])
        
        snap0 = self.create_synthetic_snapshot(0.0, grid_v_dims, spacing)
        snap1 = self.create_synthetic_snapshot(1.0, grid_v_dims, spacing)
        
        with pytest.raises(ValueError, match="Position coordinates fall outside the domain bounds."):
            integrator.advance_state(current_positions, snap0, snap1)

    def test_interpolation_boundary_near_mesh_points(self, grid_v_dims, spacing):
        """
        Tests interpolation accuracy near mesh nodes and across periodic boundaries.
        Integrates:
        - TrilinearGridInterpolator
        - TrilinearGridKernel
        """
        interpolator = TrilinearGridInterpolator()
        
        # Create a field with a gradient: vx = x
        nx, ny, nz = grid_v_dims
        x = np.linspace(0, 1, nx, endpoint=True)
        # Note: TrilinearGridInterpolator uses origin + np.mod(positions - origin, L)
        # And spacing.
        
        vx = np.zeros(grid_v_dims)
        for i in range(nx):
            vx[i, :, :] = i * spacing
        
        vy = np.zeros(grid_v_dims)
        vz = np.zeros(grid_v_dims)
        
        snap = VelocitySnapshot(
            vx=vx, vy=vy, vz=vz,
            time=0.0,
            grid_dimensions=grid_v_dims,
            spacing=spacing,
            origin=(0.0, 0.0, 0.0),
            domain_size_L=1.0
        )
        
        # Test interpolation at a mesh point
        test_pos = np.array([[0.2, 0.0, 0.0]]) # 0.2 is a multiple of spacing if nx=11, spacing=0.1.
        # For nx=10, spacing=1/9. 0.2 is not a mesh point. 0.2/ (1/9) = 1.8. 
        
        # Let's use a position that is exactly on a mesh point.
        test_pos = np.array([[2 * spacing, 0.0, 0.0]])
        vel = interpolator.evaluate_velocity_field(snap, test_pos)
        np.testing.assert_allclose(vel[0, 0], 2 * spacing)

        # Test interpolation between mesh points
        test_pos = np.array([[2.5 * spacing, 0.0, 0.0]])
        vel = interpolator.evaluate_velocity_field(snap, test_pos)
        np.testing.assert_allclose(vel[0, 0], 2.5 * spacing)
        
        # Test periodic wrapping in interpolation
        test_pos = np.array([[1.1, 0.0, 0.0]]) # 1.1 wraps to 0.1
        vel = interpolator.evaluate_velocity_field(snap, test_pos)
        np.testing.assert_allclose(vel[0, 0], 0.1, atol=1e-7)

    def test_seeding_and_trajectory_initialization(self, physics_config):
        """
        Tests the seeding of tracers and how they are used to initialize the generator.
        Integrates:
        - UniformRandomSeeder
        - RandomUniformGenerator
        - PhysicsConfig
        """
        seeder = UniformRandomSeeder(seed=123)
        positions = seeder.seed_tracers(physics_config)
        
        assert positions.shape == (physics_config.n_tracers, 3)
        assert np.all(positions >= 0.0)
        assert np.all(positions <= physics_config.domain_size_L)
        
        # Verify reproducibility
        seeder2 = UniformRandomSeeder(seed=123)
        positions2 = seeder2.seed_tracers(physics_config)
        np.testing.assert_array_equal(positions, positions2)

    def test_substep_manager_dt_calculation(self):
        """
        Tests individual temporal integration component.
        """
        manager = RK4SubstepManager()
        dt = manager.calculate_dt(t_start=1.0, t_end=2.0, n_substeps=10)
        assert dt == 0.1
        
        timestamps = manager.get_step_timestamps(t_start=1.0, dt=0.1, n_substeps=10)
        assert len(timestamps) == 10
        assert timestamps[0] == 1.0
        assert timestamps[-1] == 1.9
        
        # Zero substeps case
        assert manager.calculate_dt(1.0, 2.0, 0) == 0.0

    def test_periodic_boundary_mapper(self):
        """
        Tests direct coordinate wrapping logic.
        """
        mapper = PeriodicBoundaryMapper()
        
        # Wrap positive values
        assert mapper.wrap_coordinates(np.array([1.2, 0.5, 0.8]), domain_size=1.0).tolist() == pytest.approx([0.2, 0.5, 0.8])
        
        # Wrap negative values
        assert mapper.wrap_coordinates(np.array([-0.2, 0.5, 0.8]), domain_size=1.0).tolist() == pytest.approx([0.8, 0.5, 0.8])
        
        # Map to unit domain
        assert mapper.map_to_unit_domain(np.array([1.5, -0.5, 2.5])).tolist() == pytest.approx([0.5, 0.5, 0.5])
        
        # Error case
        with pytest.raises(ValueError, match="domain_size must be positive"):
            mapper.wrap_coordinates(np.array([0.5, 0.5, 0.5]), domain_size=-1.0)

    def test_interpolation_on_cartesian_grid(self, grid_v_dims, spacing):
        """
        Tests scalar field interpolation.
        """
        interpolator = TrilinearGridInterpolator()
        
        # Scalar field with a known value at each grid point
        field = np.zeros(grid_v_dims)
        for i in range(grid_v_dims[0]):
            for j in range(grid_v_dims[1]):
                for k in range(grid_v_dims[2]):
                    field[i, j, k] = i + 2*j + 3*k
        
        origin = (0.0, 0.0, 0.0)
        # Position at (i=1, j=1, k=1)
        positions = np.array([[spacing, spacing, spacing]])
        val = interpolator.evaluate_on_cartesian_grid(field, grid_v_dims, spacing, origin, positions)
        assert pytest.approx(val[0]) == 1 + 2*1 + 3*1

        # Position at (i=1.5, j=1.5, k=1.5)
        positions = np.array([[1.5*spacing, 1.5*spacing, 1.5*spacing]])
        val = interpolator.evaluate_on_cartesian_grid(field, grid_v_dims, spacing, origin, positions)
        assert pytest.approx(val[0]) == 1.5 + 2*1.5 + 3*1.5

    def test_trajectory_dataset_consistency_check(self):
        """
        Tests internal validation of TrajectoryDataset.
        """
        n_snapshots = 5
        n_tracers = 10
        pos = np.zeros((n_snapshots, n_tracers, 3))
        vel = np.zeros((n_snapshots, n_tracers, 3))
        times = np.arange(n_snapshots)
        
        # Valid dataset
        dataset = TrajectoryDataset(pos, vel, times, n_tracers)
        assert dataset.n_tracers == n_tracers

        # Inconsistent shapes
        with pytest.raises(ValueError, match="Position and velocity arrays must have identical shapes."):
            TrajectoryDataset(pos, np.zeros((n_snapshots, n_tracers, 2)), times, n_tracers)
            
        with pytest.raises(ValueError, match="Temporal dimension of trajectories must match timestamps."):
            TrajectoryDataset(pos, vel, np.arange(n_snapshots + 1), n_tracers)
            
        with pytest.raises(ValueError, match="Tracer count in data does not match explicit n_tracers count."):
            TrajectoryDataset(pos, vel, times, n_tracers + 1)

    def test_runge_kutta_tracer_solver(self, physics_config, grid_v_dims, spacing):
        """
        Tests the lower-level RungeKuttaTracerSolver.
        """
        interpolator = TrilinearGridInterpolator()
        mapper = PeriodicBoundaryMapper()
        kernel = RK4IntegrationKernel(interpolator, mapper)
        manager = RK4SubstepManager()
        solver = RungeKuttaTracerSolver(kernel, manager, physics_config)

        initial_positions = np.array([[0.5, 0.5, 0.5]])
        n_snapshots = 3
        # Constant velocity field
        snapshots = [self.create_synthetic_snapshot(float(i), grid_v_dims, spacing, velocity_val=0.1) for i in range(n_snapshots)]

        pos_hist, vel_hist = solver.solve_trajectories(initial_positions, snapshots)

        assert pos_hist.shape == (n_snapshots, 1, 3)
        assert vel_hist.shape == (n_snapshots, 1, 3)
        
        # Check positions
        # t=0: [0.5, 0.5, 0.5]
        # t=1: [0.6, 0.5, 0.5]
        # t=2: [0.7, 0.5, 0.5]
        expected_pos = np.array([
            [[0.5, 0.5, 0.5]],
            [[0.6, 0.5, 0.5]],
            [[0.7, 0.5, 0.5]]
        ])
        np.testing.assert_allclose(pos_hist, expected_pos, atol=1e-7)

    def test_generate_uniform_distribution(self):
        from src.integration.seeding import generate_uniform_distribution
        points = generate_uniform_distribution(n_points=50, dim=3, low=0.0, high=1.0, seed=42)
        assert points.shape == (50, 3)
        assert np.all(points >= 0.0) and np.all(points <= 1.0)
        
        with pytest.raises(ValueError, match="n_points must be positive"):
            generate_uniform_distribution(n_points=0, dim=3, low=0.0, high=1.0)
        
        with pytest.raises(ValueError, match="low boundary .* must be strictly less than high boundary"):
            generate_uniform_distribution(n_points=10, dim=3, low=1.0, high=1.0)

