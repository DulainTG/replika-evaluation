import numpy as np
from typing import Protocol, Optional
from src.experiments.config.parameters import PhysicsConfig

class TracerSeedingStrategy(Protocol):
    """
    Defines the contract for Lagrangian tracer initialization strategies.

    The strategy is responsible for generating the initial spatial coordinates 
    for the tracer ensemble before the integration loop begins.
    """

    def seed_tracers(self, config: PhysicsConfig) -> np.ndarray:
        """
        Generate initial coordinates for tracers.

        Args:
            config: Physical configuration containing n_tracers and domain_size_L.

        Returns:
            np.ndarray: Initial positions of shape (n_tracers, 3).

        Raises:
            ValueError: If the configuration parameters are invalid for the strategy.
        """

class RandomUniformGenerator:
    """
    A stateful generator for producing uniform stochastic distributions within a 3D domain.

    Encapsulates a pseudo-random number generator (PRNG) to ensure that 
    coordinate generation is consistent and reproducible across experiments.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the PRNG.

        Args:
            seed: Integer seed. If None, the generator is non-deterministic.
        """
        self.rng = np.random.default_rng(seed)

    def generate_3d_points(self, count: int, low: float, high: float) -> np.ndarray:
        """
        Generate a set of 3D points within the specified box constraints.

        Args:
            count: Number of points to generate (e.g., n_tracers=8000).
            low: Minimum coordinate value (typically 0.0).
            high: Maximum coordinate value (typically domain_size_L=1.0).

        Returns:
            np.ndarray: A (count, 3) array of coordinates.
        """
        return self.rng.uniform(low, high, (count, 3))

def generate_uniform_distribution(n_points: int, dim: int, low: float, high: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Computes a uniform random distribution of points in a multi-dimensional space.

    Pure logic function for generating coordinates (x, y, z) in the domain [low, high].
    Used to initialize the tracer ensemble in EXP1 where n_points=8000 and dim=3.

    Args:
        n_points: Total number of points (tracers) to generate.
        dim: Dimensionality of the space (e.g., 3 for 3D turbulence).
        low: Lower bound of the interval for each dimension.
        high: Upper bound of the interval for each dimension.
        seed: Optional seed for the internal random number generator.

    Returns:
        np.ndarray: Array of shape (n_points, dim) representing the distribution.

    Raises:
        ValueError: If bounds are inconsistent or count is non-positive.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if low >= high:
        raise ValueError(f"low boundary ({low}) must be strictly less than high boundary ({high})")
    
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, (n_points, dim))

class UniformRandomSeeder:
    """
    Implementation of random uniform seeding for EXP1 (Tracer Trajectory Generation).

    This class initializes n_tracers (8,000) at random (x, y, z) coordinates 
    within the interval [0, domain_size_L] as specified in the experiment procedure.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the seeder with a specific seed for reproducibility.

        Args:
            seed: The random seed for the generator.
        """
        self.generator = RandomUniformGenerator(seed)

    def seed_tracers(self, config: PhysicsConfig) -> np.ndarray:
        """
        Initialize 8,000 tracers at random coordinates in [0, 1].

        Args:
            config: Configuration defining n_tracers=8000 and domain_size_L=1.0.

        Returns:
            np.ndarray: Random positions array of shape (8000, 3).
        """
        if config.n_tracers <= 0:
            raise ValueError(f"Config n_tracers must be positive, got {config.n_tracers}")
        if config.domain_size_L <= 0:
            raise ValueError(f"Config domain_size_L must be positive, got {config.domain_size_L}")

        return self.generator.generate_3d_points(
            count=config.n_tracers,
            low=0.0,
            high=config.domain_size_L
        )
