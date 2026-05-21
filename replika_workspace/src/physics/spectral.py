import numpy as np
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.physics.fields import VelocitySnapshot
else:
    # Import at runtime for constructor access
    from src.physics.fields import VelocitySnapshot

class ThreeDimensionalFFTProcessor:
    """
    Handles 3D Discrete Fourier Transforms for grid-based velocity fields.

    Provides the mechanism to transform spatial snapshots into spectral density maps
    and vice versa, supporting the spectral filtering requirements of EXP2.
    """

    def forward_transform(self, spatial_field: np.ndarray) -> np.ndarray:
        """
        Computes the 3D Discrete Fourier Transform of a real-valued spatial field.

        Args:
            spatial_field: A 3D numpy array representing a field component (e.g., vx).

        Returns:
            A complex-valued 3D numpy array of spectral coefficients.
        """
        return np.fft.fftn(spatial_field)

    def inverse_transform_complex_to_real(self, spectral_field: np.ndarray) -> np.ndarray:
        """
        Computes the complex-to-real inverse 3D Discrete Fourier Transform.

        Args:
            spectral_field: A complex-valued 3D numpy array in Fourier space.

        Returns:
            A real-valued 3D numpy array representing the reconstructed spatial field.
        """
        return np.real(np.fft.ifftn(spectral_field))

class SharpSpectralFilter:
    """
    Implements sharp spectral low-pass filtering by zeroing out wavenumber components.

    The filter is defined by the kernel H(k):
    H(k) = 1, if k_min <= |k| <= k_max
    H(k) = 0, otherwise
    
    where |k| is the wavenumber magnitude in the 3D Fourier space.
    """

    def compute_kernel(self, dimensions: Tuple[int, int, int], k_min: int, k_max: int) -> np.ndarray:
        """
        Generates a 3D binary mask (kernel) representing the sharp wavenumber cutoff.

        Args:
            dimensions: Grid dimensions (Ni, Nj, Nk) of the field.
            k_min: The lower bound of the wavenumber range (usually 1).
            k_max: The upper bound of the wavenumber range (usually 3).

        Returns:
            An integer 3D numpy array containing 1s for passed modes and 0s for blocked modes.
        """
        ni, nj, nk = dimensions
        
        # np.fft.fftfreq(n) * n provides integer wave numbers in the correct order for FFT.
        ki = np.fft.fftfreq(ni) * ni
        kj = np.fft.fftfreq(nj) * nj
        kk = np.fft.fftfreq(nk) * nk
        
        # Create 3D meshgrid for wavenumber components. 
        # Using indexing='ij' ensures the output matches (ni, nj, nk).
        KI, KJ, KK = np.meshgrid(ki, kj, kk, indexing='ij')
        
        # Calculate wavenumber magnitude |k|
        k_mag = np.sqrt(KI**2 + KJ**2 + KK**2)
        
        # Return sharp mask: 1 where k_min <= |k| <= k_max, 0 otherwise
        return ((k_mag >= k_min) & (k_mag <= k_max)).astype(int)

    def apply_filter(self, spectral_field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies the filtering kernel to a complex-valued spectral density field.

        Args:
            spectral_field: 3D complex numpy array from a forward transform.
            kernel: 3D binary mask of the same shape as spectral_field.

        Returns:
            The filtered complex-valued 3D spectral density.
        """
        return spectral_field * kernel
class LargeScaleFieldExtractor:
    """
    Service provider for extracting the large-scale velocity field (VLS) 
    from 3D solenoidal turbulence snapshots.

    Uses a sharp Fourier low-pass field backbone to isolate driving modes (k=1 to 3) 
    required for the anisotropic dispersion analysis in EXP2.
    """

    def __init__(self, fft_processor: 'ThreeDimensionalFFTProcessor', spectral_filter: 'SharpSpectralFilter'):
        """
        Args:
            fft_processor: Operator for 3D FFT and IFFT.
            spectral_filter: Logic for generating and applying the wavenumber mask.
        """
        self.fft_processor = fft_processor
        self.spectral_filter = spectral_filter

    def extract_vls_from_snapshot(self, snapshot: VelocitySnapshot, spectral_range: Tuple[int, int]=(1, 3)) -> VelocitySnapshot:
        """
        Filters a velocity snapshot to produce the large-scale field V_LS.

        Each component (vx, vy, vz) is transformed to Fourier space, filtered by the 
        specified spectral range, and transformed back to real space.

        Args:
            snapshot: The raw input VelocitySnapshot from DS1.
            spectral_range: Inclusive (min, max) wavenumber range for the filter.

        Returns:
            A new VelocitySnapshot representing the filtered large-scale velocity field V_LS.
        """
        k_min, k_max = spectral_range
        kernel = self.spectral_filter.compute_kernel(snapshot.grid_dimensions, k_min, k_max)
        
        # Apply FFT -> Spectral Filter -> IFFT to each component
        vls_vx = self._apply_spectral_filter(snapshot.vx, kernel)
        vls_vy = self._apply_spectral_filter(snapshot.vy, kernel)
        vls_vz = self._apply_spectral_filter(snapshot.vz, kernel)
        
        return VelocitySnapshot(
            vx=vls_vx,
            vy=vls_vy,
            vz=vls_vz,
            time=snapshot.time,
            grid_dimensions=snapshot.grid_dimensions,
            spacing=snapshot.spacing,
            origin=snapshot.origin,
            domain_size_L=snapshot.domain_size_L
        )

    def _apply_spectral_filter(self, component_field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Applies the full spectral filtering pipeline to a single field component.
        """
        spectral_field = self.fft_processor.forward_transform(component_field)
        filtered_spectral = self.spectral_filter.apply_filter(spectral_field, kernel)
        return self.fft_processor.inverse_transform_complex_to_real(filtered_spectral)
