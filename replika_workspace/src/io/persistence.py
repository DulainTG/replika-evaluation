import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import h5py

from src.integration.state import TrajectoryDataset
from src.io.vtk_reader import DS1SolenoidalManifest

@dataclass(frozen=True)
class DS2TrajectoryManifest:
    """
    Dataset manifest for DS2: Lagrangian Tracer Trajectory Dataset.
    
    Captures metadata for trajectories generated via 4th-order Runge-Kutta (RK4) 
    integration of the equation dx/dt = v(x(t), t).
    
    Attributes:
        dataset_id: Unique identifier (DS2).
        dataset_name: Name of the dataset.
        n_tracers: Number of tracers (standard: 8,000).
        total_snapshots: Number of temporal snapshots (standard: 200).
        domain_size_L: Cubic domain side length (1.0).
        storage_format: Format used for persistence (HDF5).
        has_velocities: Boolean indicating if local velocities are sampled.
    """
    dataset_id: str = 'DS2'
    dataset_name: str = 'Lagrangian Tracer Trajectory Dataset'
    n_tracers: int = 8000
    total_snapshots: int = 200
    domain_size_L: float = 1.0
    storage_format: str = 'HDF5'
    has_velocities: bool = True


class TrajectoryHDF5Store:
    """
    Handles persistence for tracer trajectories (DS2) in HDF5 containers.
    
    This interface manages the storage of Cartesian coordinates and velocity 
    components collected during Experiment EXP1. It enables downstream analysis 
    of anisotropic dispersion (EXP2) and vortex trapping (EXP3).
    """

    def save(self, data: TrajectoryDataset, path: Path) -> None:
        """
        Persists a complete trajectory dataset to an HDF5 file.
        
        Args:
            data: The trajectory dataset containing positions, velocities, and timestamps.
            path: Destination filesystem path for the HDF5 archive.
            
        Raises:
            IOError: If the directory is not writable or serialization fails.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(path, 'w') as f:
                f.create_dataset('positions', data=data.positions, compression='gzip')
                f.create_dataset('velocities', data=data.velocities, compression='gzip')
                f.create_dataset('timestamps', data=data.timestamps, compression='gzip')
                f.attrs['n_tracers'] = data.n_tracers
                f.attrs['dataset_id'] = 'DS2'
        except Exception as e:
            raise IOError(f"Failed to save trajectory dataset to {path}: {str(e)}") from e

    def load(self, path: Path) -> TrajectoryDataset:
        """
        Loads a trajectory dataset from an HDF5 container.
        
        Args:
            path: Path to the HDF5 archive.
            
        Returns:
            Hydrated TrajectoryDataset with consistency validated against dimensions.
            
        Raises:
            FileNotFoundError: If the archive does not exist.
            ValueError: If the HDF5 content type or shape is incompatible.
            IOError: If file access fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 archive not found: {path}")
            
        try:
            with h5py.File(path, 'r') as f:
                if 'positions' not in f or 'velocities' not in f or 'timestamps' not in f:
                    raise ValueError(f"HDF5 archive {path} is missing required datasets.")
                
                positions = f['positions'][:]
                velocities = f['velocities'][:]
                timestamps = f['timestamps'][:]
                n_tracers = f.attrs.get('n_tracers')
                
                if n_tracers is None:
                    # Fallback to infer from shape if attribute is missing
                    n_tracers = positions.shape[1]
                
                return TrajectoryDataset(
                    positions=positions,
                    velocities=velocities,
                    timestamps=timestamps,
                    n_tracers=int(n_tracers)
                )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Incompatible HDF5 content in {path}: {str(e)}") from e
        except Exception as e:
            raise IOError(f"Error loading HDF5 archive {path}: {str(e)}") from e

    def get_manifest(self) -> DS2TrajectoryManifest:
        """
        Provides the manifest defining DS2 expectations.
        
        Returns:
            A manifest entry for the Lagrangian trajectory dataset.
        """
        return DS2TrajectoryManifest()


class SolenoidalMetadataStore:
    """
    Provider for persisting the manifest of the solenoidal turbulence snapshots (DS1).
    
    Ensures that the configuration of the 100 raw VTK snapshots (grid 129^3, L=1.0) 
    remains discoverable and reproducible across multiple experiment runs.
    """

    def save_manifest(self, manifest: DS1SolenoidalManifest, path: Path) -> None:
        """
        Serializes the DS1 manifest to a persistent metadata file (e.g., JSON).
        
        Args:
            manifest: The manifest mapping Athena++ VTK data units.
            path: Destination path for the metadata storage.
            
        Raises:
            IOError: If serialization or file access fails.
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(asdict(manifest), f, indent=4)
        except Exception as e:
            raise IOError(f"Failed to save manifest to {path}: {str(e)}") from e

    def load_manifest(self, path: Path) -> DS1SolenoidalManifest:
        """
        Retrieves the DS1 manifest from the persistence layer.
        
        Args:
            path: Source path where metadata is stored.
            
        Returns:
            The manifest required to initialize a SolenoidalSnapshotReader.
            
        Raises:
            FileNotFoundError: If the manifest file does not exist.
            IOError: If manifest parsing or file access fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # JSON serialization converts tuples to lists.
            # Convert grid_dimensions back to tuple for the dataclass.
            if "grid_dimensions" in data and isinstance(data["grid_dimensions"], list):
                data["grid_dimensions"] = tuple(data["grid_dimensions"])
                
            return DS1SolenoidalManifest(**data)
        except Exception as e:
            raise IOError(f"Failed to load manifest from {path}: {str(e)}") from e
