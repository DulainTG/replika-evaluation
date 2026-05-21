import pytest
import numpy as np
from pathlib import Path
import tempfile
import struct
import os
import json
import csv

from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
from src.io.vtk_reader import SolenoidalSnapshotReader, VTKSequenceProcessor, DS1SolenoidalManifest
from src.io.persistence import TrajectoryHDF5Store, SolenoidalMetadataStore, DS2TrajectoryManifest
from src.io.result_writers import (
    EnsembleConsistencyJSONWriter, 
    VortexTimescaleCSVWriter, 
    VortexTimescaleMetrics,
    FTLECohortMarkdownWriter,
    FTLECohortStatistics
)
from src.io.validation import IntegrationConsistencyValidator, IntegrationMetrics
from src.integration.state import TrajectoryDataset
from src.experiments.config.parameters import PhysicsConfig
from src.physics.fields import VelocitySnapshot

def create_mock_vtk(path: Path, dimensions=(3, 3, 3), title="Mock VTK", time=1.0):
    nx, ny, nz = dimensions
    n_points = nx * ny * nz
    
    # Generate mock velocity data (float32, big-endian)
    # v = (x, y, z)
    v_data = np.zeros((n_points, 3), dtype='>f4')
    for i in range(n_points):
        v_data[i] = [float(i), float(i+1), float(i+2)]
    
    with open(path, 'wb') as f:
        f.write(f"# vtk DataFile Version 2.0\n".encode('ascii'))
        f.write(f"Snapshot at t= {time}\n".encode('ascii'))
        f.write(f"BINARY\n".encode('ascii'))
        f.write(f"DATASET STRUCTURED_POINTS\n".encode('ascii'))
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n".encode('ascii'))
        f.write(f"ORIGIN -0.5 -0.5 -0.5\n".encode('ascii'))
        f.write(f"SPACING 0.1 0.1 0.1\n".encode('ascii'))
        f.write(f"POINT_DATA {n_points}\n".encode('ascii'))
        f.write(f"VECTORS v float\n".encode('ascii'))
        f.write(v_data.tobytes())

class TestIOIntegration:
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield Path(tmpdirname)

    def test_vtk_reader_integration(self, temp_dir):
        header_parser = VTKHeaderParser()
        layout_extractor = VTKStructuredPointExtractor()
        field_extractor = VTKFieldExtractor()
        
        class SmallGridReader(SolenoidalSnapshotReader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._manifest = DS1SolenoidalManifest(grid_dimensions=(3, 3, 3))

        reader = SmallGridReader(header_parser, layout_extractor, field_extractor)
        vtk_path = temp_dir / "Turb.hydro_w.10000.vtk"
        create_mock_vtk(vtk_path, dimensions=(3, 3, 3), time=100.0)
        
        snapshot = reader.load_snapshot(vtk_path)
        assert isinstance(snapshot, VelocitySnapshot)
        assert snapshot.time == 100.0
        assert snapshot.grid_dimensions == (3, 3, 3)
        assert snapshot.vx[0, 0, 0] == 0.0
        assert snapshot.vy[0, 0, 0] == 1.0
        assert snapshot.vz[0, 0, 0] == 2.0

    def test_vtk_sequence_processor_integration(self, temp_dir):
        header_parser = VTKHeaderParser()
        layout_extractor = VTKStructuredPointExtractor()
        field_extractor = VTKFieldExtractor()
        
        class SmallGridReader(SolenoidalSnapshotReader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._manifest = DS1SolenoidalManifest(grid_dimensions=(3, 3, 3))

        reader = SmallGridReader(header_parser, layout_extractor, field_extractor)
        processor = VTKSequenceProcessor(reader)
        
        paths = []
        for i in range(3):
            p = temp_dir / f"Turb.hydro_w.{10000+i}.vtk"
            create_mock_vtk(p, dimensions=(3, 3, 3), time=100.0 + i*0.1)
            paths.append(p)
            
        snapshots = list(processor.stream_snapshots(paths))
        assert len(snapshots) == 3
        pairs = list(processor.stream_snapshot_pairs(paths))
        assert len(pairs) == 2

    def test_trajectory_persistence_integration(self, temp_dir):
        store = TrajectoryHDF5Store()
        positions = np.random.rand(5, 10, 3).astype('f4')
        velocities = np.random.rand(5, 10, 3).astype('f4')
        timestamps = np.linspace(0, 1, 5).astype('f4')
        data = TrajectoryDataset(positions, velocities, timestamps, 10)
        
        h5_path = temp_dir / "test.h5"
        store.save(data, h5_path)
        loaded_data = store.load(h5_path)
        np.testing.assert_array_almost_equal(loaded_data.positions, positions)
        
    def test_solenoidal_metadata_integration(self, temp_dir):
        store = SolenoidalMetadataStore()
        manifest = DS1SolenoidalManifest(grid_dimensions=(129, 129, 129), domain_size_L=1.0)
        json_path = temp_dir / "manifest.json"
        store.save_manifest(manifest, json_path)
        loaded_manifest = store.load_manifest(json_path)
        assert loaded_manifest == manifest

    def test_validation_and_result_writing_integration(self, temp_dir):
        validator = IntegrationConsistencyValidator()
        writer = EnsembleConsistencyJSONWriter()
        data = TrajectoryDataset(
            np.random.rand(5, 10, 3).astype('f4') * 0.5,
            np.random.rand(5, 10, 3).astype('f4'),
            np.linspace(0, 1, 5).astype('f4'),
            10
        )
        config = PhysicsConfig(domain_size_L=1.0)
        metrics = validator.validate_trajectories(data, config)
        json_path = temp_dir / "metrics.json"
        writer.write_statistics(json_path, metrics)
        assert json_path.exists()

    def test_vtk_reader_errors(self, temp_dir):
        header_parser = VTKHeaderParser()
        layout_extractor = VTKStructuredPointExtractor()
        field_extractor = VTKFieldExtractor()
        reader = SolenoidalSnapshotReader(header_parser, layout_extractor, field_extractor)
        with pytest.raises(FileNotFoundError):
            reader.load_snapshot(temp_dir / "non_existent.vtk")
        invalid_vtk = temp_dir / "invalid.vtk"
        with open(invalid_vtk, 'w') as f:
            f.write("Not a VTK file")
        with pytest.raises(ValueError, match="Invalid VTK header line 1"):
            reader.load_snapshot(invalid_vtk)
        bad_dim_vtk = temp_dir / "Turb.hydro_w.10001.vtk"
        create_mock_vtk(bad_dim_vtk, dimensions=(2, 2, 2))
        with pytest.raises(ValueError, match="Grid dimensions .* do not match manifest expectations"):
            reader.load_snapshot(bad_dim_vtk)

    def test_persistence_errors(self, temp_dir):
        store = TrajectoryHDF5Store()
        with pytest.raises(FileNotFoundError):
            store.load(temp_dir / "missing.h5")
        invalid_h5 = temp_dir / "invalid.h5"
        with open(invalid_h5, 'w') as f:
            f.write("not hdf5")
        with pytest.raises(IOError):
            store.load(invalid_h5)

    def test_result_writer_errors(self):
        writer = FTLECohortMarkdownWriter()
        with pytest.raises(KeyError, match="Missing required cohort data: Exit Events"):
            writer.write_comparison_table(Path("out.md"), {
                'Trapped': FTLECohortStatistics(0,0,0),
                'Free': FTLECohortStatistics(0,0,0)
            })
