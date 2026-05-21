import os
import subprocess
import pytest
from pathlib import Path
import json

# Add system site-packages to sys.path in case they are missing
import sys
sys.path.append("/usr/local/lib/python3.12/site-packages")

try:
    import h5py
except ImportError:
    h5py = None

def get_python_executable():
    return "/usr/local/bin/python3"

@pytest.mark.timeout(300)
def test_full_benchmark_smoke(tmp_path):
    """
    Validate the canonical 'python main.py full-benchmark --mode smoke' command.
    Checks if all required artifacts are produced.
    """
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    if not os.path.exists(raw_data_dir):
        pytest.skip("RAW_DATA_DIR not available")

    output_dir = tmp_path / "outputs"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:/usr/local/lib/python3.12/site-packages"
    
    cmd = [
        get_python_executable(), "main.py", "full-benchmark",
        "--mode", "smoke",
        "--raw-data-dir", raw_data_dir,
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    
    assert result.returncode == 0, f"Command failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check artifacts
    expected_artifacts = [
        "exp1/trajectories.h5",
        "exp1/consistency_metrics.json",
        "exp2/lambda_evolution.png",
        "exp3/vortex_residence_timescale.csv",
        "exp4/displacement_pdf_evolution.png",
        "exp4/evolution_metrics.json",
        "exp5/ftle_chaos_analysis.md"
    ]
    
    for artifact in expected_artifacts:
        path = output_dir / artifact
        assert path.exists(), f"Missing artifact: {artifact}"
        assert path.stat().st_size > 0, f"Artifact is empty: {artifact}"

    # Specifically check EXP1 trajectory contract
    if h5py:
        with h5py.File(output_dir / "exp1" / "trajectories.h5", "r") as f:
            assert "positions" in f
            assert "velocities" in f
            assert "timestamps" in f
            # n_tracers is 100 in smoke mode as per main.py
            assert f["positions"].shape[1] == 100 

@pytest.mark.timeout(30)
def test_raw_data_schema_compatibility():
    """
    Guarded smoke test that reads only minimal data from real VTK files to validate schema compatibility.
    """
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    if not os.path.exists(raw_data_dir):
        pytest.skip("RAW_DATA_DIR not available")
        
    vtk_files = sorted(list(Path(raw_data_dir).glob("*.vtk")))
    if not vtk_files:
        pytest.skip("No VTK files found")
        
    from src.io.vtk_reader import SolenoidalSnapshotReader
    from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor
    
    # We want to read only the header info
    header_parser = VTKHeaderParser()
    with open(vtk_files[0], 'rb') as f:
        header = header_parser.parse(f)
        
    assert header is not None
    assert header.version == "2.0"
    assert header.encoding == "BINARY"

    # Also check layout
    from src.io.parsers import VTKStructuredPointExtractor
    layout_extractor = VTKStructuredPointExtractor()
    with open(vtk_files[0], 'rb') as f:
        # We need to skip the header to get to the layout or just let the extractor handle it
        # Actually parse() moves the stream to after the header.
        # But extractor also searches from current position.
        layout = layout_extractor.extract_layout(f)
    
    assert layout.dimensions == (129, 129, 129)

@pytest.mark.timeout(120)
@pytest.mark.parametrize("exp", ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5"])
def test_individual_experiments_smoke(tmp_path, exp):
    """
    CLI smoke test for each individual experiment command in smoke mode.
    """
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    if not os.path.exists(raw_data_dir):
        pytest.skip("RAW_DATA_DIR not available")

    output_dir = tmp_path / f"outputs_{exp}"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:/usr/local/lib/python3.12/site-packages"

    cmd = [
        get_python_executable(), "main.py", exp,
        "--mode", "smoke",
        "--raw-data-dir", raw_data_dir,
        "--output-dir", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, f"{exp} failed with output:\n{result.stdout}\n{result.stderr}"
    
    # Check primary artifacts for each
    artifact_map = {
        "EXP1": ["exp1/trajectories.h5", "exp1/consistency_metrics.json"],
        "EXP2": ["exp2/lambda_evolution.png"],
        "EXP3": ["exp3/vortex_residence_timescale.csv"],
        "EXP4": ["exp4/displacement_pdf_evolution.png", "exp4/evolution_metrics.json"],
        "EXP5": ["exp5/ftle_chaos_analysis.md"]
    }
    
    for artifact in artifact_map[exp]:
        path = output_dir / artifact
        assert path.exists(), f"Missing artifact for {exp}: {artifact}"
