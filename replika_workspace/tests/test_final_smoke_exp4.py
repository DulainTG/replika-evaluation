import os
import subprocess
import pytest
from pathlib import Path

@pytest.mark.timeout(60)
def test_exp4_smoke_run(tmp_path):
    """
    Validates the canonical smoke-mode CLI path for EXP4.
    
    This test runs the EXP4 experiment in smoke mode and verifies that:
    1. The command completes successfully (exit code 0).
    2. The required artifacts (plot and metrics) are created.
    """
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    if not os.path.exists(raw_data_dir):
        pytest.skip(f"Raw data directory not found at {raw_data_dir}")

    output_dir = tmp_path / "smoke_out_exp4"
    
    # Canonical smoke command for EXP4
    cmd = [
        "python", "main.py", "EXP4",
        "--mode", "smoke",
        "--raw-data-dir", raw_data_dir,
        "--output-dir", str(output_dir)
    ]
    
    # Set PYTHONPATH to include installed packages if necessary
    env = os.environ.copy()
    if "/usr/local/lib/python3.12/site-packages" not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + ":/usr/local/lib/python3.12/site-packages").strip(":")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    
    # Verify exit status
    assert result.returncode == 0, f"Smoke command failed with exit code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    
    # Verify artifact contract
    # EXP4 outputs are expected in <output_dir>/exp4/
    exp4_out_dir = output_dir / "exp4"
    assert exp4_out_dir.exists(), "EXP4 output directory was not created"
    
    expected_artifacts = [
        "displacement_pdf_evolution.png",
        "evolution_metrics.json"
    ]
    
    for artifact in expected_artifacts:
        artifact_path = exp4_out_dir / artifact
        assert artifact_path.exists(), f"Required artifact {artifact} is missing from {exp4_out_dir}"
        assert artifact_path.stat().st_size > 0, f"Artifact {artifact} is empty"

    print("EXP4 smoke validation passed successfully.")

if __name__ == "__main__":
    # Allow running directly for quick validation
    pytest.main([__file__])
