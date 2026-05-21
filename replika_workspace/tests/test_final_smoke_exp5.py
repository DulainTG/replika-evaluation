import os
import subprocess
import pytest
from pathlib import Path

@pytest.mark.timeout(60)
def test_exp5_smoke_run(tmp_path):
    """
    Validates the canonical smoke-mode CLI path for EXP5.
    
    This test runs the EXP5 experiment in smoke mode and verifies that:
    1. The command completes successfully (exit code 0).
    2. The required artifact (markdown table) is created.
    """
    raw_data_dir = os.environ.get("RAW_DATA_DIR", "/raw_data")
    if not os.path.exists(raw_data_dir):
        pytest.skip(f"Raw data directory not found at {raw_data_dir}")

    output_dir = tmp_path / "smoke_out_exp5"
    
    # Canonical smoke command for EXP5
    cmd = [
        "python", "main.py", "EXP5",
        "--mode", "smoke",
        "--raw-data-dir", raw_data_dir,
        "--output-dir", str(output_dir)
    ]
    
    # Set PYTHONPATH to include installed packages
    env = os.environ.copy()
    if "/usr/local/lib/python3.12/site-packages" not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + ":/usr/local/lib/python3.12/site-packages").strip(":")

    # EXP5 might be slow due to memory-intensive snapshot loading.
    # We use a generous timeout but expect it to be reasonable.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("EXP5 smoke test timed out after 120 seconds")
    
    # Verify exit status
    assert result.returncode == 0, f"Smoke command failed with exit code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    
    # Verify artifact contract
    # EXP5 outputs are expected in <output_dir>/exp5/
    exp5_out_dir = output_dir / "exp5"
    assert exp5_out_dir.exists(), "EXP5 output directory was not created"
    
    expected_artifacts = [
        "ftle_chaos_analysis.md"
    ]
    
    for artifact in expected_artifacts:
        artifact_path = exp5_out_dir / artifact
        assert artifact_path.exists(), f"Required artifact {artifact} is missing from {exp5_out_dir}"
        assert artifact_path.stat().st_size > 0, f"Artifact {artifact} is empty"
        
        # Verify markdown content based on the EXP5 artifact contract
        with open(artifact_path, "r") as f:
            content = f.read()
            assert "# Experiment 5: FTLE and Chaos Analysis" in content
            assert "| Cohort | Mean FTLE | Std Dev | Pearson r |" in content
            assert "| Trapped Tracers |" in content
            assert "| Free Tracers |" in content
            assert "| Exit Events |" in content

    print("EXP5 smoke validation passed successfully.")

if __name__ == "__main__":
    # Allow running directly for quick validation
    pytest.main([__file__])
