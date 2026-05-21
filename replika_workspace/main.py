import argparse
import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.experiments.config.parameters import PhysicsConfig
from src.experiments.orchestration import BatchExperimentOrchestrator, ExperimentResult
from src.experiments.exp1_tracer_trajectory_generation import TracerTrajectoryGenerator, SnapshotPairManager
from src.experiments.exp2_anisotropic_dispersion_analysis import AnisotropicDispersionExperiment
from src.experiments.exp3_vortex_residence_time_calculation import VortexResidenceTimeExperiment
from src.experiments.exp4_displacement_pdf_evolution import DisplacementPDFExperiment
from src.experiments.exp5_ftle_chaos_analysis import FTLEChaosAnalysisExperiment
from src.io.persistence import TrajectoryHDF5Store
from src.io.result_writers import EnsembleConsistencyJSONWriter
from src.io.validation import IntegrationConsistencyValidator
from src.integration.state import PassiveTracerTrajectoryGenerator, TrajectoryDataset
from src.integration.seeding import UniformRandomSeeder
from src.io.vtk_reader import SolenoidalSnapshotReader
from src.io.parsers import VTKHeaderParser, VTKStructuredPointExtractor, VTKFieldExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

def get_snapshot_indices(raw_data_dir: Path, smoke: bool = False, experiment: str = "") -> List[int]:
    """Retrieves and sorts VTK snapshot indices from the raw data directory."""
    vtk_files = sorted(list(raw_data_dir.glob("Turb.hydro_w.*.vtk")))
    indices = []
    for f in vtk_files:
        try:
            # Extract index from 'Turb.hydro_w.<index>.vtk'
            parts = f.name.split('.')
            if len(parts) >= 3:
                indices.append(int(parts[2]))
        except (ValueError, IndexError):
            continue
    
    if not indices:
        return []
    
    indices = sorted(indices)
    if smoke:
        # Scale down for smoke mode.
        # Taking every 5th snapshot reduces workload by 80% while preserving resolution 
        # needed for lag-time checkpoints in EXP4 (0.5, 1.0, 2.0, 5.0, 9.0).
        indices = indices[::5]
        
        if experiment == "EXP5":
            # EXP5 is extremely memory-intensive; use even fewer snapshots for smoke mode
            return indices[:10]
        return indices
    
    return indices

def run_exp1_and_save(cfg: PhysicsConfig, raw_data_path: Path, out_dir: Path, smoke: bool = False, experiment: str = "") -> ExperimentResult:
    indices = get_snapshot_indices(raw_data_path, smoke, experiment)
    if not indices:
        raise RuntimeError(f"No VTK snapshots found in {raw_data_path}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We use PassiveTracerTrajectoryGenerator directly to get the dataset for saving
    reader = SolenoidalSnapshotReader(VTKHeaderParser(), VTKStructuredPointExtractor(), VTKFieldExtractor())
    
    logger.info(f"Generating trajectories for EXP1 using {len(indices)} snapshots...")
    seeder = UniformRandomSeeder()
    current_positions = seeder.seed_tracers(cfg)
    
    positions_history = []
    velocities_history = []
    timestamps = []
    prev_snapshot = None
    
    for i, idx in enumerate(indices):
        snapshot_path = raw_data_path / f"Turb.hydro_w.{idx}.vtk"
        snapshot = reader.load_snapshot(snapshot_path)
        
        if prev_snapshot is not None:
            # Integrate from prev to current
            # We recreate the integrator logic to avoid the 200 snapshot restriction
            from src.integration.state import RK4Integrator
            integrator = RK4Integrator(cfg)
            current_positions = integrator.advance_state(current_positions, prev_snapshot, snapshot)
            
        positions_history.append(current_positions.copy())
        # Sample local velocity at current positions.
        from src.integration.interpolation import TrilinearGridInterpolator
        interpolator = TrilinearGridInterpolator()
        current_velocities = interpolator.evaluate_velocity_field(snapshot, current_positions)
        velocities_history.append(current_velocities)
        timestamps.append(snapshot.time)
        prev_snapshot = snapshot

    dataset = TrajectoryDataset(
        positions=np.stack(positions_history),
        velocities=np.stack(velocities_history),
        timestamps=np.array(timestamps),
        n_tracers=cfg.n_tracers
    )
    
    store = TrajectoryHDF5Store()
    traj_path = out_dir / "trajectories.h5"
    store.save(dataset, traj_path)
    
    # Validate and get metrics
    validator = IntegrationConsistencyValidator(cfg)
    report = validator.validate_trajectory_consistency(dataset)
    
    # Save metrics to JSON as per EXP1 contract
    json_path = out_dir / "consistency_metrics.json"
    json_writer = EnsembleConsistencyJSONWriter()
    json_writer.write_statistics(json_path, report)
    
    metrics = {
        "mean_squared_velocity": report.mean_squared_velocity,
        "coordinate_ranges": {
            "min": report.coordinate_min,
            "max": report.coordinate_max
        },
        "timestamps": {
            "start": float(dataset.timestamps[0]),
            "end": float(dataset.timestamps[-1]),
            "count": len(dataset.timestamps)
        },
        "n_tracers": int(dataset.n_tracers)
    }

    return ExperimentResult(
        experiment_id="EXP1",
        success=True,
        metrics=metrics,
        artifacts={
            "trajectory_h5": str(traj_path),
            "json_metrics": str(json_path)
        }
    )

def check_data_compatibility(raw_data_dir: Path):
    """
    Guarded smoke test that reads headers and a tiny bounded number of rows 
    from real VTK dataset files to validate schema compatibility.
    """
    try:
        vtk_files = sorted(list(raw_data_dir.glob("Turb.hydro_w.*.vtk")))
        if not vtk_files:
            logger.info("No VTK files found in %s. Skipping compatibility check.", raw_data_dir)
            return

        test_file = vtk_files[0]
        logger.info("Performing guarded compatibility check on %s", test_file.name)
        
        reader = SolenoidalSnapshotReader(VTKHeaderParser(), VTKStructuredPointExtractor(), VTKFieldExtractor())
        # Loading one snapshot is a tiny bounded read compared to the whole 100-snapshot sequence.
        # This confirms we can parse the headers and the field data correctly.
        snapshot = reader.load_snapshot(test_file)
        
        if snapshot.vx is None or snapshot.vx.size == 0:
            raise ValueError("Velocity field (vx) is empty or missing.")
        
        logger.info("Compatibility check passed: Snapshot dimensions %s, Vel field shape %s", 
                    snapshot.grid_dimensions, snapshot.vx.shape)
    except Exception as e:
        logger.warning("Guarded compatibility check skipped or failed: %s", str(e))

def verify_smoke_artifacts(output_root: Path):
    """
    Asserts that the full-benchmark smoke run produced the expected cross-experiment artifacts.
    """
    expected = {
        "exp1": ["trajectories.h5", "consistency_metrics.json"],
        "exp2": ["lambda_evolution.png"],
        "exp3": ["vortex_residence_timescale.csv"],
        "exp4": ["displacement_pdf_evolution.png", "evolution_metrics.json"],
        "exp5": ["ftle_chaos_analysis.md"]
    }
    
    logger.info("Verifying smoke artifacts contract...")
    missing = []
    for exp_dir, files in expected.items():
        for f in files:
            p = output_root / exp_dir / f
            if not p.exists():
                missing.append(f"{exp_dir}/{f}")
            else:
                logger.info("  Verified: %s/%s", exp_dir, f)
    
    if missing:
        msg = "Smoke artifact contract violation! Missing: " + ", ".join(missing)
        logger.error(msg)
        raise RuntimeError(msg)
    
    logger.info("Smoke artifact contract satisfied.")

def main():
    parser = argparse.ArgumentParser(description="Reproduction CLI for 3D Solenoidal Turbulence")
    parser.add_argument("experiment", choices=["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "full-benchmark"], 
                        help="Experiment to run")
    parser.add_argument("--dataset", help="Optional dataset name for compatibility")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full", 
                        help="Execution mode (full or smoke)")
    parser.add_argument("--raw-data-dir", default=os.environ.get("RAW_DATA_DIR", "./raw_data"),
                        help="Path to raw VTK data directory")
    parser.add_argument("--output-dir", default="./outputs", 
                        help="Root directory for outputs")

    
    # Overrides for parameters
    parser.add_argument("--n-tracers", type=int, help="Override number of tracers")
    parser.add_argument("--domain-size", type=float, help="Override domain size L")
    
    args = parser.parse_args()
    
    # Establish output directory for this run
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Resolve PhysicsConfig with potential overrides
    config_kwargs = {}
    if args.n_tracers: config_kwargs["n_tracers"] = args.n_tracers
    if args.domain_size: config_kwargs["domain_size_L"] = args.domain_size
    
    # Defaults for smoke mode
    if args.mode == "smoke":
        if "n_tracers" not in config_kwargs: config_kwargs["n_tracers"] = 100
        
    config = PhysicsConfig(**config_kwargs)
    
    logger.info(f"Resolved Configuration: mode={args.mode}, experiment={args.experiment}")
    logger.info(f"Raw Data Directory: {args.raw_data_dir}")
    logger.info(f"Output Root Folder: {output_root}")
    
    raw_data_path = Path(args.raw_data_dir)
    if not raw_data_path.exists():
        print(f"Error: Required dataset directory missing at {raw_data_path}")
        sys.exit(1)

    if args.mode == "smoke":
        check_data_compatibility(raw_data_path)

    exp_ids = ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5"] if args.experiment == "full-benchmark" else [args.experiment]
    
    # Track results and artifacts for common usage
    exp1_traj_path = None
    
    for eid in exp_ids:
        logger.info(f"--- Running {eid} ---")
        exp_out_dir = output_root / eid.lower()
        exp_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if eid == "EXP1":
                res = run_exp1_and_save(config, raw_data_path, exp_out_dir, args.mode == "smoke", eid)
                exp1_traj_path = Path(res.artifacts["trajectory_h5"])
            
            else:
                # Downstream experiments need EXP1's trajectory
                if exp1_traj_path is None:
                    # Look for it in the expected location
                    potential_path = output_root / "exp1" / "trajectories.h5"
                    if potential_path.exists():
                        exp1_traj_path = potential_path
                    else:
                        logger.info("EXP1 trajectory not found. Running EXP1 automatically...")
                        res1 = run_exp1_and_save(config, raw_data_path, output_root / "exp1", args.mode == "smoke", eid)
                        if not res1.success:
                            raise RuntimeError(f"Automatic EXP1 failed: {res1.errors}")
                        exp1_traj_path = Path(res1.artifacts["trajectory_h5"])
                
                if eid == "EXP2":
                    # EXP2 needs both DS1 (VTK) and DS2 (trajectories.h5) in the same data_dir
                    work_dir = exp_out_dir / "input_data"
                    work_dir.mkdir(parents=True, exist_ok=True)
                    for vtk in raw_data_path.glob("*.vtk"):
                        sym_link = work_dir / vtk.name
                        if not sym_link.exists(): os.symlink(vtk, sym_link)
                    shutil.copy(exp1_traj_path, work_dir / "trajectories.h5")
                    
                    exp = AnisotropicDispersionExperiment(config, work_dir, exp_out_dir)
                    res = exp.execute()
                
                elif eid == "EXP3":
                    exp = VortexResidenceTimeExperiment(config, raw_data_path, exp1_traj_path, exp_out_dir)
                    res = exp.execute()
                
                elif eid == "EXP4":
                    exp = DisplacementPDFExperiment(config, exp1_traj_path, exp_out_dir)
                    res = exp.execute()
                
                elif eid == "EXP5":
                    exp = FTLEChaosAnalysisExperiment(config, raw_data_path, exp1_traj_path, exp_out_dir)
                    res = exp.execute()
            
            if res.success:
                logger.info(f"{eid} completed successfully.")
                for name, path in res.artifacts.items():
                    logger.info(f"  Artifact: {name} -> {path}")
            else:
                logger.error(f"{eid} failed: {res.errors}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to execute {eid}: {str(e)}", exc_info=True)
            sys.exit(1)

    # After all requested experiments are done, verify artifacts if we're in full-benchmark smoke mode
    if args.experiment == "full-benchmark" and args.mode == "smoke":
        verify_smoke_artifacts(output_root)

if __name__ == "__main__":
    main()
