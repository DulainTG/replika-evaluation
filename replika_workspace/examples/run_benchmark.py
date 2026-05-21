"""
Example: Running the Full Reproduction Benchmark.

This script demonstrates how to use the BatchExperimentOrchestrator to run
the entire reproduction pipeline (EXP1 to EXP5) programmatically.
"""

import os
import sys
from pathlib import Path
import logging
import shutil

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments.config.parameters import PhysicsConfig
from src.experiments.orchestration import BatchExperimentOrchestrator
from src.experiments.exp1_tracer_trajectory_generation import TracerTrajectoryGenerator
from src.experiments.exp2_anisotropic_dispersion_analysis import AnisotropicDispersionExperiment
from src.experiments.exp3_vortex_residence_time_calculation import VortexResidenceTimeExperiment
from src.experiments.exp4_displacement_pdf_evolution import DisplacementPDFExperiment
from src.experiments.exp5_ftle_chaos_analysis import FTLEChaosAnalysisExperiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_benchmark_example():
    # 1. Setup paths
    raw_data_dir = Path(os.environ.get("RAW_DATA_DIR", "/raw_data"))
    output_root = Path("./benchmark_example_outputs")
    
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    # 2. Configuration
    # Using 'smoke' style parameters for demonstration speed
    config = PhysicsConfig(n_tracers=100)
    
    # 3. Setup Orchestrator
    orchestrator = BatchExperimentOrchestrator(output_root)
    
    # 4. Prepare Experiments
    # Note: In the real main.py, some experiments might need specific data preparation.
    # Here we show how to instantiate them.
    
    # For simplicity in this example, we assume we want to run all 5.
    # However, because EXP2-5 depend on the output of EXP1, 
    # we need to be careful about where they look for trajectories.h5.
    
    # Let's define the experiments.
    # EXP1
    exp1_out = output_root / "exp1"
    # In this repo, EXP1 is usually run via main.run_exp1_and_save.
    # We'll use a simplified approach here or call the function from main if possible.
    # Actually, let's keep it consistent with main.py's logic.
    
    import main
    logger.info("Starting Batch Execution...")
    
    # Note: BatchExperimentOrchestrator expects a sequence of Experiment objects.
    # In main.py, the loop handles the dependencies.
    # Here we'll demonstrate the programmatic flow.
    
    try:
        # Run EXP1
        logger.info("Running EXP1...")
        res1 = main.run_exp1_and_save(config, raw_data_dir, exp1_out)
        traj_path = Path(res1.artifacts["trajectory_h5"])
        
        # Run EXP2
        logger.info("Running EXP2...")
        exp2_out = output_root / "exp2"
        exp2_work = exp2_out / "input"
        exp2_work.mkdir(parents=True)
        shutil.copy(traj_path, exp2_work / "trajectories.h5")
        for vtk in sorted(list(raw_data_dir.glob("*.vtk")))[:10]: # minimal for example
            os.symlink(vtk, exp2_work / vtk.name)
        exp2 = AnisotropicDispersionExperiment(config, exp2_work, exp2_out)
        res2 = exp2.execute()
        
        # Run EXP3
        logger.info("Running EXP3...")
        exp3_out = output_root / "exp3"
        exp3 = VortexResidenceTimeExperiment(config, raw_data_dir, traj_path, exp3_out)
        res3 = exp3.execute()
        
        # Run EXP4
        logger.info("Running EXP4...")
        exp4_out = output_root / "exp4"
        exp4 = DisplacementPDFExperiment(config, traj_path, exp4_out)
        res4 = exp4.execute()
        
        # Run EXP5
        logger.info("Running EXP5...")
        exp5_out = output_root / "exp5"
        exp5 = FTLEChaosAnalysisExperiment(config, raw_data_dir, traj_path, exp5_out)
        res5 = exp5.execute()
        
        logger.info("Batch execution finished successfully!")
        
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        raise

if __name__ == "__main__":
    run_benchmark_example()
