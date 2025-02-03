"""
eclare: Main package for the ECLARE project.
"""

from pathlib import Path
import os
import sys
import subprocess

def initialize_environment():

    ## Set ECLARE_ROOT to the root directory of the ECLARE project.
    ECLARE_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL)
    ECLARE_ROOT = ECLARE_ROOT.strip().decode()
    ECLARE_ROOT = Path(ECLARE_ROOT)
    print(f"ECLARE_ROOT detected: {ECLARE_ROOT}")

    ## Import ECLARE_ROOT env variable.
    if ECLARE_ROOT is None:
        raise KeyError(
            "ECLARE_ROOT environment variable is not set. Please set it to the root directory of the ECLARE project. "
            "Example, from terminal: export ECLARE_ROOT=/path/to/ECLARE"
        )

    # Construct the path to the config file
    config_file = ECLARE_ROOT / "config" / "config.yaml"

    # Import configuration loader
    sys.path.insert(0, str(ECLARE_ROOT))  # Add ECLARE_ROOT to searchable paths
    from config.config import ConfigLoader  # Absolute import

    # Initialize ConfigLoader
    config = ConfigLoader(config_file=Path(str(config_file)))

    # Expose configuration parameters
    ECLARE_ROOT     = config.get_directory('ECLARE_ROOT') # redundant, but useful for clarity
    OUTPATH         = config.get_directory('OUTPATH')
    DATAPATH        = config.get_directory('DATAPATH')
    NAMPATH         = os.path.join(ECLARE_ROOT, 'neural-additive-models-pt')

    os.environ['ECLARE_ROOT']   = str(ECLARE_ROOT)
    os.environ['OUTPATH']       = str(OUTPATH)
    os.environ['DATAPATH']      = str(DATAPATH)
    os.environ['NAMPATH']       = str(NAMPATH)

    return ECLARE_ROOT, OUTPATH, DATAPATH, NAMPATH

ECLARE_ROOT, OUTPATH, DATAPATH, NAMPATH = initialize_environment()

from .models import CLIP, load_CLIP_model
from .setup_utils import return_setup_func_from_dataset, mdd_setup, teachers_setup, merged_dataset_setup
from .eval_utils import align_metrics, compute_mdd_eval_metrics, foscttm_moscot
from .data_utils import keep_CREs_and_adult_only, merge_major_cell_group, create_loaders, fetch_data_from_loaders
from .losses_and_distances_utils import clip_loss, cosine_distance, clip_loss_split_by_ct, Knowledge_distillation_fn, ct_losses
from .run_utils import run_CLIP, save_latents
from .tune_utils import study_summary
from .triplet_utils import get_triplet_loss

# Define package version
__version__ = "0.1.0"

# Define public API
__all__ = [
    'CLIP',
    'load_CLIP_model',
    'return_setup_func_from_dataset',
    'mdd_setup',
    'align_metrics',
    'keep_CREs_and_adult_only',
    'merge_major_cell_group',
    'create_loaders',
    'fetch_data_from_loaders',
    'clip_loss',
    'run_CLIP',
    'study_summary',
    'cosine_distance',
    'clip_loss_split_by_ct',
    'get_triplet_loss',
    'compute_mdd_eval_metrics',
    'foscttm_moscot',
    'Knowledge_distillation_fn',
    'ct_losses',
    'save_latents',
    'teachers_setup',
    'ECLARE_ROOT',
    'OUTPATH',
    'DATAPATH',
    'NAMPATH',
]