"""
eclare: Main package for the ECLARE project.
"""

from pathlib import Path
import os
import sys

## Import ECLARE_root env variable.
try:
    ECLARE_ROOT = Path(os.environ['ECLARE_root'])
    print(f"ECLARE_root: {ECLARE_ROOT}")
except KeyError:
    raise KeyError(
        "ECLARE_root environment variable is not set. Please set it to the root directory of the ECLARE project. "
        "Example, from terminal: export ECLARE_root=/path/to/ECLARE"
    )

# Construct the path to the config file
config_file = ECLARE_ROOT / "config" / "config.yaml"

# Import configuration loader
sys.path.insert(0, str(ECLARE_ROOT))  # Add ECLARE_root to searchable paths
from config.config import ConfigLoader  # Absolute import

# Initialize ConfigLoader
config = ConfigLoader(config_file=Path(str(config_file)))

# Expose configuration parameters
ECLARE_ROOT     = config.get_directory('ECLARE_root') # redundant, but useful for clarity
OUTPATH         = config.get_directory('outpath')
DATAPATH        = config.get_directory('datapath')
NAMPATH         = os.path.join(ECLARE_ROOT, 'neural-additive-models-pt')

os.environ['ECLARE_root']   = str(ECLARE_ROOT)
os.environ['outpath']       = str(OUTPATH)
os.environ['datapath']      = str(DATAPATH)
os.environ['nampath']       = str(NAMPATH)

from .models import scTripletgrate, load_scTripletgrate_model
from .setup_utils import return_setup_func_from_dataset, mdd_setup, teachers_setup, merged_dataset_setup
from .eval_utils import align_metrics, compute_mdd_eval_metrics, foscttm_moscot
from .data_utils import keep_CREs_and_adult_only, merge_major_cell_group, create_loaders, fetch_data_from_loaders
from .losses_and_distances_utils import clip_loss, cosine_distance, clip_loss_split_by_ct, Knowledge_distillation_fn, ct_losses
from .run_utils import run_scTripletgrate, save_latents
from .tune_utils import study_summary
from .triplet_utils import get_triplet_loss

# Define package version
__version__ = "0.1.0"

# Define public API
__all__ = [
    'scTripletgrate',
    'load_scTripletgrate_model',
    'return_setup_func_from_dataset',
    'mdd_setup',
    'align_metrics',
    'keep_CREs_and_adult_only',
    'merge_major_cell_group',
    'create_loaders',
    'fetch_data_from_loaders',
    'clip_loss',
    'run_scTripletgrate',
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