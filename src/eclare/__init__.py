"""
eclare: Main package for the ECLARE project.
"""

# Import configuration loader
from config.config import ConfigLoader  # Absolute import
from pathlib import Path
import os

# Construct the path to the config file
current_file = Path(__file__).resolve()
current_dir = current_file.parent
config_file = current_dir.parent.parent / "config" / "config.yaml"

# Initialize ConfigLoader
config = ConfigLoader(config_file=Path(str(config_file)))

# Expose configuration parameters
ECLARE_ROOT     = config.get_directory('ECLARE_root')
OUTPATH         = config.get_directory('outpath')
DATAPATH        = config.get_directory('datapath')
NAMPATH         = os.path.join(ECLARE_ROOT, 'neural-additive-models-pt')

os.environ['ECLARE_root']   = str(ECLARE_ROOT)
os.environ['outpath']       = str(OUTPATH)
os.environ['datapath']      = str(DATAPATH)
os.environ['nampath']       = str(NAMPATH)

from .models import scTripletgrate, load_scTripletgrate_model
from .setup_utils import return_setup_func_from_dataset, mdd_setup
from .eval_utils import align_metrics, compute_mdd_eval_metrics, foscttm_moscot
from .data_utils import keep_CREs_and_adult_only, merge_major_cell_group, create_loaders, fetch_data_from_loaders
from .losses_and_distances_utils import clip_loss, cosine_distance, clip_loss_split_by_ct
from .run_utils import run_scTripletgrate
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
    'ECLARE_ROOT',
    'OUTPATH',
    'DATAPATH',
    'NAMPATH',
]