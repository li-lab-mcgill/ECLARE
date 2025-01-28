"""
eclare: Main package for the ECLARE project.
"""

from .models import load_scTripletgrate_model
from .setup_utils import return_setup_func_from_dataset, mdd_setup, merged_dataset_setup
from .eval_utils import align_metrics
from .data_utils import fetch_data_from_loaders
from .losses_and_distances_utils import clip_loss

from .run_utils import run_scTripletgrate
from .tune_utils import study_summary

# Import configuration loader
from config.config import ConfigLoader  # Absolute import

# Initialize ConfigLoader
config = ConfigLoader(config_path="config/config.yaml")

# Expose configuration parameters
ECLARE_ROOT = config.get('ECLARE_root')
OUTPATH = config.get('OUTPATH')
DATAPATH = config.get('DATAPATH')

# Define package version
__version__ = "0.1.0"

# Define public API
__all__ = [
    'load_scTripletgrate_model',
    'return_setup_func_from_dataset',
    'mdd_setup',
    'align_metrics',
    'fetch_data_from_loaders',
    'clip_loss',
    'run_scTripletgrate',
    'study_summary',
    'ECLARE_ROOT',
    'OUTPATH',
    'DATAPATH',
]