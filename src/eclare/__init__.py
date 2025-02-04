"""
eclare: Main package for the ECLARE project.
"""

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
    'teachers_setup'
]