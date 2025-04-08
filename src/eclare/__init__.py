"""
eclare: Main package for the ECLARE project.
"""

from .models import CLIP, load_CLIP_model
from .setup_utils import return_setup_func_from_dataset, mdd_setup, teachers_setup, merged_dataset_setup
from .eval_utils import align_metrics, compute_mdd_eval_metrics, foscttm_moscot
from .data_utils import keep_CREs_and_adult_only, merge_major_cell_group, create_loaders, fetch_data_from_loaders
from .losses_and_distances_utils import clip_loss, cosine_distance, clip_loss_split_by_ct, Knowledge_distillation_fn
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

# Suppress warnings
import warnings
from numba import NumbaDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=NumbaDeprecationWarning,
    module="umap.umap_"
)
'''
/home/mcb/users/dmannk/.conda/envs/eclare_env/lib/python3.9/site-packages/umap/umap_.py:660: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  @numba.jit()
'''

warnings.filterwarnings("ignore", category=UserWarning, message=".*Choices for a categorical distribution should be a tuple.*")
'''
/home/mcb/users/dmannk/.conda/envs/eclare_env/lib/python3.9/site-packages/optuna/distributions.py:524: UserWarning: Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains <PrintableLambda: PoissonNLLLoss> which is of type PrintableLambda.
'''
