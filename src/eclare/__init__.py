"""
eclare: Main package for the ECLARE project.
"""

import os
import sys

def set_env_variables(config_path='config'):
    """
    Set essential environment variables for ECLARE.
    This function should be called before importing any other ECLARE modules.
    """
    # Check if environment variables are already set
    eclare_root = os.environ.get('ECLARE_ROOT')
    outpath = os.environ.get('OUTPATH')
    datapath = os.environ.get('DATAPATH')

    # Print status of environment variables
    if all([eclare_root, outpath, datapath]):
        print(f"Environment variables already set:")
        print(f"ECLARE_ROOT: {eclare_root}")
        print(f"OUTPATH: {outpath}")
        print(f"DATAPATH: {datapath}")
    else:
        print(f"Missing environment variables")

        sys.path.insert(0, config_path)

        from export_env_variables import export_env_variables
        export_env_variables(config_path)

# Define package version
__version__ = "0.1.0"

# Define public API
__all__ = [
    'set_env_variables',
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
