import os
import socket
import sys
hostname = socket.gethostname()

if hostname.startswith('narval'):
    os.environ['machine'] = 'narval'
elif hostname == 'MBP-de-Dylan.lan':
    os.environ['machine'] = 'local'

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

'''
from .triplet_utils import find_mnn_idxs
from .data_utils import *
from .losses_and_distances_utils import *
from .models import *
from .run_utils import *
from .setup_utils import *
from .tune_utils import *
'''