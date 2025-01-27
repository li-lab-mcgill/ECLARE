import os
import socket
hostname = socket.gethostname()

if hostname.startswith('narval'):
    os.environ['machine'] = 'narval'
elif hostname == 'MBP-de-Dylan.lan':
    os.environ['machine'] = 'local'

'''
from .triplet_utils import find_mnn_idxs
from .data_utils import *
from .losses_and_distances_utils import *
from .models import *
from .run_utils import *
from .setup_utils import *
from .tune_utils import *
'''