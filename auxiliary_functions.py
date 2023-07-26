from setup import SEED, GPU
from os import environ
# fixing hash randomisation seed
environ['PYTHONHASHSEED']=str(SEED) 
from tensorflow import random as tf_random
from tensorflow import config
import random
from numpy import random as np_random
from shutil import copy


def reset_random_seeds():
    """Auxiliary function re-setting the random seeds.

    Necessary to ensure determinism.
    """
    environ['PYTHONHASHSEED']=str(SEED)
    tf_random.set_seed(SEED)
    np_random.seed(SEED)
    random.seed(SEED)


def determine_device():
    """Auxiliary function setting up the device for TensorFlow to utilise.

    Returns:
    --------
       active_device (str): device choice for TensorFlow to uutilise
    """
    if GPU:
        # Checking that GPU processing is available
        assert len(config.list_physical_devices('GPU')) > 0, 'No CUDA-enabled GPUs found.'
        # Setting the first GPU as active device
        active_device = '/GPU:0'
    else:
        active_device = '/CPU:0'
        # Hiding available GPUs (as TensorFlow has been reported to still 
        # utilise them, even when CPU processing has been specified)
        environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return active_device


def save_settings(foldername):
    """Auxiliary function copying the settings file into the results folder.
    
    Parameters:
    ------------
    Foldername (str): name of the destination folder
    """
    source_path = './setup.py'
    destination_path = foldername + 'settings.py'
    copy(source_path, destination_path, follow_symlinks=True)
