# Shared init code for noteoboks
# Usage: from notebook_init import *

import torch
import numpy as np
from os import makedirs
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pickle

import sys
sys.path.insert(0, '..')
from models import get_instrumented_model, get_model
from notebook_utils import create_strip, create_strip_centered, prettify_name, save_frames, pad_frames
from config import Config
from decomposition import get_or_compute

torch.autograd.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if has_gpu else 'cpu')
inst = None