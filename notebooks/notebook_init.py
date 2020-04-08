# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

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