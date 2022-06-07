import sys
import os
import shutil
import glob
import platform
from pathlib import Path

current_path = os.getcwd()

module_path = Path(__file__).parent / 'stylegan2-ada-pytorch'
sys.path.append(str(module_path.resolve()))
os.chdir(module_path)

import generate
import legacy
import dnnlib

os.chdir(current_path)
