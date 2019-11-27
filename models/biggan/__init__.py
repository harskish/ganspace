from pathlib import Path
import sys

module_path = Path(__file__).parent / 'pytorch_biggan'
sys.path.append(str(module_path.resolve()))
from pytorch_pretrained_biggan import *
from pytorch_pretrained_biggan.model import GenBlock
from pytorch_pretrained_biggan.file_utils import http_get, s3_get