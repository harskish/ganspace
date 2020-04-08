# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import torch, numpy as np
from types import SimpleNamespace
import itertools

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import get_model
from config import Config

torch.backends.cudnn.benchmark = True
has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if has_gpu else 'cpu')
B = 2 # test batch support

models = [
    ('BigGAN-128', 'husky'),
    ('BigGAN-256', 'husky'),
    ('BigGAN-512', 'husky'),
    ('StyleGAN', 'ffhq'),
    ('StyleGAN2', 'ffhq'),
]

for model_name, classname in models:
    with torch.no_grad():
        model = get_model(model_name, classname, device).to(device)
        print(f'Testing {model_name}-{classname}', end='')

        n_latents = model.get_max_latents()
        assert n_latents > 1, 'Model reports max_latents=1'
    
        #if hasattr(model, 'use_w'):
        #    model.use_w()

        seed = 1234
        torch.manual_seed(seed)
        np.random.seed(seed)
        latents = [model.sample_latent(B, seed=seed) for _ in range(10)]

        # Test that partial-forward supports layerwise latent inputs
        try:
            layer_name, _ = list(model.named_modules())[-1]
            _ = model.partial_forward(n_latents*[latents[0]], layer_name)
        except Exception as e:
            print('Error:', e)
            raise RuntimeError(f"{model_name} partial forward doesn't support layerwise latent!")

        # Test that layerwise and single give same result
        for z in latents:
            torch.manual_seed(0)
            np.random.seed(0)
            out1 = model.forward(z)
            
            torch.manual_seed(0)
            np.random.seed(0)
            out2 = model.forward(n_latents*[z])
        
            dist_rel = (torch.abs(out1 - out2).sum() / out1.sum()).item()
            assert dist_rel < 1e-3, f'Layerwise latent mode working incorrectly for model {model_name}-{classname}: difference = {dist_rel*100}%'
            
            print('.', end='')
    
    print('OK!')


