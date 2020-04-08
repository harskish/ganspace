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
from models import get_instrumented_model


SEED = 1369
SAMPLES = 100
B = 10

torch.backends.cudnn.benchmark = True
has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if has_gpu else 'cpu')


def compare(model, layer, z1, z2):
    # Run partial forward
    torch.manual_seed(0)
    np.random.seed(0)
    inst._retained[layer] = None
    with torch.no_grad():
        model.partial_forward(z1, layer)
    
    assert inst._retained[layer] is not None, 'Layer not retained (partial)'
    feat_partial = inst._retained[layer].cpu().numpy().copy().reshape(-1)

    # Run standard forward
    torch.manual_seed(0)
    np.random.seed(0)
    inst._retained[layer] = None
    with torch.no_grad():
        model.forward(z2)
    
    assert inst._retained[layer] is not None, 'Layer not retained (full)'
    feat_full = inst.retained_features()[layer].cpu().numpy().copy().reshape(-1)

    diff = np.sum(np.abs(feat_partial - feat_full))
    return diff


configs = []

# StyleGAN2
models = ['StyleGAN2']
layers = ['convs.0',]
classes = ['cat', 'ffhq']
configs.append(itertools.product(models, layers, classes))

# StyleGAN
models = ['StyleGAN']
layers = [
    'g_synthesis.blocks.128x128.conv0_up',
    'g_synthesis.blocks.128x128.conv0_up.upscale',
    'g_synthesis.blocks.256x256.conv0_up',
    'g_synthesis.blocks.1024x1024.epi2.style_mod.lin'
]
classes = ['ffhq']
configs.append(itertools.product(models, layers, classes))

# ProGAN
models = ['ProGAN']
layers = ['layer2', 'layer7']
classes = ['churchoutdoor', 'bedroom']
configs.append(itertools.product(models, layers, classes))

# BigGAN
models = ['BigGAN-512', 'BigGAN-256', 'BigGAN-128']
layers = ['generator.layers.2.conv_1', 'generator.layers.5.relu', 'generator.layers.10.bn_2']
classes = ['husky']
configs.append(itertools.product(models, layers, classes))

# Run all configurations
for config in configs:
    for model_name, layer, outclass in config:
        print('Testing', model_name, layer, outclass)
        inst = get_instrumented_model(model_name, outclass, layer, device)
        model = inst.model

        # Test negative
        z_dummy = model.sample_latent(B)
        z1 = torch.zeros_like(z_dummy).to(device)
        z2 = torch.ones_like(z_dummy).to(device)
        diff = compare(model, layer, z1, z2)
        assert diff > 1e-8, 'Partial and full should differ, but they do not!'

        # Test model randomness (should be seeded away)
        z1 = model.sample_latent(1)
        inst._retained[layer] = None
        with torch.no_grad():
            model.forward(z1)
            feat1 = inst._retained[layer].reshape(-1)
            model.forward(z1)
            feat2 = inst._retained[layer].reshape(-1)
            diff = torch.sum(torch.abs(feat1 - feat2))
            assert diff < 1e-8, f'Layer {layer} output contains randomness, diff={diff}'
    

        # Test positive
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        latents = model.sample_latent(SAMPLES, seed=SEED)

        for i in range(0, SAMPLES, B):
            print(f'Layer {layer}: {i}/{SAMPLES}', end='\r')
            z = latents[i:i+B]
            diff = compare(model, layer, z, z)
            assert diff < 1e-8, f'Partial and full forward differ by {diff}'

        del model
        torch.cuda.empty_cache()