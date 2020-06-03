# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import torch
import numpy as np
from os import makedirs
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import prettify_name, pad_frames

# Apply edit to given latents, return strip of images
def create_strip(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, sigma, layer_start, layer_end, num_frames=5):
    return _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev,
                lat_stdev, None, None, sigma, layer_start, layer_end, num_frames, center=False)

# Strip where the sample is centered along the component before manipulation
def create_strip_centered(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames=5):
    return _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev,
                lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center=True)

def _create_strip_impl(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center):
    if not isinstance(latents, list):
        latents = list(latents)

    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)

    if len(latents) > num_frames:
        # Batch over latents
        return _create_strip_batch_lat(inst, mode, layer, latents, x_comp, z_comp,
            act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center)
    else:
        # Batch over strip frames
        return _create_strip_batch_sigma(inst, mode, layer, latents, x_comp, z_comp,
            act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center)

# Batch over frames if there are more frames in strip than latents
def _create_strip_batch_sigma(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center):    
    inst.close()
    batch_frames = [[] for _ in range(len(latents))]
    
    B = min(num_frames, 5)
    lep_padded = ((num_frames - 1) // B + 1) * B
    sigma_range = np.linspace(-sigma, sigma, num_frames)
    sigma_range = np.concatenate([sigma_range, np.zeros((lep_padded - num_frames))])
    sigma_range = torch.from_numpy(sigma_range).float().to(inst.model.device)
    normalize = lambda v : v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-8)

    for i_batch in range(lep_padded // B):
        sigmas = sigma_range[i_batch*B:(i_batch+1)*B]
        
        for i_lat in range(len(latents)):
            z_single = latents[i_lat]
            z_batch = z_single.repeat_interleave(B, axis=0)
            
            zeroing_offset_act = 0
            zeroing_offset_lat = 0
            if center:
                if mode == 'activation':
                    # Center along activation before applying offset
                    inst.retain_layer(layer)
                    _ = inst.model.sample_np(z_single)
                    value = inst.retained_features()[layer].clone()
                    dotp = torch.sum((value - act_mean)*normalize(x_comp), dim=-1, keepdim=True)
                    zeroing_offset_act = normalize(x_comp)*dotp # offset that sets coordinate to zero
                else:
                    # Shift latent to lie on mean along given component
                    dotp = torch.sum((z_single - lat_mean)*normalize(z_comp), dim=-1, keepdim=True)
                    zeroing_offset_lat = dotp*normalize(z_comp)

            with torch.no_grad():
                z = z_batch

                if mode in ['latent', 'both']:
                    z = [z]*inst.model.get_max_latents()
                    delta = z_comp * sigmas.reshape([-1] + [1]*(z_comp.ndim - 1)) * lat_stdev
                    for i in range(layer_start, layer_end):
                        z[i] = z[i] - zeroing_offset_lat + delta

                if mode in ['activation', 'both']:
                    comp_batch = x_comp.repeat_interleave(B, axis=0)
                    delta = comp_batch * sigmas.reshape([-1] + [1]*(comp_batch.ndim - 1))
                    inst.edit_layer(layer, offset=delta * act_stdev - zeroing_offset_act)

                img_batch = inst.model.sample_np(z)
                if img_batch.ndim == 3:
                    img_batch = np.expand_dims(img_batch, axis=0)
                    
                for j, img in enumerate(img_batch):
                    idx = i_batch*B + j
                    if idx < num_frames:
                        batch_frames[i_lat].append(img)

    return batch_frames

# Batch over latents if there are more latents than frames in strip
def _create_strip_batch_lat(inst, mode, layer, latents, x_comp, z_comp, act_stdev, lat_stdev, act_mean, lat_mean, sigma, layer_start, layer_end, num_frames, center):    
    n_lat = len(latents)
    B = min(n_lat, 5)

    max_lat = inst.model.get_max_latents()
    if layer_end < 0 or layer_end > max_lat:
        layer_end = max_lat
    layer_start = np.clip(layer_start, 0, layer_end)
    
    len_padded = ((n_lat - 1) // B + 1) * B
    batch_frames = [[] for _ in range(n_lat)]

    for i_batch in range(len_padded // B):
        zs = latents[i_batch*B:(i_batch+1)*B]
        if len(zs) == 0:
            continue
        
        z_batch_single = torch.cat(zs, 0)

        inst.close() # don't retain, remove edits
        sigma_range = np.linspace(-sigma, sigma, num_frames, dtype=np.float32)

        normalize = lambda v : v / torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True) + 1e-8)
        
        zeroing_offset_act = 0
        zeroing_offset_lat = 0
        if center:
            if mode == 'activation':
                # Center along activation before applying offset
                inst.retain_layer(layer)
                _ = inst.model.sample_np(z_batch_single)
                value = inst.retained_features()[layer].clone()
                dotp = torch.sum((value - act_mean)*normalize(x_comp), dim=-1, keepdim=True)
                zeroing_offset_act = normalize(x_comp)*dotp # offset that sets coordinate to zero
            else:
                # Shift latent to lie on mean along given component
                dotp = torch.sum((z_batch_single - lat_mean)*normalize(z_comp), dim=-1, keepdim=True)
                zeroing_offset_lat = dotp*normalize(z_comp)

        for i in range(len(sigma_range)):
            s = sigma_range[i]

            with torch.no_grad():
                z = [z_batch_single]*inst.model.get_max_latents() # one per layer

                if mode in ['latent', 'both']:
                    delta = z_comp*s*lat_stdev
                    for i in range(layer_start, layer_end):
                        z[i] = z[i] - zeroing_offset_lat + delta

                if mode in ['activation', 'both']:
                    act_delta = x_comp*s*act_stdev
                    inst.edit_layer(layer, offset=act_delta - zeroing_offset_act)

                img_batch = inst.model.sample_np(z)
                if img_batch.ndim == 3:
                    img_batch = np.expand_dims(img_batch, axis=0)
                    
                for j, img in enumerate(img_batch):
                    img_idx = i_batch*B + j
                    if img_idx < n_lat:
                        batch_frames[img_idx].append(img)

    return batch_frames


def save_frames(title, model_name, rootdir, frames, strip_width=10):
    test_name = prettify_name(title)
    outdir = f'{rootdir}/{model_name}/{test_name}'
    makedirs(outdir, exist_ok=True)
    
    # Limit maximum resolution
    max_H = 512
    real_H = frames[0][0].shape[0]
    ratio = min(1.0, max_H / real_H)
    
    # Combined with first 10
    strips = [np.hstack(frames) for frames in frames[:strip_width]]
    if len(strips) >= strip_width:
        left_col = np.vstack(strips[0:strip_width//2])
        right_col = np.vstack(strips[5:10])
        grid = np.hstack([left_col, np.ones_like(left_col[:, :30]), right_col])
        im = Image.fromarray((255*grid).astype(np.uint8))
        im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
        im.save(f'{outdir}/{test_name}_all.png')
    else:
        print('Too few strips to create grid, creating just strips!')
    
    for ex_num, strip in enumerate(frames[:strip_width]):
        im = Image.fromarray(np.uint8(255*np.hstack(pad_frames(strip))))
        im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
        im.save(f'{outdir}/{test_name}_{ex_num}.png')