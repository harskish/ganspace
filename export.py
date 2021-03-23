# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# An interactive glumpy (OpenGL) + tkinter viewer for interacting with principal components.
# Requires OpenGL and CUDA support for rendering.
import threading
import time
from queue import Queue
from threading import Thread

import torch
import numpy as np
#from mttkinter import mtTkinter as tk
#from tkinter import ttk
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs
from torchvision.utils import save_image
from ganmodels import get_instrumented_model
# from config import Config
from decomposition import get_or_compute
from torch.nn.functional import interpolate
#from TkTorchWindow import TorchImageView
from functools import partial
from platform import system
from PIL import Image
from utils import pad_frames, prettify_name
import pickle

# For platform specific UI tweaks
is_windows = 'Windows' in system()
is_linux = 'Linux' in system()
is_mac = 'Darwin' in system()

# Don't bother without GPU
assert torch.cuda.is_available(), 'Interactive mode requires CUDA'


class Config:
    def __init__(self, model, layer, output_class, estimator, sparsity, video, batch_mode, batch_size, components, n,
                 use_w, sigma, inputs, seed):
        self.model = model
        self.layer = layer
        self.output_class = output_class
        self.estimator = estimator
        self.sparsity = sparsity
        self.video = video
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.components = components
        self.n = n
        self.use_w = use_w
        self.sigma = sigma
        self.inputs = inputs
        self.seed = seed


# Use syntax from paper
def get_edit_name(idx, s, e, name=None):
    return 'E({comp}, {edit_range}){edit_name}'.format(
        comp=idx,
        edit_range=f'{s}-{e}' if e > s else s,
        edit_name=f': {name}' if name else ''
    )


class GanModel(Thread):
    def __init__(self, model='StyleGAN', layer='g_mapping', class_name=None, est='ipca', sparsity=1.0, video=True,
                 batch=True, b=None, c=80, n=300_000, use_w=True, sigma=2.0, inputs=None, seed=None):
        Thread.__init__(self)
        print("initiate")

        self.q = Queue()

        self._running = True
        self.model_state = None
        self.components = None
        self.use_named_latents = None

        # App state
        self.state = SimpleNamespace(
            z=None,  # current latent(s)
            lat_slider_offset=0,  # part of lat that is explained by sliders
            act_slider_offset=0,  # part of act that is explained by sliders
            component_class=None,  # name of current PCs' image class
            seed=0,  # Latent z_i generated by seed+i
            base_act=None,  # activation of considered layer given z
        )

        self.model_name = model
        self.layer_name = layer
        self.class_name = class_name
        self.estimator = est
        self.sparsity = sparsity
        self.video = video
        self.batch = batch
        self.b = b
        self.c = c
        self.n = n
        self.use_w = use_w
        self.sigma = sigma
        self.inputs = inputs
        self.seed = seed

        # Speed up pytorch
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        # Load model
        self.inst = get_instrumented_model(self.model_name, self.class_name, self.layer_name, torch.device('cuda'),
                                           use_w=use_w)
        self.model = self.inst.model

        self.feat_shape = self.inst.feature_shape[self.layer_name]
        self.sample_dims = np.prod(self.feat_shape)

        # Initialize
        if self.inputs:
            self.load_named_components(self.inputs, self.class_name)
        else:
            self.load_components(self.class_name, self.inst)

        self.setup_model()
        self.cache = self.ParamCache()
        self.resample_latent(seed)

    # Load or compute PCA basis vectors
    def load_components(self, class_name, inst):

        config = Config(model=self.model_name, layer=self.layer_name, output_class=self.class_name,
                        estimator=self.estimator, sparsity=self.sparsity, batch_mode=self.batch, video=self.video,
                        batch_size=self.b, components=self.c, n=self.n, use_w=self.use_w,
                        sigma=self.sigma, inputs=self.inputs, seed=self.seed)
        dump_name = get_or_compute(config, inst)
        data = np.load(dump_name, allow_pickle=False)
        X_comp = data['act_comp']
        X_mean = data['act_mean']
        X_stdev = data['act_stdev']
        Z_comp = data['lat_comp']
        Z_mean = data['lat_mean']
        Z_stdev = data['lat_stdev']
        random_stdev_act = np.mean(data['random_stdevs'])
        n_comp = X_comp.shape[0]
        data.close()

        # Transfer to GPU
        self.components = SimpleNamespace(
            X_comp=torch.from_numpy(X_comp).cuda().float(),
            X_mean=torch.from_numpy(X_mean).cuda().float(),
            X_stdev=torch.from_numpy(X_stdev).cuda().float(),
            Z_comp=torch.from_numpy(Z_comp).cuda().float(),
            Z_stdev=torch.from_numpy(Z_stdev).cuda().float(),
            Z_mean=torch.from_numpy(Z_mean).cuda().float(),
            names=[f'Component {i}' for i in range(n_comp)],
            latent_types=[self.model.latent_space_name()] * n_comp,
            ranges=[(0, self.model.get_max_latents())] * n_comp,
        )

        # state.component_class = class_name  # invalidates cache
        self.use_named_latents = False
        print('Loaded components for', self.class_name, 'from', dump_name)

    # Load previously exported named components from
    # directory specified with '--inputs=path/to/comp'
    def load_named_components(self, path, class_name):
        import glob
        matches = glob.glob(f'{path}/*.pkl')

        selected = []
        for dump_path in matches:
            with open(dump_path, 'rb') as f:
                data = pickle.load(f)
                if data['model_name'] != self.model_name or data['output_class'] != class_name:
                    continue

                if data['latent_space'] != self.model.latent_space_name():
                    print('Skipping', dump_path, '(wrong latent space)')
                    continue

                selected.append(data)
                print('Using', dump_path)

        if len(selected) == 0:
            raise RuntimeError('No valid components in given path.')

        comp_dict = {k: [] for k in
                     ['X_comp', 'Z_comp', 'X_stdev', 'Z_stdev', 'names', 'types', 'layer_names', 'ranges',
                      'latent_types']}
        self.components = SimpleNamespace(**comp_dict)

        for d in selected:
            s = d['edit_start']
            e = d['edit_end']
            title = get_edit_name(d['component_index'], s, e - 1, d['name'])  # show inclusive
            self.components.X_comp.append(torch.from_numpy(d['act_comp']).cuda())
            self.components.Z_comp.append(torch.from_numpy(d['lat_comp']).cuda())
            self.components.X_stdev.append(d['act_stdev'])
            self.components.Z_stdev.append(d['lat_stdev'])
            self.components.names.append(title)
            self.components.types.append(d['edit_type'])
            self.components.layer_names.append(d['decomposition']['layer'])  # only for act
            self.components.ranges.append((s, e))
            self.components.latent_types.append(d['latent_space'])  # W or Z

        use_named_latents = True
        print('Loaded named components')

    # Project tensor 'X' onto orthonormal basis 'comp', return coordinates

    def reset_sliders(self, zero_on_failure=True):

        mode = self.model_state.mode

        # Not orthogonal: need to solve least-norm problem
        # Not batch size 1: one set of sliders not enough
        # Not principal components: unsupported format
        is_ortho = not (mode == 'latent' and self.model.latent_space_name() == 'Z')
        is_single = self.state.z.shape[0] == 1
        is_pcs = not self.use_named_latents

        self.state.lat_slider_offset = 0
        self.state.act_slider_offset = 0

        enabled = False
        if not (enabled and is_ortho and is_single and is_pcs):
            if zero_on_failure:
                self.zero_sliders()
            return

        if mode == 'activation':
            val = self.state.base_act
            mean = self.components.X_mean
            comp = self.components.X_comp
            stdev = self.components.X_stdev
        else:
            val = self.state.z
            mean = self.components.Z_mean
            comp = self.components.Z_comp
            stdev = self.components.Z_stdev

        n_sliders = len(self.model_state.sliders)
        coords = self.project_ortho(val - mean, comp)
        offset = torch.sum(coords[:n_sliders] * comp[:n_sliders], dim=0)
        scaled_coords = (coords.view(-1) / stdev).detach().cpu().numpy()

        # Part representable by sliders
        if mode == 'activation':
            self.state.act_slider_offset = offset
        else:
            self.state.lat_slider_offset = offset

        for i in range(n_sliders):
            self.model_state.sliders[i].set(round(scaled_coords[i], ndigits=1))

    def setup_model(self):
        print("ui")

        scale = 1.0

        N_COMPONENTS = min(70, len(self.components.names))
        self.model_state = SimpleNamespace(
            sliders=[0.0 for _ in range(N_COMPONENTS)],
            scales=[],
            truncation=0.9,
            outclass=self.class_name,
            random_seed='0',
            mode='latent',
            batch_size=1,  # how many images to show in window
            edit_layer_start=0,
            edit_layer_end=self.model.get_max_latents() - 1,
            slider_max_val=10.0
        )

        # Choose range where latents are modified
        def set_min(val):
            self.model_state.edit_layer_start.set(min(int(val), self.model_state.edit_layer_end.get()))

        def set_max(val):
            self.model_state.edit_layer_end.set(max(int(val), self.model_state.edit_layer_start.get()))

        max_latent_idx = self.model.get_max_latents() - 1

        # Random seed
        def update_seed():
            seed_str = self.model_state.random_seed.get()
            if seed_str.isdigit():
                self.resample_latent(int(seed_str))

    def resample_latent(self, seed=None, only_style=False):
        self.class_name = self.model_state.outclass
        if self.class_name.isnumeric():
            class_name = int(self.class_name)

        if hasattr(self.model, 'is_valid_class'):
            if not self.model.is_valid_class(self.class_name):
                return

        self.model.set_output_class(self.class_name)

        B = self.model_state.batch_size
        self.state.seed = np.random.randint(np.iinfo(np.int32).max - B) if seed is None else seed
        self.model_state.random_seed = self.state.seed

        # Use consecutive seeds along batch dimension (for easier reproducibility)
        trunc = self.model_state.truncation
        latents = [self.model.sample_latent(1, seed=self.state.seed + i, truncation=trunc) for i in range(B)]

        self.state.z = torch.cat(latents).clone().detach()  # make leaf node
        assert self.state.z.is_leaf, 'Latent is not leaf node!'

        if hasattr(self.model, 'truncation'):
            self.model.truncation = self.model_state.truncation
        print(f'Seeds: {self.state.seed} -> {self.state.seed + B - 1}' if B > 1 else f'Seed: {self.state.seed}')

        torch.manual_seed(self.state.seed)
        self.model.partial_forward(self.state.z, self.layer_name)
        self.state.base_act = self.inst.retained_features()[self.layer_name]

        self.reset_sliders(zero_on_failure=False)

    # Used to recompute after changing class of conditional model
    def recompute_components(self, ):
        self.class_name = self.model_state.outclass.get()
        if self.class_name.isnumeric():
            self.class_name = int(self.class_name)

        if hasattr(self.model, 'is_valid_class'):
            if not self.model.is_valid_class(self.class_name):
                return

        if hasattr(self.model, 'set_output_class'):
            self.model.set_output_class(self.class_name)

        self.load_components(self.class_name, self.inst)

    # Used to detect parameter changes for lazy recomputation
    class ParamCache:
        def update(self, **kwargs):
            dirty = False
            for argname, val in kwargs.items():
                # Check pointer, then value
                current = getattr(self, argname, 0)
                if current is not val and pickle.dumps(current) != pickle.dumps(val):
                    setattr(self, argname, val)
                    dirty = True
            return dirty

    def l2norm(self, t):
        return torch.norm(t.view(t.shape[0], -1), p=2, dim=1, keepdim=True)

    def apply_edit(self, z0, delta):
        return z0 + delta

    def on_draw(self, file='img1.png', slider_values=[]):
        self.set_sliders(slider_values)

        n_comp = len(self.model_state.sliders)
        slider_vals = np.array([s for s in self.model_state.sliders], dtype=np.float32)
        # Run model sparingly
        mode = self.model_state.mode
        latent_start = self.model_state.edit_layer_start
        latent_end = self.model_state.edit_layer_end + 1  # save as exclusive, show as inclusive

        if self.cache.update(coords=slider_vals, comp=self.state.component_class, mode=mode, z=self.state.z,
                             s=latent_start,
                             e=latent_end):
            with torch.no_grad():
                z_base = self.state.z - self.state.lat_slider_offset
                z_deltas = [0.0] * self.model.get_max_latents()
                z_delta_global = 0.0

                n_comp = slider_vals.size
                act_deltas = {}

                if torch.is_tensor(self.state.act_slider_offset):
                    act_deltas[self.layer_name] = -self.state.act_slider_offset

                for space in self.components.latent_types:
                    assert space == self.model.latent_space_name(), \
                        'Cannot mix latent spaces (for now)'

                for c in range(n_comp):
                    coord = slider_vals[c]
                    if coord == 0:
                        continue

                    edit_mode = self.components.types[c] if self.use_named_latents else mode

                    # Activation offset
                    if edit_mode in ['activation', 'both']:
                        delta = self.components.X_comp[c] * self.components.X_stdev[c] * coord
                        name = self.components.layer_names[c] if self.use_named_latents else self.layer_name
                        act_deltas[name] = act_deltas.get(name, 0.0) + delta

                    # Latent offset
                    if edit_mode in ['latent', 'both']:
                        delta = self.components.Z_comp[c] * self.components.Z_stdev[c] * coord
                        edit_range = self.components.ranges[c] if self.use_named_latents else (latent_start, latent_end)
                        full_range = (edit_range == (0, self.model.get_max_latents()))

                        # Single or multiple offsets?
                        if full_range:
                            z_delta_global = z_delta_global + delta
                        else:
                            for l in range(*edit_range):
                                z_deltas[l] = z_deltas[l] + delta

                # Apply activation deltas
                self.inst.remove_edits()
                for layer, delta in act_deltas.items():
                    self.inst.edit_layer(layer, offset=delta)

                # Evaluate
                has_offsets = any(torch.is_tensor(t) for t in z_deltas)
                z_final = self.apply_edit(z_base, z_delta_global)
                if has_offsets:
                    z_final = [self.apply_edit(z_final, d) for d in z_deltas]
                self.img = self.model.forward(z_final).clamp(0.0, 1.0)

                image = self.img.cpu()
                save_image(image, file)
                self.img.cuda()

        # app.draw(img)

    # Shared by glumpy and tkinter
    def handle_keypress(self, code):
        if code == 65307:  # ESC
            self.shutdown()
        elif code == 65360:  # HOME
            self.reset_sliders()
        elif code == 114:  # R
            pass  # reset_sliders()

    def on_key_release(self, symbol, modifiers):
        self.handle_keypress(symbol)

    def project_ortho(self, X, comp):
        N = comp.shape[0]
        coords = (comp.reshape(N, -1) * X.reshape(-1)).sum(dim=1)
        return coords.reshape([N] + [1] * X.ndim)

    def zero_sliders(self):
        for v in self.model_state.sliders:
            v.set(0.0)

    def set_sliders(self, values):
        if len(values) < len(self.model_state.sliders):
            values.extend([0 for x in range(len(self.model_state.sliders) - len(values))])
        for i in range(len(self.model_state.sliders)):
            self.model_state.sliders[i] = values[i]

    def run(self, file='img1.png', slider_values=[], seed=None):
        startTime = time.time()
        self.resample_latent(seed)
        self.on_draw(file, slider_values)
        executiontime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executiontime))


if __name__ == '__main__':
    q = Queue()
    car = GanModel(model='StyleGAN2', class_name='car', layer='style', n=1_000_000, b=10_000)
    #dog = GanModel(model='BigGAN-512', class_name='husky', layer='generator.gen_z', n=1_000_000, use_w=False)
    car.setDaemon(True)
    #dog.setDaemon(True)

    car.start()
    # dog.start()

    car.run(file='images/car0.png', seed=0, slider_values=[0 for x in range(70)])
    car.set_sliders(slider_values=[1 for x in range(70)])

    # car.run(file='images/car1.png', seed=1, slider_values=[0 for x in range(70)])
    # car.run(file='images/car2.png', seed=2, slider_values=[0 for x in range(70)])
    # car.run(file='images/car3.png', seed=3, slider_values=[0 for x in range(70)])
    # car.run(file='images/car4.png', seed=4, slider_values=[0 for x in range(70)])

    # dog.run(file='images/dog0.png', seed=0, slider_values=[0 for x in range(70)])
    # dog.run(file='images/dog1.png', seed=1, slider_values=[0 for x in range(70)])
    # dog.run(file='images/dog2.png', seed=2, slider_values=[0 for x in range(70)])
    # dog.run(file='images/dog3.png', seed=3, slider_values=[0 for x in range(70)])
    # dog.run(file='images/dog4.png', seed=4, slider_values=[0 for x in range(70)])

    car.join()
    #dog.join()
    exit(0)
