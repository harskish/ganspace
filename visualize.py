# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Patch for broken CTRL+C handler
# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch, json, numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs
from PIL import Image
from netdissect import proggan, nethook, easydict, zdataset
from netdissect.modelconfig import create_instrumented_model
from estimators import get_estimator
from models import get_instrumented_model
from scipy.cluster.vq import kmeans
import re
import sys
import datetime
import argparse
from tqdm import trange
from config import Config
from decomposition import get_random_dirs, get_or_compute, get_max_batch_size, SEED_VISUALIZATION
from utils import pad_frames 

def x_closest(p):
    distances = np.sqrt(np.sum((X - p)**2, axis=-1))
    idx = np.argmin(distances)
    return distances[idx], X[idx]

def make_gif(imgs, duration_secs, outname):
    head, *tail = [Image.fromarray((x * 255).astype(np.uint8)) for x in imgs]
    ms_per_frame = 1000 * duration_secs / instances
    head.save(outname, format='GIF', append_images=tail, save_all=True, duration=ms_per_frame, loop=0)

def make_mp4(imgs, duration_secs, outname):
    import shutil
    import subprocess as sp

    FFMPEG_BIN = shutil.which("ffmpeg")
    assert FFMPEG_BIN is not None, 'ffmpeg not found, install with "conda install -c conda-forge ffmpeg"'
    assert len(imgs[0].shape) == 3, 'Invalid shape of frame data'
    
    resolution = imgs[0].shape[0:2]
    fps = int(len(imgs) / duration_secs)

    command = [ FFMPEG_BIN,
        '-y', # overwrite output file
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{resolution[0]}x{resolution[1]}', # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', f'{fps}',
        '-i', '-', # imput from pipe
        '-an', # no audio
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '17',
        str(Path(outname).with_suffix('.mp4')) ]
    
    frame_data = np.concatenate([(x * 255).astype(np.uint8).reshape(-1) for x in imgs])
    with sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE) as p:
        ret = p.communicate(frame_data.tobytes())
        if p.returncode != 0:
            print(ret[1].decode("utf-8"))
            raise sp.CalledProcessError(p.returncode, command)


def make_grid(latent, lat_mean, lat_comp, lat_stdev, act_mean, act_comp, act_stdev, scale=1, n_rows=10, n_cols=5, make_plots=True, edit_type='latent'):
    from notebooks.notebook_utils import create_strip_centered

    inst.remove_edits()
    x_range = np.linspace(-scale, scale, n_cols, dtype=np.float32) # scale in sigmas

    rows = []
    for r in range(n_rows):
        curr_row = []
        out_batch = create_strip_centered(inst, edit_type, layer_key, [latent],
            act_comp[r], lat_comp[r], act_stdev[r], lat_stdev[r], act_mean, lat_mean, scale, 0, -1, n_cols)[0]
        for i, img in enumerate(out_batch):
            curr_row.append(('c{}_{:.2f}'.format(r, x_range[i]), img))

        rows.append(curr_row[:n_cols])

    inst.remove_edits()
    
    if make_plots:
        # If more rows than columns, make several blocks side by side
        n_blocks = 2 if n_rows > n_cols else 1
        
        for r, data in enumerate(rows):
            # Add white borders
            imgs = pad_frames([img for _, img in data]) 
            
            coord = ((r * n_blocks) % n_rows) + ((r * n_blocks) // n_rows)
            plt.subplot(n_rows//n_blocks, n_blocks, 1 + coord)
            plt.imshow(np.hstack(imgs))
            
            # Custom x-axis labels
            W = imgs[0].shape[1] # image width
            P = imgs[1].shape[1] # padding width
            locs = [(0.5*W + i*(W+P)) for i in range(n_cols)]
            plt.xticks(locs, ["{:.2f}".format(v) for v in x_range])
            plt.yticks([])
            plt.ylabel(f'C{r}')

        plt.tight_layout()
        plt.subplots_adjust(top=0.96) # make room for suptitle

    return [img for row in rows for img in row]


######################
### Visualize results
######################

if __name__ == '__main__':
    global max_batch, sample_shape, feature_shape, inst, args, layer_key, model

    args = Config().from_args()
    t_start = datetime.datetime.now()
    timestamp = lambda : datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] {args.model}, {args.layer}, {args.estimator}')

    # Ensure reproducibility
    torch.manual_seed(0) # also sets cuda seeds
    np.random.seed(0)

    # Speed up backend
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_grad_enabled(False)

    has_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if has_gpu else 'cpu')
    layer_key = args.layer
    layer_name = layer_key #layer_key.lower().split('.')[-1]

    basedir = Path(__file__).parent.resolve()
    outdir = basedir / 'out'

    # Load model
    inst = get_instrumented_model(args.model, args.output_class, layer_key, device, use_w=args.use_w)
    model = inst.model
    feature_shape = inst.feature_shape[layer_key]
    latent_shape = model.get_latent_shape()
    print('Feature shape:', feature_shape)

    # Layout of activations
    if len(feature_shape) != 4: # non-spatial
        axis_mask = np.ones(len(feature_shape), dtype=np.int32)
    else:
        axis_mask = np.array([0, 1, 1, 1]) # only batch fixed => whole activation volume used

    # Shape of sample passed to PCA
    sample_shape = feature_shape*axis_mask
    sample_shape[sample_shape == 0] = 1

    # Load or compute components
    dump_name = get_or_compute(args, inst)
    data = np.load(dump_name, allow_pickle=False) # does not contain object arrays
    X_comp = data['act_comp']
    X_global_mean = data['act_mean']
    X_stdev = data['act_stdev']
    X_var_ratio = data['var_ratio']
    X_stdev_random = data['random_stdevs']
    Z_global_mean = data['lat_mean']
    Z_comp = data['lat_comp']
    Z_stdev = data['lat_stdev']
    n_comp = X_comp.shape[0]
    data.close()

    # Transfer components to device
    tensors = SimpleNamespace(
        X_comp = torch.from_numpy(X_comp).to(device).float(), #-1, 1, C, H, W
        X_global_mean = torch.from_numpy(X_global_mean).to(device).float(), # 1, C, H, W
        X_stdev = torch.from_numpy(X_stdev).to(device).float(),
        Z_comp = torch.from_numpy(Z_comp).to(device).float(),
        Z_stdev = torch.from_numpy(Z_stdev).to(device).float(),
        Z_global_mean = torch.from_numpy(Z_global_mean).to(device).float(),
    )

    transformer = get_estimator(args.estimator, n_comp, args.sparsity)
    tr_param_str = transformer.get_param_str()

    # Compute max batch size given VRAM usage
    max_batch = args.batch_size or (get_max_batch_size(inst, device) if has_gpu else 1)
    print('Batch size:', max_batch)

    def show():
        if args.batch_mode:
            plt.close('all')
        else:
            plt.show()

    print(f'[{timestamp()}] Creating visualizations')

    # Ensure visualization gets new samples
    torch.manual_seed(SEED_VISUALIZATION)
    np.random.seed(SEED_VISUALIZATION)

    # Make output directories
    est_id = f'spca_{args.sparsity}' if args.estimator == 'spca' else args.estimator
    outdir_comp = outdir/model.name/layer_key.lower()/est_id/'comp'
    outdir_inst = outdir/model.name/layer_key.lower()/est_id/'inst'
    outdir_summ = outdir/model.name/layer_key.lower()/est_id/'summ'
    makedirs(outdir_comp, exist_ok=True)
    makedirs(outdir_inst, exist_ok=True)
    makedirs(outdir_summ, exist_ok=True)

    # Measure component sparsity (!= activation sparsity)
    sparsity = np.mean(X_comp == 0) # percentage of zero values in components
    print(f'Sparsity: {sparsity:.2f}')

    def get_edit_name(mode):
        if mode == 'activation':
            is_stylegan = 'StyleGAN' in args.model
            is_w = layer_key in ['style', 'g_mapping']
            return 'W' if (is_stylegan and is_w) else 'ACT'
        elif mode == 'latent':
            return model.latent_space_name()
        elif mode == 'both':
            return 'BOTH'
        else:
            raise RuntimeError(f'Unknown edit mode {mode}')

    # Only visualize applicable edit modes
    if args.use_w and layer_key in ['style', 'g_mapping']:
        edit_modes = ['latent'] # activation edit is the same
    else:
        edit_modes = ['activation', 'latent']

    # Summary grid, real components
    for edit_mode in edit_modes:
        plt.figure(figsize = (14,12))
        plt.suptitle(f"{args.estimator.upper()}: {model.name} - {layer_name}, {get_edit_name(edit_mode)} edit", size=16)
        make_grid(tensors.Z_global_mean, tensors.Z_global_mean, tensors.Z_comp, tensors.Z_stdev, tensors.X_global_mean,
            tensors.X_comp, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=14)
        plt.savefig(outdir_summ / f'components_{get_edit_name(edit_mode)}.jpg', dpi=300)
        show()

    if args.make_video:
        components = 15
        instances = 150
        
        # One reasonable, one over the top
        for sigma in [args.sigma, 3*args.sigma]:
            for c in range(components):
                for edit_mode in edit_modes:
                    frames = make_grid(tensors.Z_global_mean, tensors.Z_global_mean, tensors.Z_comp[c:c+1, :, :], tensors.Z_stdev[c:c+1], tensors.X_global_mean,
                        tensors.X_comp[c:c+1, :, :], tensors.X_stdev[c:c+1], n_rows=1, n_cols=instances, scale=sigma, make_plots=False, edit_type=edit_mode)
                    plt.close('all')

                    frames = [x for _, x in frames]
                    frames = frames + frames[::-1]
                    make_mp4(frames, 5, outdir_comp / f'{get_edit_name(edit_mode)}_sigma{sigma}_comp{c}.mp4')

    
    # Summary grid, random directions
    # Using the stdevs of the principal components for same norm
    random_dirs_act = torch.from_numpy(get_random_dirs(n_comp, np.prod(sample_shape)).reshape(-1, *sample_shape)).to(device)
    random_dirs_z = torch.from_numpy(get_random_dirs(n_comp, np.prod(inst.input_shape)).reshape(-1, *latent_shape)).to(device)
    
    for edit_mode in edit_modes:
        plt.figure(figsize = (14,12))
        plt.suptitle(f"{model.name} - {layer_name}, random directions w/ PC stdevs, {get_edit_name(edit_mode)} edit", size=16)
        make_grid(tensors.Z_global_mean, tensors.Z_global_mean, random_dirs_z, tensors.Z_stdev,
            tensors.X_global_mean, random_dirs_act, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=14)
        plt.savefig(outdir_summ / f'random_dirs_{get_edit_name(edit_mode)}.jpg', dpi=300)
        show()

    # Random instances w/ components added
    n_random_imgs = 10
    latents = model.sample_latent(n_samples=n_random_imgs)

    for img_idx in trange(n_random_imgs, desc='Random images', ascii=True):
        #print(f'Creating visualizations for random image {img_idx+1}/{n_random_imgs}')
        z = latents[img_idx][None, ...]

        # Summary grid, real components
        for edit_mode in edit_modes:
            plt.figure(figsize = (14,12))
            plt.suptitle(f"{args.estimator.upper()}: {model.name} - {layer_name}, {get_edit_name(edit_mode)} edit", size=16)
            make_grid(z, tensors.Z_global_mean, tensors.Z_comp, tensors.Z_stdev,
                tensors.X_global_mean, tensors.X_comp, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=14)
            plt.savefig(outdir_summ / f'samp{img_idx}_real_{get_edit_name(edit_mode)}.jpg', dpi=300)
            show()

        if args.make_video:
            components = 5
            instances = 150
            
            # One reasonable, one over the top
            for sigma in [args.sigma, 3*args.sigma]: #[2, 5]:
                for edit_mode in edit_modes:
                    imgs = make_grid(z, tensors.Z_global_mean, tensors.Z_comp, tensors.Z_stdev, tensors.X_global_mean, tensors.X_comp, tensors.X_stdev,
                        n_rows=components, n_cols=instances, scale=sigma, make_plots=False, edit_type=edit_mode)
                    plt.close('all')

                    for c in range(components):
                        frames = [x for _, x in imgs[c*instances:(c+1)*instances]]
                        frames = frames + frames[::-1]
                        make_mp4(frames, 5, outdir_inst / f'{get_edit_name(edit_mode)}_sigma{sigma}_img{img_idx}_comp{c}.mp4')

    print('Done in', datetime.datetime.now() - t_start)