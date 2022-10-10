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
from tqdm import trange, tqdm
from config import Config
from decomposition import get_random_dirs, get_or_compute, get_max_batch_size, SEED_VISUALIZATION
from utils import pad_frames
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle
from skimage.transform import resize
from matplotlib.patches import Ellipse
from estimators import get_estimator

def make_2Dscatter(X_comp,X_global_mean,X_stdev,inst,model,layer_key,outdir,device,n_samples=100,with_images=False,x_axis_pc=1,y_axis_pc=2):
    assert n_samples % 5 == 0, "n_samples has to be dividable by 5"
    samples_are_from_w = layer_key in ['g_mapping', 'mapping', 'style'] and inst.model.latent_space_name() == 'W'
    with torch.no_grad():
        #draw new latents
        latents = model.sample_latent(n_samples=n_samples)

        if(samples_are_from_w):
            activations = latents
        else:
            all_activations = []
            for i in range(0,int(n_samples/5)):
                z = latents[i*5:(i+1)*5:1]
                model.partial_forward(z,layer_name)

                activations_part = inst.retained_features()[layer_key].reshape((5, -1))
                all_activations.append(activations_part)
            activations = torch.cat(all_activations)

        global_mean = torch.from_numpy(X_global_mean.reshape(-1))
        activations = torch.sub(activations.cpu(),global_mean)

    X_comp_2 = X_comp.squeeze().reshape((X_comp.shape[0],-1)).transpose(1,0)[:,[x_axis_pc-1,y_axis_pc-1]]
    activations_reduced = activations @ X_comp_2
    x = activations_reduced[:,0]
    y = activations_reduced[:,1]

    fig, ax = plt.subplots(1)
    plt.scatter(x,y)
    plt.xlabel("PC"+str(x_axis_pc))
    plt.ylabel("PC"+str(y_axis_pc))
    plt.plot(0,0,'rx',alpha=0.5,markersize=10,zorder=10)

    sigma1 = Ellipse(xy=(0, 0), width=X_stdev[x_axis_pc-1], height=X_stdev[y_axis_pc-1],
                        edgecolor='r', fc='None', lw=2,alpha=0.3,zorder=10)
    sigma2 = Ellipse(xy=(0, 0), width=2*X_stdev[x_axis_pc-1], height=2*X_stdev[y_axis_pc-1],
                        edgecolor='r', fc='None', lw=2,alpha=0.25,zorder=10)
    sigma3 = Ellipse(xy=(0, 0), width=3*X_stdev[x_axis_pc-1], height=3*X_stdev[y_axis_pc-1],
                        edgecolor='r', fc='None', lw=2,alpha=0.2,zorder=10)
    sigma4 = Ellipse(xy=(0, 0), width=4*X_stdev[x_axis_pc-1], height=4*X_stdev[y_axis_pc-1],
                        edgecolor='r', fc='None', lw=2,alpha=0.15,zorder=10)
    ax.add_patch(sigma1)
    ax.add_patch(sigma2)
    ax.add_patch(sigma3)
    ax.add_patch(sigma4)

    if(with_images):
        w_primary_save = model.w_primary
        model.w_primary = True
        pbar = tqdm(total=len(list(zip(x, y))),desc='Generating images')
        for x0, y0 in zip(x, y):
            #activations are already centered | x_0 and y_0 are therefore offsets based from the mean
            #shift the mean in the corresponding directions and pass it to the network
            with torch.no_grad():
                #print(X_global_mean.reshape((X_global_mean.shape[0],-1)).shape,X_comp_2.shape,np.array([x0,y0]).shape)
                latent = (X_global_mean.reshape((X_global_mean.shape[0],-1)).squeeze() + X_comp_2 @ np.array([x0,y0]).T).reshape(X_global_mean.shape)
                latent = torch.from_numpy(latent).to(device)
                #print(latent.shape)
                img = model.forward(latent).squeeze()

                if(len(img.shape) == 3):
                    _cmap = 'viridis'
                    img = np.clip(img.cpu().numpy().transpose(1,2,0).astype(np.float32),0,1)
                else:
                    _cmap = 'gray'
                    img = img.cpu().numpy()

            img = resize(img,(256,256)) #downscale images
            ab = AnnotationBbox(OffsetImage(img,0.2,cmap=_cmap), (x0, y0), frameon=False)
            ax.add_artist(ab)
            pbar.update(1)

        pbar.close()
        model.w_primary = w_primary_save
        #Save interactive image as binary
        with open(outdir/model.name/layer_key.lower()/est_id/f'scatter_images{str(n_samples)}_{"PC"+str(x_axis_pc)}_{"PC"+str(y_axis_pc)}.pickle', 'wb') as pickle_file:
            pickle.dump(fig, pickle_file)
    else:
        plt.savefig(outdir/model.name/layer_key.lower()/est_id / f'scatter{str(n_samples)}_{"PC"+str(x_axis_pc)}_{"PC"+str(y_axis_pc)}.jpg', dpi=300)

    show()

def plot_explained_variance(X_var_ratio,X_dim,args):
    #PCA on complete random space to compare:
    transformer = get_estimator(args.estimator, args.components, args.sparsity)
    seed = np.random.randint(np.iinfo(np.int32).max) # use (reproducible) global rand state
    random_samples = torch.from_numpy(np.random.RandomState(seed).randn(10000, X_dim[-1]))
    transformer.fit(random_samples)
    _, _, random_var_ratio = transformer.get_components()

    fig, ax = plt.subplots(1)

    X_cumm_lst = []
    X_cumm = 0
    random_cumm_lst = []
    random_cumm = 0
    for i in range(X_var_ratio.shape[0]):
        X_cumm += X_var_ratio[i]
        X_cumm_lst.append(X_cumm)
        random_cumm += random_var_ratio[i]
        random_cumm_lst.append(random_cumm)

    plt.plot(X_cumm_lst,label="activation space")
    plt.plot(random_cumm_lst,label="random space")
    #plt.plot(X_var_ratio,label="activation space")
    #plt.plot(random_var_ratio,label="random space")
    plt.title("cumulative variance ratio")
    #plt.title("variance ratio")
    plt.xlabel("principal component")
    plt.ylabel("ratio")
    plt.legend()
    plt.show()

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
    format_str = 'rgb24' if imgs[0].shape[-1] > 1 else 'gray'
    resolution = imgs[0].shape[0:2]
    fps = int(len(imgs) / duration_secs)

    command = [ FFMPEG_BIN,
        '-y', # overwrite output file
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', f'{resolution[0]}x{resolution[1]}', # size of one frame
        '-pix_fmt', f'{format_str}',
        '-r', f'{fps}',
        '-i', '-', # imput from pipe
        '-an', # no audio
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '17',
        str(Path(outname).with_suffix('.mp4')) ]

    print((imgs[0] * 255).astype(np.uint8).reshape(-1).shape)
    frame_data = np.concatenate([(x * 255).astype(np.uint8).reshape(-1) for x in imgs])
    print(frame_data.shape)
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
    for r in trange(n_rows,unit="component"):
        curr_row = []
        out_batch = create_strip_centered(inst, edit_type, layer_key, [latent],
            act_comp[r], lat_comp[r], act_stdev[r], lat_stdev[r], act_mean, lat_mean, scale, 0, -1, n_cols)[0]
        #print("len(out_batch) =",len(out_batch))
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

            if(imgs[0].shape[2] > 1):
              img_row = np.hstack(imgs)
              _cmap = 'viridis'
            else:
              img_row = np.hstack(imgs).squeeze()
              _cmap = 'gray'
            plt.imshow(img_row,cmap=_cmap)

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

def get_edit_name(mode):
    if mode == 'activation':
        is_stylegan = 'StyleGAN' in args.model
        is_w = layer_key in ['style', 'g_mapping','mapping']
        return 'W' if (is_stylegan and is_w) else 'ACT'
    elif mode == 'latent':
        return model.latent_space_name()
    elif mode == 'both':
        return 'BOTH'
    else:
        raise RuntimeError(f'Unknown edit mode {mode}')

def show():
    if args.batch_mode:
        plt.close('all')
    else:
        plt.show()

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

    # Only visualize applicable edit modes
    if args.use_w and layer_key in ['style', 'g_mapping','mapping']:
        edit_modes = ['latent'] # activation edit is the same
    else:
        edit_modes = ['activation', 'latent']

    #plot_explained_variance(X_var_ratio,X_comp.shape[1:],args)

    #Scatter 2D of PC1 - PC2
    #(X_comp,inst,model,layer_key,outdir,n_samples=100
    if(args.show_scatter):
        make_2Dscatter(X_comp,X_global_mean,X_stdev,inst,model,layer_key,outdir,device,
        n_samples=args.scatter_samples,with_images=args.scatter_images,x_axis_pc=args.scatter_x_axis_pc,y_axis_pc=args.scatter_y_axis_pc)

    # Summary grid, real components
    for edit_mode in edit_modes:
        print("edit_mode =",edit_mode)
        plt.figure(figsize = (14,12))
        plt.suptitle(f"{args.estimator.upper()}: {model.name} - {layer_name}, {get_edit_name(edit_mode)} edit", size=16)
        make_grid(tensors.Z_global_mean, tensors.Z_global_mean, tensors.Z_comp, tensors.Z_stdev, tensors.X_global_mean,
            tensors.X_comp, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=args.np_directions, n_cols=args.np_images)
        plt.savefig(outdir_summ / f'components_{get_edit_name(edit_mode)}.jpg', dpi=300)
        show()

    print("args.make_video =",args.make_video)
    if args.make_video:
        components = args.nv_directions #10
        instances = args.nv_images#150

        # One reasonable, one over the top
        for sigma in [args.sigma, 3*args.sigma]:
            for c in range(components):
                for edit_mode in edit_modes:
                    print("Make grid for video")
                    frames = make_grid(tensors.Z_global_mean, tensors.Z_global_mean, tensors.Z_comp[c:c+1, :, :], tensors.Z_stdev[c:c+1], tensors.X_global_mean,
                        tensors.X_comp[c:c+1, :, :], tensors.X_stdev[c:c+1], n_rows=1, n_cols=instances, scale=sigma, make_plots=False, edit_type=edit_mode)
                    plt.close('all')
                    print("Done!")

                    frames = [x for _, x in frames]
                    frames = frames + frames[::-1]
                    print("num_frames =",len(frames)) #num_frames = 300
                    make_mp4(frames, 5, outdir_comp / f'{get_edit_name(edit_mode)}_sigma{sigma}_comp{c}.mp4')


    # Summary grid, random directions
    # Using the stdevs of the principal components for same norm
    random_dirs_act = torch.from_numpy(get_random_dirs(n_comp, np.prod(sample_shape)).reshape(-1, *sample_shape)).to(device)
    random_dirs_z = torch.from_numpy(get_random_dirs(n_comp, np.prod(inst.input_shape)).reshape(-1, *latent_shape)).to(device)

    for edit_mode in edit_modes:
        plt.figure(figsize = (14,12))
        plt.suptitle(f"{model.name} - {layer_name}, random directions w/ PC stdevs, {get_edit_name(edit_mode)} edit", size=16)
        make_grid(tensors.Z_global_mean, tensors.Z_global_mean, random_dirs_z, tensors.Z_stdev,
            tensors.X_global_mean, random_dirs_act, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=args.np_directions, n_cols=args.np_images)
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
                tensors.X_global_mean, tensors.X_comp, tensors.X_stdev, scale=args.sigma, edit_type=edit_mode, n_rows=args.np_directions, n_cols=args.np_images)
            plt.savefig(outdir_summ / f'samp{img_idx}_real_{get_edit_name(edit_mode)}.jpg', dpi=300)
            show()

        if args.make_video:
            components = args.nv_directions #10
            instances = args.nv_images#150

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
