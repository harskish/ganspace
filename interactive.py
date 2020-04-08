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

import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs
from models import get_instrumented_model
from config import Config
from decomposition import get_or_compute
from torch.nn.functional import interpolate
from TkTorchWindow import TorchImageView
from functools import partial
from platform import system
from PIL import Image
from utils import pad_frames, prettify_name
import pickle

# For platform specific UI tweaks
is_windows = 'Windows' in system()
is_linux = 'Linux' in system()
is_mac = 'Darwin' in system()

# Read input parameters
args = Config().from_args()

# Don't bother without GPU
assert torch.cuda.is_available(), 'Interactive mode requires CUDA'

# Use syntax from paper
def get_edit_name(idx, s, e, name=None):
    return 'E({comp}, {edit_range}){edit_name}'.format(
        comp = idx,
        edit_range = f'{s}-{e}' if e > s else s,
        edit_name = f': {name}' if name else ''
    )

# Load or compute PCA basis vectors
def load_components(class_name, inst):
    global components, state, use_named_latents

    config = args.from_dict({ 'output_class': class_name })
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
    components = SimpleNamespace(
        X_comp = torch.from_numpy(X_comp).cuda().float(),
        X_mean = torch.from_numpy(X_mean).cuda().float(),
        X_stdev = torch.from_numpy(X_stdev).cuda().float(),
        Z_comp = torch.from_numpy(Z_comp).cuda().float(),
        Z_stdev = torch.from_numpy(Z_stdev).cuda().float(),
        Z_mean = torch.from_numpy(Z_mean).cuda().float(),
        names = [f'Component {i}' for i in range(n_comp)],
        latent_types = [model.latent_space_name()]*n_comp,
        ranges = [(0, model.get_max_latents())]*n_comp,
    )
    
    state.component_class = class_name # invalidates cache
    use_named_latents = False
    print('Loaded components for', class_name, 'from', dump_name)

# Load previously exported named components from
# directory specified with '--inputs=path/to/comp'
def load_named_components(path, class_name):
    global components, state, use_named_latents

    import glob
    matches = glob.glob(f'{path}/*.pkl')

    selected = []
    for dump_path in matches:
        with open(dump_path, 'rb') as f:
            data = pickle.load(f)
            if data['model_name'] != model_name or data['output_class'] != class_name:
                continue

            if data['latent_space'] != model.latent_space_name():
                print('Skipping', dump_path, '(wrong latent space)')
                continue
            
            selected.append(data)
            print('Using', dump_path)

    if len(selected) == 0:
        raise RuntimeError('No valid components in given path.')

    comp_dict = { k : [] for k in ['X_comp', 'Z_comp', 'X_stdev', 'Z_stdev', 'names', 'types', 'layer_names', 'ranges', 'latent_types'] }
    components = SimpleNamespace(**comp_dict)

    for d in selected:
        s = d['edit_start']
        e = d['edit_end']
        title = get_edit_name(d['component_index'], s, e - 1, d['name']) # show inclusive
        components.X_comp.append(torch.from_numpy(d['act_comp']).cuda())
        components.Z_comp.append(torch.from_numpy(d['lat_comp']).cuda())
        components.X_stdev.append(d['act_stdev'])
        components.Z_stdev.append(d['lat_stdev'])
        components.names.append(title)
        components.types.append(d['edit_type'])
        components.layer_names.append(d['decomposition']['layer']) # only for act
        components.ranges.append((s, e))
        components.latent_types.append(d['latent_space']) # W or Z
    
    use_named_latents = True
    print('Loaded named components')

def setup_model():
    global model, inst, layer_name, model_name, feat_shape, args, class_name

    model_name = args.model
    layer_name = args.layer
    class_name = args.output_class

    # Speed up pytorch
    torch.autograd.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # Load model
    inst = get_instrumented_model(model_name, class_name, layer_name, torch.device('cuda'), use_w=args.use_w)
    model = inst.model

    feat_shape = inst.feature_shape[layer_name]
    sample_dims = np.prod(feat_shape)

    # Initialize 
    if args.inputs:
        load_named_components(args.inputs, class_name)
    else:
        load_components(class_name, inst)

# Project tensor 'X' onto orthonormal basis 'comp', return coordinates
def project_ortho(X, comp):
    N = comp.shape[0]
    coords = (comp.reshape(N, -1) * X.reshape(-1)).sum(dim=1)
    return coords.reshape([N]+[1]*X.ndim)

def zero_sliders():
    for v in ui_state.sliders:
        v.set(0.0)

def reset_sliders(zero_on_failure=True):
    global ui_state

    mode = ui_state.mode.get()

    # Not orthogonal: need to solve least-norm problem
    # Not batch size 1: one set of sliders not enough
    # Not principal components: unsupported format
    is_ortho = not (mode == 'latent' and model.latent_space_name() == 'Z')
    is_single = state.z.shape[0] == 1
    is_pcs = not use_named_latents

    state.lat_slider_offset = 0
    state.act_slider_offset = 0

    enabled = False
    if not (enabled and is_ortho and is_single and is_pcs):
        if zero_on_failure:
            zero_sliders()
        return

    if  mode == 'activation':
        val = state.base_act
        mean = components.X_mean
        comp = components.X_comp
        stdev = components.X_stdev
    else:
        val = state.z
        mean = components.Z_mean
        comp = components.Z_comp
        stdev = components.Z_stdev

    n_sliders = len(ui_state.sliders)
    coords = project_ortho(val - mean, comp)
    offset = torch.sum(coords[:n_sliders] * comp[:n_sliders], dim=0)
    scaled_coords = (coords.view(-1) / stdev).detach().cpu().numpy()
    
    # Part representable by sliders
    if mode == 'activation':
        state.act_slider_offset = offset
    else:
        state.lat_slider_offset = offset

    for i in range(n_sliders):
        ui_state.sliders[i].set(round(scaled_coords[i], ndigits=1))

def setup_ui():
    global root, toolbar, ui_state, app, canvas

    root = tk.Tk()
    scale = 1.0
    app = TorchImageView(root, width=int(scale*1024), height=int(scale*1024), show_fps=False)
    app.pack(fill=tk.BOTH, expand=tk.YES)
    root.protocol("WM_DELETE_WINDOW", shutdown)
    root.title('GANspace')

    toolbar = tk.Toplevel(root)
    toolbar.protocol("WM_DELETE_WINDOW", shutdown)
    toolbar.geometry("215x800+0+0")
    toolbar.title('')

    N_COMPONENTS = min(70, len(components.names))
    ui_state = SimpleNamespace(
        sliders = [tk.DoubleVar(value=0.0) for _ in range(N_COMPONENTS)],
        scales = [],
        truncation = tk.DoubleVar(value=0.9),
        outclass = tk.StringVar(value=class_name),
        random_seed = tk.StringVar(value='0'),
        mode = tk.StringVar(value='latent'),
        batch_size = tk.IntVar(value=1), # how many images to show in window
        edit_layer_start = tk.IntVar(value=0),
        edit_layer_end = tk.IntVar(value=model.get_max_latents() - 1),
        slider_max_val = 10.0
    )

    # Z vs activation mode button
    #tk.Radiobutton(toolbar, text=f"Latent ({model.latent_space_name()})", variable=ui_state.mode, command=reset_sliders, value='latent').pack(fill="x")
    #tk.Radiobutton(toolbar, text="Activation", variable=ui_state.mode, command=reset_sliders, value='activation').pack(fill="x")

    # Choose range where latents are modified
    def set_min(val):
        ui_state.edit_layer_start.set(min(int(val), ui_state.edit_layer_end.get()))
    def set_max(val):
        ui_state.edit_layer_end.set(max(int(val), ui_state.edit_layer_start.get()))
    max_latent_idx = model.get_max_latents() - 1
    
    if not use_named_latents:
        slider_min = tk.Scale(toolbar, command=set_min, variable=ui_state.edit_layer_start,
            label='Layer start', from_=0, to=max_latent_idx, orient=tk.HORIZONTAL).pack(fill="x")
        slider_max = tk.Scale(toolbar, command=set_max, variable=ui_state.edit_layer_end,
            label='Layer end', from_=0, to=max_latent_idx, orient=tk.HORIZONTAL).pack(fill="x")

    # Scrollable list of components
    outer_frame = tk.Frame(toolbar, borderwidth=2, relief=tk.SUNKEN)
    canvas = tk.Canvas(outer_frame, highlightthickness=0, borderwidth=0)
    frame = tk.Frame(canvas)
    vsb = tk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)

    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((4,4), window=frame, anchor="nw")

    def onCanvasConfigure(event):
        canvas.itemconfigure("all", width=event.width)
        canvas.configure(scrollregion=canvas.bbox("all"))
    canvas.bind("<Configure>", onCanvasConfigure)

    def on_scroll(event):
        delta = 1 if (event.num == 5 or event.delta < 0) else -1
        canvas.yview_scroll(delta, "units")

    canvas.bind_all("<Button-4>", on_scroll)
    canvas.bind_all("<Button-5>", on_scroll)
    canvas.bind_all("<MouseWheel>", on_scroll)
    canvas.bind_all("<Key>", lambda event : handle_keypress(event.keysym_num))

    # Sliders and buttons
    for i in range(N_COMPONENTS):
        inner = tk.Frame(frame, borderwidth=1, background="#aaaaaa")
        scale = tk.Scale(inner, variable=ui_state.sliders[i], from_=-ui_state.slider_max_val,
            to=ui_state.slider_max_val, resolution=0.1, orient=tk.HORIZONTAL, label=components.names[i])
        scale.pack(fill=tk.X, side=tk.LEFT, expand=True)
        ui_state.scales.append(scale) # for changing label later
        if not use_named_latents:
            tk.Button(inner, text=f"Save", command=partial(export_direction, i, inner)).pack(fill=tk.Y, side=tk.RIGHT)
        inner.pack(fill=tk.X)

    outer_frame.pack(fill="both", expand=True, pady=0)

    tk.Button(toolbar, text="Reset", command=reset_sliders).pack(anchor=tk.CENTER, fill=tk.X, padx=4, pady=4)

    tk.Scale(toolbar, variable=ui_state.truncation, from_=0.01, to=1.0,
        resolution=0.01, orient=tk.HORIZONTAL, label='Truncation').pack(fill="x")

    tk.Scale(toolbar, variable=ui_state.batch_size, from_=1, to=9,
        resolution=1, orient=tk.HORIZONTAL, label='Batch size').pack(fill="x")
    
    # Output class
    frame = tk.Frame(toolbar)
    tk.Label(frame, text="Class name").pack(fill="x", side="left")
    tk.Entry(frame, textvariable=ui_state.outclass).pack(fill="x", side="right", expand=True, padx=5)
    frame.pack(fill=tk.X, pady=3)

    # Random seed
    def update_seed():
        seed_str = ui_state.random_seed.get()
        if seed_str.isdigit():
            resample_latent(int(seed_str))
    frame = tk.Frame(toolbar)
    tk.Label(frame, text="Seed").pack(fill="x", side="left")
    tk.Entry(frame, textvariable=ui_state.random_seed, width=12).pack(fill="x", side="left", expand=True, padx=2)
    tk.Button(frame, text="Update", command=update_seed).pack(fill="y", side="right", padx=3)
    frame.pack(fill=tk.X, pady=3)
    
    # Get new latent or new components
    tk.Button(toolbar, text="Resample latent", command=partial(resample_latent, None, False)).pack(anchor=tk.CENTER, fill=tk.X, padx=4, pady=4)
    #tk.Button(toolbar, text="Recompute", command=recompute_components).pack(anchor=tk.CENTER, fill=tk.X)

# App state
state = SimpleNamespace(
    z=None,                  # current latent(s)
    lat_slider_offset = 0,   # part of lat that is explained by sliders
    act_slider_offset = 0,   # part of act that is explained by sliders
    component_class=None,    # name of current PCs' image class
    seed=0,                  # Latent z_i generated by seed+i
    base_act = None,         # activation of considered layer given z
)

def resample_latent(seed=None, only_style=False):
    class_name = ui_state.outclass.get()
    if class_name.isnumeric():
        class_name = int(class_name)
    
    if hasattr(model, 'is_valid_class'):
        if not model.is_valid_class(class_name):
            return

    model.set_output_class(class_name)
    
    B = ui_state.batch_size.get()
    state.seed = np.random.randint(np.iinfo(np.int32).max - B) if seed is None else seed
    ui_state.random_seed.set(str(state.seed))
    
    # Use consecutive seeds along batch dimension (for easier reproducibility)
    trunc = ui_state.truncation.get()
    latents = [model.sample_latent(1, seed=state.seed + i, truncation=trunc) for i in range(B)]

    state.z = torch.cat(latents).clone().detach() # make leaf node
    assert state.z.is_leaf, 'Latent is not leaf node!'
    
    if hasattr(model, 'truncation'):
        model.truncation = ui_state.truncation.get()
    print(f'Seeds: {state.seed} -> {state.seed + B - 1}' if B > 1 else f'Seed: {state.seed}')

    torch.manual_seed(state.seed)
    model.partial_forward(state.z, layer_name)
    state.base_act = inst.retained_features()[layer_name]
    
    reset_sliders(zero_on_failure=False)

    # Remove focus from text entry
    canvas.focus_set()

# Used to recompute after changing class of conditional model
def recompute_components():
    class_name = ui_state.outclass.get()
    if class_name.isnumeric():
        class_name = int(class_name)
    
    if hasattr(model, 'is_valid_class'):
        if not model.is_valid_class(class_name):
            return

    if hasattr(model, 'set_output_class'):
        model.set_output_class(class_name)
    
    load_components(class_name, inst)

# Used to detect parameter changes for lazy recomputation
class ParamCache():
    def update(self, **kwargs):
        dirty = False
        for argname, val in kwargs.items():
            # Check pointer, then value
            current = getattr(self, argname, 0)
            if current is not val and pickle.dumps(current) != pickle.dumps(val):
                setattr(self, argname, val)
                dirty = True
        return dirty

cache = ParamCache()

def l2norm(t):
    return torch.norm(t.view(t.shape[0], -1), p=2, dim=1, keepdim=True)

def apply_edit(z0, delta):
    return z0 + delta

def reposition_toolbar():
    size, X, Y = root.winfo_geometry().split('+')
    W, H = size.split('x')
    toolbar_W = toolbar.winfo_geometry().split('x')[0]
    offset_y = -30 if is_linux else 0 # window title bar
    toolbar.geometry(f'{toolbar_W}x{H}+{int(X)-int(toolbar_W)}+{int(Y)+offset_y}')
    toolbar.update()

def on_draw():
    global img

    n_comp = len(ui_state.sliders)
    slider_vals = np.array([s.get() for s in ui_state.sliders], dtype=np.float32)
    
    # Run model sparingly
    mode = ui_state.mode.get()
    latent_start = ui_state.edit_layer_start.get()
    latent_end = ui_state.edit_layer_end.get() + 1 # save as exclusive, show as inclusive

    if cache.update(coords=slider_vals, comp=state.component_class, mode=mode, z=state.z, s=latent_start, e=latent_end):
        with torch.no_grad():
            z_base = state.z - state.lat_slider_offset
            z_deltas = [0.0]*model.get_max_latents()
            z_delta_global = 0.0
            
            n_comp = slider_vals.size
            act_deltas = {}
            
            if torch.is_tensor(state.act_slider_offset):
                act_deltas[layer_name] = -state.act_slider_offset

            for space in components.latent_types:
                assert space == model.latent_space_name(), \
                    'Cannot mix latent spaces (for now)'

            for c in range(n_comp):
                coord = slider_vals[c]
                if coord == 0:
                    continue

                edit_mode = components.types[c] if use_named_latents else mode
                
                # Activation offset
                if edit_mode in ['activation', 'both']:
                    delta = components.X_comp[c] * components.X_stdev[c] * coord
                    name = components.layer_names[c] if use_named_latents else layer_name
                    act_deltas[name] = act_deltas.get(name, 0.0) + delta

                # Latent offset
                if edit_mode in ['latent', 'both']:
                    delta = components.Z_comp[c] * components.Z_stdev[c] * coord
                    edit_range = components.ranges[c] if use_named_latents else (latent_start, latent_end)
                    full_range = (edit_range == (0, model.get_max_latents()))
                    
                    # Single or multiple offsets?
                    if full_range:
                        z_delta_global = z_delta_global + delta
                    else:
                        for l in range(*edit_range):
                            z_deltas[l] = z_deltas[l] + delta

            # Apply activation deltas
            inst.remove_edits()
            for layer, delta in act_deltas.items():
                inst.edit_layer(layer, offset=delta)
            
            # Evaluate
            has_offsets = any(torch.is_tensor(t) for t in z_deltas)
            z_final = apply_edit(z_base, z_delta_global)
            if has_offsets:
                z_final = [apply_edit(z_final, d) for d in z_deltas]
            img = model.forward(z_final).clamp(0.0, 1.0)

    app.draw(img)

# Save necessary data to disk for later loading
def export_direction(idx, button_frame):
    name = tk.StringVar(value='')
    num_strips = tk.IntVar(value=0)
    strip_width = tk.IntVar(value=5)

    slider_values = np.array([s.get() for s in ui_state.sliders])
    slider_value = slider_values[idx]
    if (slider_values != 0).sum() > 1:
        print('Please modify only one slider')
        return
    elif slider_value == 0:
        print('Modify selected slider to set usable range (currently 0)')
        return
    
    popup = tk.Toplevel(root)
    popup.geometry("200x200+0+0")
    tk.Label(popup, text="Edit name").pack()
    tk.Entry(popup, textvariable=name).pack(pady=5)
    # tk.Scale(popup, from_=0, to=30, variable=num_strips,
    #    resolution=1, orient=tk.HORIZONTAL, length=200, label='Image strips to export').pack()
    # tk.Scale(popup, from_=3, to=15, variable=strip_width,
    #    resolution=1, orient=tk.HORIZONTAL, length=200, label='Image strip width').pack()
    tk.Button(popup, text='OK', command=popup.quit).pack()
    
    canceled = False
    def on_close():
        nonlocal canceled
        canceled = True
        popup.quit()

    popup.protocol("WM_DELETE_WINDOW", on_close)
    x = button_frame.winfo_rootx()
    y = button_frame.winfo_rooty()
    w = int(button_frame.winfo_geometry().split('x')[0])
    popup.geometry('%dx%d+%d+%d' % (180, 90, x + w, y))
    popup.mainloop()
    popup.destroy()

    # Update slider name
    label = get_edit_name(idx, ui_state.edit_layer_start.get(),
        ui_state.edit_layer_end.get(), name.get())
    ui_state.scales[idx].config(label=label)

    if canceled:
        return

    params = {
        'name': name.get(),
        'sigma_range': slider_value,
        'component_index': idx,
        'act_comp': components.X_comp[idx].detach().cpu().numpy(),
        'lat_comp': components.Z_comp[idx].detach().cpu().numpy(), # either Z or W
        'latent_space': model.latent_space_name(),
        'act_stdev': components.X_stdev[idx].item(),
        'lat_stdev': components.Z_stdev[idx].item(),
        'model_name': model_name,
        'output_class': ui_state.outclass.get(), # applied onto
        'decomposition': {
            'name': args.estimator,
            'components': args.components,
            'samples': args.n,
            'layer': args.layer,
            'class_name': state.component_class # computed from
        },
        'edit_type': ui_state.mode.get(),
        'truncation': ui_state.truncation.get(),
        'edit_start': ui_state.edit_layer_start.get(),
        'edit_end': ui_state.edit_layer_end.get() + 1, # show as inclusive, save as exclusive
        'example_seed': state.seed,
    }

    edit_mode_str = params['edit_type']
    if edit_mode_str == 'latent':
        edit_mode_str = model.latent_space_name().lower()

    comp_class = state.component_class
    appl_class = params['output_class']
    if comp_class != appl_class:
        comp_class = f'{comp_class}_onto_{appl_class}'

    file_ident = "{model}-{name}-{cls}-{est}-{mode}-{layer}-comp{idx}-range{start}-{end}".format(
        model=model_name,
        name=prettify_name(params['name']),
        cls=comp_class,
        est=args.estimator,
        mode=edit_mode_str,
        layer=args.layer,
        idx=idx,
        start=params['edit_start'],
        end=params['edit_end'],
    )

    out_dir = Path(__file__).parent / 'out' / 'directions'
    makedirs(out_dir / file_ident, exist_ok=True)

    with open(out_dir / f"{file_ident}.pkl", 'wb') as outfile:
        pickle.dump(params, outfile)

    print(f'Direction "{name.get()}" saved as "{file_ident}.pkl"')

    batch_size = ui_state.batch_size.get()
    len_padded = ((num_strips.get() - 1) // batch_size + 1) * batch_size
    orig_seed = state.seed

    reset_sliders()

    # Limit max resolution
    max_H = 512
    ratio = min(1.0, max_H / inst.output_shape[2])

    strips = [[] for _ in range(len_padded)]
    for b in range(0, len_padded, batch_size):
        # Resample
        resample_latent((orig_seed + b) % np.iinfo(np.int32).max)

        sigmas = np.linspace(slider_value, -slider_value, strip_width.get(), dtype=np.float32)
        for sid, sigma in enumerate(sigmas):
            ui_state.sliders[idx].set(sigma)
            
            # Advance and show results on screen
            on_draw()
            root.update()
            app.update()
            
            batch_res = (255*img).byte().permute(0, 2, 3, 1).detach().cpu().numpy()

            for i, data in enumerate(batch_res):
                # Save individual
                name_nodots = file_ident.replace('.', '_')
                outname = out_dir / file_ident / f"{name_nodots}_ex{b+i}_{sid}.png"
                im = Image.fromarray(data)
                im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
                im.save(outname)
                strips[b+i].append(data)

    for i, strip in enumerate(strips[:num_strips.get()]):
        print(f'Saving strip {i + 1}/{num_strips.get()}', end='\r', flush=True)
        data = np.hstack(pad_frames(strip))
        im = Image.fromarray(data)
        im = im.resize((int(ratio*im.size[0]), int(ratio*im.size[1])), Image.ANTIALIAS)
        im.save(out_dir / file_ident / f"{file_ident}_ex{i}.png")

    # Reset to original state
    resample_latent(orig_seed)
    ui_state.sliders[idx].set(slider_value)


# Shared by glumpy and tkinter
def handle_keypress(code):
    if code == 65307: # ESC
        shutdown()
    elif code == 65360: # HOME
        reset_sliders()
    elif code == 114: # R
        pass #reset_sliders()
    
def shutdown():
    global pending_close
    pending_close = True

def on_key_release(symbol, modifiers):
    handle_keypress(symbol)

if __name__=='__main__':
    setup_model()
    setup_ui()
    resample_latent()

    pending_close = False
    while not pending_close:
        root.update()
        app.update()
        on_draw()
        reposition_toolbar()

    root.destroy()