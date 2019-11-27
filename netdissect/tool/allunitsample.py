'''
A simple tool to generate sample of output of a GAN,
subject to filtering, sorting, or intervention.
'''

import torch, numpy, os, argparse, sys, shutil, errno, numbers
from PIL import Image
from torch.utils.data import TensorDataset
from netdissect.zdataset import standard_z_sample
from netdissect.progress import default_progress, verbose_progress
from netdissect.autoeval import autoimport_eval
from netdissect.workerpool import WorkerBase, WorkerPool
from netdissect.nethook import retain_layers
from netdissect.runningstats import RunningTopK

def main():
    parser = argparse.ArgumentParser(description='GAN sample making utility')
    parser.add_argument('--model', type=str, default=None,
            help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
            help='filename of .pth file for the model')
    parser.add_argument('--outdir', type=str, default='images',
            help='directory for image output')
    parser.add_argument('--size', type=int, default=100,
            help='number of images to output')
    parser.add_argument('--test_size', type=int, default=None,
            help='number of images to test')
    parser.add_argument('--layer', type=str, default=None,
            help='layer to inspect')
    parser.add_argument('--seed', type=int, default=1,
            help='seed')
    parser.add_argument('--quiet', action='store_true', default=False,
            help='silences console output')
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    verbose_progress(not args.quiet)

    # Instantiate the model
    model = autoimport_eval(args.model)
    if args.pthfile is not None:
        data = torch.load(args.pthfile)
        if 'state_dict' in data:
            meta = {}
            for key in data:
                if isinstance(data[key], numbers.Number):
                    meta[key] = data[key]
            data = data['state_dict']
        model.load_state_dict(data)
    # Unwrap any DataParallel-wrapped model
    if isinstance(model, torch.nn.DataParallel):
        model = next(model.children())
    # Examine first conv in model to determine input feature size.
    first_layer = [c for c in model.modules()
            if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                torch.nn.Linear))][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        z_channels = first_layer.in_channels
        spatialdims = (1, 1)
    else:
        z_channels = first_layer.in_features
        spatialdims = ()
    # Instrument the model
    retain_layers(model, [args.layer])
    model.cuda()

    if args.test_size is None:
        args.test_size = args.size * 20
    z_universe = standard_z_sample(args.test_size, z_channels,
            seed=args.seed)
    z_universe = z_universe.view(tuple(z_universe.shape) + spatialdims)
    indexes = get_all_highest_znums(
            model, z_universe, args.size, seed=args.seed)
    save_chosen_unit_images(args.outdir, model, z_universe, indexes,
            lightbox=True)


def get_all_highest_znums(model, z_universe, size,
        batch_size=10, seed=1):
    # The model should have been instrumented already
    retained_items = list(model.retained.items())
    assert len(retained_items) == 1
    layer = retained_items[0][0]
    # By default, a 10% sample
    progress = default_progress()
    num_units = None
    with torch.no_grad():
        # Pass 1: collect max activation stats
        z_loader = torch.utils.data.DataLoader(TensorDataset(z_universe),
                    batch_size=batch_size, num_workers=2,
                    pin_memory=True)
        rtk = RunningTopK(k=size)
        for [z] in progress(z_loader, desc='Finding max activations'):
            z = z.cuda()
            model(z)
            feature = model.retained[layer]
            num_units = feature.shape[1]
            max_feature = feature.view(
                    feature.shape[0], num_units, -1).max(2)[0]
            rtk.add(max_feature)
        td, ti = rtk.result()
        highest = ti.sort(1)[0]
    return highest

def save_chosen_unit_images(dirname, model, z_universe, indices,
        shared_dir="shared_images",
        unitdir_template="unit_{}",
        name_template="image_{}.jpg",
        lightbox=False, batch_size=50, seed=1):
    all_indices = torch.unique(indices.view(-1), sorted=True)
    z_sample = z_universe[all_indices]
    progress = default_progress()
    sdir = os.path.join(dirname, shared_dir)
    created_hashdirs = set()
    for index in range(len(z_universe)):
        hd = hashdir(index)
        if hd not in created_hashdirs:
            created_hashdirs.add(hd)
            os.makedirs(os.path.join(sdir, hd), exist_ok=True)
    with torch.no_grad():
        # Pass 2: now generate images
        z_loader = torch.utils.data.DataLoader(TensorDataset(z_sample),
                    batch_size=batch_size, num_workers=2,
                    pin_memory=True)
        saver = WorkerPool(SaveImageWorker)
        for batch_num, [z] in enumerate(progress(z_loader,
                desc='Saving images')):
            z = z.cuda()
            start_index = batch_num * batch_size
            im = ((model(z) + 1) / 2 * 255).clamp(0, 255).byte().permute(
                    0, 2, 3, 1).cpu()
            for i in range(len(im)):
                index = all_indices[i + start_index].item()
                filename = os.path.join(sdir, hashdir(index),
                        name_template.format(index))
                saver.add(im[i].numpy(), filename)
        saver.join()
    linker = WorkerPool(MakeLinkWorker)
    for u in progress(range(len(indices)), desc='Making links'):
        udir = os.path.join(dirname, unitdir_template.format(u))
        os.makedirs(udir, exist_ok=True)
        for r in range(indices.shape[1]):
            index = indices[u,r].item()
            fn = name_template.format(index)
            # sourcename = os.path.join('..', shared_dir, fn)
            sourcename = os.path.join(sdir, hashdir(index), fn)
            targname = os.path.join(udir, fn)
            linker.add(sourcename, targname)
        if lightbox:
            copy_lightbox_to(udir)
    linker.join()

def copy_lightbox_to(dirname):
   srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
   shutil.copy(os.path.join(srcdir, 'lightbox.html'),
           os.path.join(dirname, '+lightbox.html'))

def hashdir(index):
    # To keep the number of files the shared directory lower, split it
    # into 100 subdirectories named as follows.
    return '%02d' % (index % 100)

class SaveImageWorker(WorkerBase):
    # Saving images can be sped up by sending jpeg encoding and
    # file-writing work to a pool.
    def work(self, data, filename):
        Image.fromarray(data).save(filename, optimize=True, quality=100)

class MakeLinkWorker(WorkerBase):
    # Creating symbolic links is a bit slow and can be done faster
    # in parallel rather than waiting for each to be created.
    def work(self, sourcename, targname):
        try:
            os.link(sourcename, targname)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(targname)
                os.link(sourcename, targname)
            else:
                raise

class MakeSyminkWorker(WorkerBase):
    # Creating symbolic links is a bit slow and can be done faster
    # in parallel rather than waiting for each to be created.
    def work(self, sourcename, targname):
        try:
            os.symlink(sourcename, targname)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(targname)
                os.symlink(sourcename, targname)
            else:
                raise

if __name__ == '__main__':
    main()
