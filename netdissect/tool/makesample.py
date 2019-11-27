'''
A simple tool to generate sample of output of a GAN,
subject to filtering, sorting, or intervention.
'''

import torch, numpy, os, argparse, numbers, sys, shutil
from PIL import Image
from torch.utils.data import TensorDataset
from netdissect.zdataset import standard_z_sample
from netdissect.progress import default_progress, verbose_progress
from netdissect.autoeval import autoimport_eval
from netdissect.workerpool import WorkerBase, WorkerPool
from netdissect.nethook import edit_layers, retain_layers

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
    parser.add_argument('--maximize_units', type=int, nargs='+', default=None,
            help='units to maximize')
    parser.add_argument('--ablate_units', type=int, nargs='+', default=None,
            help='units to ablate')
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
    # Instrument the model if needed
    if args.maximize_units is not None:
        retain_layers(model, [args.layer])
    model.cuda()

    # Get the sample of z vectors
    if args.maximize_units is None:
        indexes = torch.arange(args.size)
        z_sample = standard_z_sample(args.size, z_channels, seed=args.seed)
        z_sample = z_sample.view(tuple(z_sample.shape) + spatialdims)
    else:
        # By default, if maximizing units, get a 'top 5%' sample.
        if args.test_size is None:
            args.test_size = args.size * 20
        z_universe = standard_z_sample(args.test_size, z_channels,
                seed=args.seed)
        z_universe = z_universe.view(tuple(z_universe.shape) + spatialdims)
        indexes = get_highest_znums(model, z_universe, args.maximize_units,
                args.size, seed=args.seed)
        z_sample = z_universe[indexes]

    if args.ablate_units:
        edit_layers(model, [args.layer])
        dims = max(2, max(args.ablate_units) + 1) # >=2 to avoid broadcast
        model.ablation[args.layer] = torch.zeros(dims)
        model.ablation[args.layer][args.ablate_units] = 1

    save_znum_images(args.outdir, model, z_sample, indexes,
            args.layer, args.ablate_units)
    copy_lightbox_to(args.outdir)


def get_highest_znums(model, z_universe, max_units, size,
        batch_size=100, seed=1):
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
        scores = []
        for [z] in progress(z_loader, desc='Finding max activations'):
            z = z.cuda()
            model(z)
            feature = model.retained[layer]
            num_units = feature.shape[1]
            max_feature = feature[:, max_units, ...].view(
                    feature.shape[0], len(max_units), -1).max(2)[0]
            total_feature = max_feature.sum(1)
            scores.append(total_feature.cpu())
        scores = torch.cat(scores, 0)
        highest = (-scores).sort(0)[1][:size].sort(0)[0]
    return highest


def save_znum_images(dirname, model, z_sample, indexes, layer, ablated_units,
        name_template="image_{}.png", lightbox=False, batch_size=100, seed=1):
    progress = default_progress()
    os.makedirs(dirname, exist_ok=True)
    with torch.no_grad():
        # Pass 2: now generate images
        z_loader = torch.utils.data.DataLoader(TensorDataset(z_sample),
                    batch_size=batch_size, num_workers=2,
                    pin_memory=True)
        saver = WorkerPool(SaveImageWorker)
        if ablated_units is not None:
            dims = max(2, max(ablated_units) + 1) # >=2 to avoid broadcast
            mask = torch.zeros(dims)
            mask[ablated_units] = 1
            model.ablation[layer] = mask[None,:,None,None].cuda()
        for batch_num, [z] in enumerate(progress(z_loader,
                desc='Saving images')):
            z = z.cuda()
            start_index = batch_num * batch_size
            im = ((model(z) + 1) / 2 * 255).clamp(0, 255).byte().permute(
                    0, 2, 3, 1).cpu()
            for i in range(len(im)):
                index = i + start_index
                if indexes is not None:
                    index = indexes[index].item()
                filename = os.path.join(dirname, name_template.format(index))
                saver.add(im[i].numpy(), filename)
    saver.join()

def copy_lightbox_to(dirname):
   srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
   shutil.copy(os.path.join(srcdir, 'lightbox.html'),
           os.path.join(dirname, '+lightbox.html'))

class SaveImageWorker(WorkerBase):
    def work(self, data, filename):
        Image.fromarray(data).save(filename, optimize=True, quality=100)

if __name__ == '__main__':
    main()
