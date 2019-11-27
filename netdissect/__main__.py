import torch, sys, os, argparse, textwrap, numbers, numpy, json, PIL
from torchvision import transforms
from torch.utils.data import TensorDataset
from netdissect.progress import verbose_progress, print_progress
from netdissect import InstrumentedModel, BrodenDataset, dissect
from netdissect import MultiSegmentDataset, GeneratorSegRunner
from netdissect import ImageOnlySegRunner
from netdissect.parallelfolder import ParallelImageFolders
from netdissect.zdataset import z_dataset_for_model
from netdissect.autoeval import autoimport_eval
from netdissect.modelconfig import create_instrumented_model
from netdissect.pidfile import exit_if_job_done, mark_job_done

help_epilog = '''\
Example: to dissect three layers of the pretrained alexnet in torchvision:

python -m netdissect \\
        --model "torchvision.models.alexnet(pretrained=True)" \\
        --layers features.6:conv3 features.8:conv4 features.10:conv5 \\
        --imgsize 227 \\
        --outdir dissect/alexnet-imagenet

To dissect a progressive GAN model:

python -m netdissect \\
        --model "proggan.from_pth_file('model/churchoutdoor.pth')" \\
        --gan
'''

def main():
    # Training settings
    def strpair(arg):
        p = tuple(arg.split(':'))
        if len(p) == 1:
            p = p + p
        return p
    def intpair(arg):
        p = arg.split(',')
        if len(p) == 1:
            p = p + p
        return tuple(int(v) for v in p)

    parser = argparse.ArgumentParser(description='Net dissect utility',
            prog='python -m netdissect',
            epilog=textwrap.dedent(help_epilog),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None,
                        help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
                        help='filename of .pth file for the model')
    parser.add_argument('--unstrict', action='store_true', default=False,
                        help='ignore unexpected pth parameters')
    parser.add_argument('--submodule', type=str, default=None,
                        help='submodule to load from pthfile')
    parser.add_argument('--outdir', type=str, default='dissect',
                        help='directory for dissection output')
    parser.add_argument('--layers', type=strpair, nargs='+',
                        help='space-separated list of layer names to dissect' +
                        ', in the form layername[:reportedname]')
    parser.add_argument('--segments', type=str, default='dataset/broden',
                        help='directory containing segmentation dataset')
    parser.add_argument('--segmenter', type=str, default=None,
                        help='constructor for asegmenter class')
    parser.add_argument('--download', action='store_true', default=False,
                        help='downloads Broden dataset if needed')
    parser.add_argument('--imagedir', type=str, default=None,
                        help='directory containing image-only dataset')
    parser.add_argument('--imgsize', type=intpair, default=(227, 227),
                        help='input image size to use')
    parser.add_argument('--netname', type=str, default=None,
                        help='name for network in generated reports')
    parser.add_argument('--meta', type=str, nargs='+',
                        help='json files of metadata to add to report')
    parser.add_argument('--merge', type=str,
                        help='json file of unit data to merge in report')
    parser.add_argument('--examples', type=int, default=20,
                        help='number of image examples per unit')
    parser.add_argument('--size', type=int, default=10000,
                        help='dataset subset size to use')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for forward pass')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='number of DataLoader workers')
    parser.add_argument('--quantile_threshold', type=strfloat, default=None,
                        choices=[FloatRange(0.0, 1.0), 'iqr'],
                        help='quantile to use for masks')
    parser.add_argument('--no-labels', action='store_true', default=False,
                        help='disables labeling of units')
    parser.add_argument('--maxiou', action='store_true', default=False,
                        help='enables maxiou calculation')
    parser.add_argument('--covariance', action='store_true', default=False,
                        help='enables covariance calculation')
    parser.add_argument('--rank_all_labels', action='store_true', default=False,
                        help='include low-information labels in rankings')
    parser.add_argument('--no-images', action='store_true', default=False,
                        help='disables generation of unit images')
    parser.add_argument('--no-report', action='store_true', default=False,
                        help='disables generation report summary')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA usage')
    parser.add_argument('--gen', action='store_true', default=False,
                        help='test a generator model (e.g., a GAN)')
    parser.add_argument('--gan', action='store_true', default=False,
                        help='synonym for --gen')
    parser.add_argument('--perturbation', default=None,
                        help='filename of perturbation attack to apply')
    parser.add_argument('--add_scale_offset', action='store_true', default=None,
                        help='offsets masks according to stride and padding')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='silences console output')
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    args.images = not args.no_images
    args.report = not args.no_report
    args.labels = not args.no_labels
    if args.gan:
        args.gen = args.gan

    # Set up console output
    verbose_progress(not args.quiet)

    # Exit right away if job is already done or being done.
    if args.outdir is not None:
        exit_if_job_done(args.outdir)

    # Speed up pytorch
    torch.backends.cudnn.benchmark = True

    # Special case: download flag without model to test.
    if args.model is None and args.download:
        from netdissect.broden import ensure_broden_downloaded
        for resolution in [224, 227, 384]:
            ensure_broden_downloaded(args.segments, resolution, 1)
        from netdissect.segmenter import ensure_upp_segmenter_downloaded
        ensure_upp_segmenter_downloaded('dataset/segmodel')
        sys.exit(0)

    # Help if broden is not present
    if not args.gen and not args.imagedir and not os.path.isdir(args.segments):
        print_progress('Segmentation dataset not found at %s.' % args.segments)
        print_progress('Specify dataset directory using --segments [DIR]')
        print_progress('To download Broden, run: netdissect --download')
        sys.exit(1)

    # Default segmenter class
    if args.gen and args.segmenter is None:
        args.segmenter = ("netdissect.segmenter.UnifiedParsingSegmenter(" +
                "segsizes=[256], segdiv='quad')")

    # Default threshold
    if args.quantile_threshold is None:
        if args.gen:
            args.quantile_threshold = 'iqr'
        else:
            args.quantile_threshold = 0.005

    # Set up CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    # Construct the network with specified layers instrumented
    if args.model is None:
        print_progress('No model specified')
        sys.exit(1)
    model = create_instrumented_model(args)

    # Update any metadata from files, if any
    meta = getattr(model, 'meta', {})
    if args.meta:
        for mfilename in args.meta:
            with open(mfilename) as f:
                meta.update(json.load(f))

    # Load any merge data from files
    mergedata = None
    if args.merge:
        with open(args.merge) as f:
            mergedata = json.load(f)

    # Set up the output directory, verify write access
    if args.outdir is None:
        args.outdir = os.path.join('dissect', type(model).__name__)
        exit_if_job_done(args.outdir)
        print_progress('Writing output into %s.' % args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    train_dataset = None

    if not args.gen:
        # Load dataset for classifier case.
        # Load perturbation
        perturbation = numpy.load(args.perturbation
                ) if args.perturbation else None
        segrunner = None

        # Load broden dataset
        if args.imagedir is not None:
            dataset = try_to_load_images(args.imagedir, args.imgsize,
                    perturbation, args.size)
            segrunner = ImageOnlySegRunner(dataset)
        else:
            dataset = try_to_load_broden(args.segments, args.imgsize, 1,
                perturbation, args.download, args.size)
        if dataset is None:
            dataset = try_to_load_multiseg(args.segments, args.imgsize,
                    perturbation, args.size)
        if dataset is None:
            print_progress('No segmentation dataset found in %s',
                    args.segments)
            print_progress('use --download to download Broden.')
            sys.exit(1)
    else:
        # For segmenter case the dataset is just a random z
        dataset = z_dataset_for_model(model, args.size)
        train_dataset = z_dataset_for_model(model, args.size, seed=2)
        segrunner = GeneratorSegRunner(autoimport_eval(args.segmenter))

    # Run dissect
    dissect(args.outdir, model, dataset,
            train_dataset=train_dataset,
            segrunner=segrunner,
            examples_per_unit=args.examples,
            netname=args.netname,
            quantile_threshold=args.quantile_threshold,
            meta=meta,
            merge=mergedata,
            make_images=args.images,
            make_labels=args.labels,
            make_maxiou=args.maxiou,
            make_covariance=args.covariance,
            make_report=args.report,
            make_row_images=args.images,
            make_single_images=True,
            rank_all_labels=args.rank_all_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            settings=vars(args))

    # Mark the directory so that it's not done again.
    mark_job_done(args.outdir)

class AddPerturbation(object):
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, pic):
        if self.perturbation is None:
            return pic
        # Convert to a numpy float32 array
        npyimg = numpy.array(pic, numpy.uint8, copy=False
                ).astype(numpy.float32)
        # Center the perturbation
        oy, ox = ((self.perturbation.shape[d] - npyimg.shape[d]) // 2
                for d in [0, 1])
        npyimg += self.perturbation[
                oy:oy+npyimg.shape[0], ox:ox+npyimg.shape[1]]
        # Pytorch conventions: as a float it should be [0..1]
        npyimg.clip(0, 255, npyimg)
        return npyimg / 255.0

def test_dissection():
    verbose_progress(True)
    from torchvision.models import alexnet
    from torchvision import transforms
    model = InstrumentedModel(alexnet(pretrained=True))
    model.eval()
    # Load an alexnet
    model.retain_layers([
        ('features.0', 'conv1'),
        ('features.3', 'conv2'),
        ('features.6', 'conv3'),
        ('features.8', 'conv4'),
        ('features.10', 'conv5') ])
    # load broden dataset
    bds = BrodenDataset('dataset/broden',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=100)
    # run dissect
    dissect('dissect/test', model, bds,
            examples_per_unit=10)

def try_to_load_images(directory, imgsize, perturbation, size):
    # Load plain image dataset
    # TODO: allow other normalizations.
    return ParallelImageFolders(
            [directory],
            transform=transforms.Compose([
                transforms.Resize(imgsize),
                AddPerturbation(perturbation),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=size)

def try_to_load_broden(directory, imgsize, broden_version, perturbation,
        download, size):
    # Load broden dataset
    ds_resolution = (224 if max(imgsize) <= 224 else
                     227 if max(imgsize) <= 227 else 384)
    if not os.path.isfile(os.path.join(directory,
           'broden%d_%d' % (broden_version, ds_resolution), 'index.csv')):
        return None
    return BrodenDataset(directory,
            resolution=ds_resolution,
            download=download,
            broden_version=broden_version,
            transform=transforms.Compose([
                transforms.Resize(imgsize),
                AddPerturbation(perturbation),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=size)

def try_to_load_multiseg(directory, imgsize, perturbation, size):
    if not os.path.isfile(os.path.join(directory, 'labelnames.json')):
        return None
    minsize = min(imgsize) if hasattr(imgsize, '__iter__') else imgsize
    return MultiSegmentDataset(directory,
            transform=(transforms.Compose([
                transforms.Resize(minsize),
                transforms.CenterCrop(imgsize),
                AddPerturbation(perturbation),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            transforms.Compose([
                transforms.Resize(minsize, interpolation=PIL.Image.NEAREST),
                transforms.CenterCrop(imgsize)])),
            size=size)

def add_scale_offset_info(model, layer_names):
    '''
    Creates a 'scale_offset' property on the model which guesses
    how to offset the featuremap, in cases where the convolutional
    padding does not exacly correspond to keeping featuremap pixels
    centered on the downsampled regions of the input.  This mainly
    shows up in AlexNet: ResNet and VGG pad convolutions to keep
    them centered and do not need this.
    '''
    model.scale_offset = {}
    seen = set()
    sequence = []
    aka_map = {}
    for name in layer_names:
        aka = name
        if not isinstance(aka, str):
            name, aka = name
        aka_map[name] = aka
    for name, layer in model.named_modules():
        sequence.append(layer)
        if name in aka_map:
            seen.add(name)
            aka = aka_map[name]
            model.scale_offset[aka] = sequence_scale_offset(sequence)
    for name in aka_map:
        assert name in seen, ('Layer %s not found' % name)

def dilation_scale_offset(dilations):
    '''Composes a list of (k, s, p) into a single total scale and offset.'''
    if len(dilations) == 0:
        return (1, 0)
    scale, offset = dilation_scale_offset(dilations[1:])
    kernel, stride, padding = dilations[0]
    scale *= stride
    offset *= stride
    offset += (kernel - 1) / 2.0 - padding
    return scale, offset

def dilations(modulelist):
    '''Converts a list of modules to (kernel_size, stride, padding)'''
    result = []
    for module in modulelist:
        settings = tuple(getattr(module, n, d)
            for n, d in (('kernel_size', 1), ('stride', 1), ('padding', 0)))
        settings = (((s, s) if not isinstance(s, tuple) else s)
            for s in settings)
        if settings != ((1, 1), (1, 1), (0, 0)):
            result.append(zip(*settings))
    return zip(*result)

def sequence_scale_offset(modulelist):
    '''Returns (yscale, yoffset), (xscale, xoffset) given a list of modules'''
    return tuple(dilation_scale_offset(d) for d in dilations(modulelist))


def strfloat(s):
    try:
        return float(s)
    except:
        return s

class FloatRange(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return isinstance(other, float) and self.start <= other <= self.end
    def __repr__(self):
        return '[%g-%g]' % (self.start, self.end)

# Many models use this normalization.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STDEV = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    main()
