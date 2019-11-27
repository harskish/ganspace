import torch, sys, os, argparse, textwrap, numbers, numpy, json, PIL
from torchvision import transforms
from torch.utils.data import TensorDataset
from netdissect.progress import default_progress, post_progress, desc_progress
from netdissect.progress import verbose_progress, print_progress
from netdissect.nethook import edit_layers
from netdissect.zdataset import standard_z_sample
from netdissect.autoeval import autoimport_eval
from netdissect.easydict import EasyDict
from netdissect.modelconfig import create_instrumented_model

help_epilog = '''\
Example:

python -m netdissect.evalablate \
      --segmenter "netdissect.GanImageSegmenter(segvocab='lowres', segsizes=[160,288], segdiv='quad')" \
      --model "proggan.from_pth_file('models/lsun_models/${SCENE}_lsun.pth')" \
      --outdir dissect/dissectdir \
      --classname tree \
      --layer layer4 \
      --size 1000

Output layout:
dissectdir/layer5/ablation/mirror-iqr.json
{ class: "mirror",
  classnum: 43,
  pixel_total: 41342300,
  class_pixels: 1234531,
  layer: "layer5",
  ranking: "mirror-iqr",
  ablation_units: [341, 23, 12, 142, 83, ...]
  ablation_pixels: [143242, 132344, 429931, ...]
}

'''

def main():
    # Training settings
    def strpair(arg):
        p = tuple(arg.split(':'))
        if len(p) == 1:
            p = p + p
        return p

    parser = argparse.ArgumentParser(description='Ablation eval',
            epilog=textwrap.dedent(help_epilog),
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default=None,
                        help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
                        help='filename of .pth file for the model')
    parser.add_argument('--outdir', type=str, default='dissect', required=True,
                        help='directory for dissection output')
    parser.add_argument('--layer', type=strpair,
                        help='space-separated list of layer names to edit' + 
                        ', in the form layername[:reportedname]')
    parser.add_argument('--classname', type=str,
                        help='class name to ablate')
    parser.add_argument('--metric', type=str, default='iou',
                        help='ordering metric for selecting units')
    parser.add_argument('--unitcount', type=int, default=30,
                        help='number of units to ablate')
    parser.add_argument('--segmenter', type=str,
                        help='directory containing segmentation dataset')
    parser.add_argument('--netname', type=str, default=None,
                        help='name for network in generated reports')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='batch size for forward pass')
    parser.add_argument('--mixed_units', action='store_true', default=False,
                        help='true to keep alpha for non-zeroed units')
    parser.add_argument('--size', type=int, default=200,
                        help='number of images to test')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA usage')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='silences console output')
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # Set up console output
    verbose_progress(not args.quiet)

    # Speed up pytorch
    torch.backends.cudnn.benchmark = True

    # Set up CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.backends.cudnn.benchmark = True

    # Take defaults for model constructor etc from dissect.json settings.
    with open(os.path.join(args.outdir, 'dissect.json')) as f:
        dissection = EasyDict(json.load(f))
    if args.model is None:
        args.model = dissection.settings.model
    if args.pthfile is None:
        args.pthfile = dissection.settings.pthfile
    if args.segmenter is None:
        args.segmenter = dissection.settings.segmenter
    if args.layer is None:
        args.layer = dissection.settings.layers[0]
    args.layers = [args.layer]

    # Also load specific analysis
    layername = args.layer[1]
    if args.metric == 'iou':
        summary = dissection
    else:
        with open(os.path.join(args.outdir, layername, args.metric,
                args.classname, 'summary.json')) as f:
            summary = EasyDict(json.load(f))

    # Instantiate generator
    model = create_instrumented_model(args, gen=True, edit=True)
    if model is None:
        print('No model specified')
        sys.exit(1)

    # Instantiate model
    device = next(model.parameters()).device
    input_shape = model.input_shape

    # 4d input if convolutional, 2d input if first layer is linear.
    raw_sample = standard_z_sample(args.size, input_shape[1], seed=3).view(
            (args.size,) + input_shape[1:])
    dataset = TensorDataset(raw_sample)

    # Create the segmenter
    segmenter = autoimport_eval(args.segmenter)

    # Now do the actual work.
    labelnames, catnames = (
                segmenter.get_label_and_category_names(dataset))
    label_category = [catnames.index(c) if c in catnames else 0
            for l, c in labelnames]
    labelnum_from_name = {n[0]: i for i, n in enumerate(labelnames)}

    segloader = torch.utils.data.DataLoader(dataset,
                batch_size=args.batch_size, num_workers=10,
                pin_memory=(device.type == 'cuda'))

    # Index the dissection layers by layer name.

    # First, collect a baseline
    for l in model.ablation:
        model.ablation[l] = None

    # For each sort-order, do an ablation
    progress = default_progress()
    classname = args.classname
    classnum = labelnum_from_name[classname]

    # Get iou ranking from dissect.json
    iou_rankname = '%s-%s' % (classname, 'iou')
    dissect_layer = {lrec.layer: lrec for lrec in dissection.layers}
    iou_ranking = next(r for r in dissect_layer[layername].rankings
                if r.name == iou_rankname)

    # Get trained ranking from summary.json
    rankname = '%s-%s' % (classname, args.metric)
    summary_layer = {lrec.layer: lrec for lrec in summary.layers}
    ranking = next(r for r in summary_layer[layername].rankings
                if r.name == rankname)

    # Get ordering, first by ranking, then break ties by iou.
    ordering = [t[2] for t in sorted([(s1, s2, i)
        for i, (s1, s2) in enumerate(zip(ranking.score, iou_ranking.score))])]
    values = (-numpy.array(ranking.score))[ordering]
    if not args.mixed_units:
        values[...] = 1

    ablationdir = os.path.join(args.outdir, layername, 'fullablation')
    measurements = measure_full_ablation(segmenter, segloader,
            model, classnum, layername,
            ordering[:args.unitcount], values[:args.unitcount])
    measurements = measurements.cpu().numpy().tolist()
    os.makedirs(ablationdir, exist_ok=True)
    with open(os.path.join(ablationdir, '%s.json'%rankname), 'w') as f:
        json.dump(dict(
            classname=classname,
            classnum=classnum,
            baseline=measurements[0],
            layer=layername,
            metric=args.metric,
            ablation_units=ordering,
            ablation_values=values.tolist(),
            ablation_effects=measurements[1:]), f)

def measure_full_ablation(segmenter, loader, model, classnum, layer,
        ordering, values):
    '''
    Quick and easy counting of segmented pixels reduced by ablating units.
    '''
    progress = default_progress()
    device = next(model.parameters()).device
    feature_units = model.feature_shape[layer][1]
    feature_shape = model.feature_shape[layer][2:]
    repeats = len(ordering)
    total_scores = torch.zeros(repeats + 1)
    print(ordering)
    print(values.tolist())
    with torch.no_grad():
        for l in model.ablation:
            model.ablation[l] = None
        for i, [ibz] in enumerate(progress(loader)):
            ibz = ibz.cuda()
            for num_units in progress(range(len(ordering) + 1)):
                ablation = torch.zeros(feature_units, device=device)
                ablation[ordering[:num_units]] = torch.tensor(
                        values[:num_units]).to(ablation.device, ablation.dtype)
                model.ablation[layer] = ablation
                tensor_images = model(ibz)
                seg = segmenter.segment_batch(tensor_images, downsample=2)
                mask = (seg == classnum).max(1)[0]
                total_scores[num_units] += mask.sum().float().cpu()
    return total_scores

def count_segments(segmenter, loader, model):
    total_bincount = 0
    data_size = 0
    progress = default_progress()
    for i, batch in enumerate(progress(loader)):
        tensor_images = model(z_batch.to(device))
        seg = segmenter.segment_batch(tensor_images, downsample=2)
        bc = (seg + index[:, None, None, None] * self.num_classes).view(-1
                ).bincount(minlength=z_batch.shape[0] * self.num_classes)
        data_size += seg.shape[0] * seg.shape[2] * seg.shape[3]
        total_bincount += batch_label_counts.float().sum(0)
    normalized_bincount = total_bincount / data_size
    return normalized_bincount

if __name__ == '__main__':
    main()
