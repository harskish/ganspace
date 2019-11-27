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
      --segmenter "netdissect.segmenter.UnifiedParsingSegmenter(segsizes=[256], segdiv='quad')" \
      --model "proggan.from_pth_file('models/lsun_models/${SCENE}_lsun.pth')" \
      --outdir dissect/dissectdir \
      --classes mirror coffeetable tree \
      --layers layer4 \
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
    parser.add_argument('--layers', type=strpair, nargs='+',
                        help='space-separated list of layer names to edit' + 
                        ', in the form layername[:reportedname]')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='space-separated list of class names to ablate')
    parser.add_argument('--metric', type=str, default='iou',
                        help='ordering metric for selecting units')
    parser.add_argument('--unitcount', type=int, default=30,
                        help='number of units to ablate')
    parser.add_argument('--segmenter', type=str,
                        help='directory containing segmentation dataset')
    parser.add_argument('--netname', type=str, default=None,
                        help='name for network in generated reports')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='batch size for forward pass')
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

    # Instantiate generator
    model = create_instrumented_model(args, gen=True, edit=True)
    if model is None:
        print('No model specified')
        sys.exit(1)

    # Instantiate model
    device = next(model.parameters()).device
    input_shape = model.input_shape

    # 4d input if convolutional, 2d input if first layer is linear.
    raw_sample = standard_z_sample(args.size, input_shape[1], seed=2).view(
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
    dissect_layer = {lrec.layer: lrec for lrec in dissection.layers}

    # First, collect a baseline
    for l in model.ablation:
        model.ablation[l] = None

    # For each sort-order, do an ablation
    progress = default_progress()
    for classname in progress(args.classes):
        post_progress(c=classname)
        for layername in progress(model.ablation):
            post_progress(l=layername)
            rankname = '%s-%s' % (classname, args.metric)
            classnum = labelnum_from_name[classname]
            try:
                ranking = next(r for r in dissect_layer[layername].rankings
                        if r.name == rankname)
            except:
                print('%s not found' % rankname)
                sys.exit(1)
            ordering = numpy.argsort(ranking.score)
            # Check if already done
            ablationdir = os.path.join(args.outdir, layername, 'pixablation')
            if os.path.isfile(os.path.join(ablationdir, '%s.json'%rankname)):
                with open(os.path.join(ablationdir, '%s.json'%rankname)) as f:
                    data = EasyDict(json.load(f))
                # If the unit ordering is not the same, something is wrong
                if not all(a == o
                        for a, o in zip(data.ablation_units, ordering)):
                    continue
                if len(data.ablation_effects) >= args.unitcount:
                    continue # file already done.
                measurements = data.ablation_effects
            measurements = measure_ablation(segmenter, segloader,
                    model, classnum, layername, ordering[:args.unitcount])
            measurements = measurements.cpu().numpy().tolist()
            os.makedirs(ablationdir, exist_ok=True)
            with open(os.path.join(ablationdir, '%s.json'%rankname), 'w') as f:
                json.dump(dict(
                    classname=classname,
                    classnum=classnum,
                    baseline=measurements[0],
                    layer=layername,
                    metric=args.metric,
                    ablation_units=ordering.tolist(),
                    ablation_effects=measurements[1:]), f)

def measure_ablation(segmenter, loader, model, classnum, layer, ordering):
    total_bincount = 0
    data_size = 0
    device = next(model.parameters()).device
    progress = default_progress()
    for l in model.ablation:
        model.ablation[l] = None
    feature_units = model.feature_shape[layer][1]
    feature_shape = model.feature_shape[layer][2:]
    repeats = len(ordering)
    total_scores = torch.zeros(repeats + 1)
    for i, batch in enumerate(progress(loader)):
        z_batch = batch[0]
        model.ablation[layer] = None
        tensor_images = model(z_batch.to(device))
        seg = segmenter.segment_batch(tensor_images, downsample=2)
        mask = (seg == classnum).max(1)[0]
        downsampled_seg = torch.nn.functional.adaptive_avg_pool2d(
                mask.float()[:,None,:,:], feature_shape)[:,0,:,:]
        total_scores[0] += downsampled_seg.sum().cpu()
        # Now we need to do an intervention for every location
        # that had a nonzero downsampled_seg, if any.
        interventions_needed = downsampled_seg.nonzero()
        location_count = len(interventions_needed)
        if location_count == 0:
            continue
        interventions_needed = interventions_needed.repeat(repeats, 1)
        inter_z = batch[0][interventions_needed[:,0]].to(device)
        inter_chan = torch.zeros(repeats, location_count, feature_units,
                device=device)
        for j, u in enumerate(ordering):
            inter_chan[j:, :, u] = 1
        inter_chan = inter_chan.view(len(inter_z), feature_units)
        inter_loc = interventions_needed[:,1:]
        scores = torch.zeros(len(inter_z))
        batch_size = len(batch[0])
        for j in range(0, len(inter_z), batch_size):
            ibz = inter_z[j:j+batch_size]
            ibl = inter_loc[j:j+batch_size].t()
            imask = torch.zeros((len(ibz),) + feature_shape, device=ibz.device)
            imask[(torch.arange(len(ibz)),) + tuple(ibl)] = 1
            ibc = inter_chan[j:j+batch_size]
            model.ablation[layer] = (
                    imask.float()[:,None,:,:] * ibc[:,:,None,None])
            tensor_images = model(ibz)
            seg = segmenter.segment_batch(tensor_images, downsample=2)
            mask = (seg == classnum).max(1)[0]
            downsampled_iseg = torch.nn.functional.adaptive_avg_pool2d(
                    mask.float()[:,None,:,:], feature_shape)[:,0,:,:]
            scores[j:j+batch_size] = downsampled_iseg[
                    (torch.arange(len(ibz)),) + tuple(ibl)]
        scores = scores.view(repeats, location_count).sum(1)
        total_scores[1:] += scores
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
