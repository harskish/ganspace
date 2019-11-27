# Instantiate the segmenter gadget.
# Instantiate the GAN to optimize over
# Instrument the GAN for editing and optimization.
# Read quantile stats to learn 99.9th percentile for each unit,
# and also the 0.01th percentile.
# Read the median activation conditioned on door presence.

import os, sys, numpy, torch, argparse, skimage, json, shutil
from PIL import Image
from torch.utils.data import TensorDataset
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from scipy.ndimage.morphology import binary_dilation

import netdissect.zdataset
import netdissect.nethook
from netdissect.dissection import safe_dir_name
from netdissect.progress import verbose_progress, default_progress
from netdissect.progress import print_progress, desc_progress, post_progress
from netdissect.easydict import EasyDict
from netdissect.workerpool import WorkerPool, WorkerBase
from netdissect.runningstats import RunningQuantile
from netdissect.pidfile import pidfile_taken
from netdissect.modelconfig import create_instrumented_model
from netdissect.autoeval import autoimport_eval

def main():
    parser = argparse.ArgumentParser(description='ACE optimization utility',
            prog='python -m netdissect.aceoptimize')
    parser.add_argument('--model', type=str, default=None,
                        help='constructor for the model to test')
    parser.add_argument('--pthfile', type=str, default=None,
                        help='filename of .pth file for the model')
    parser.add_argument('--segmenter', type=str, default=None,
                        help='constructor for asegmenter class')
    parser.add_argument('--classname', type=str, default=None,
                        help='intervention classname')
    parser.add_argument('--layer', type=str, default='layer4',
                        help='layer name')
    parser.add_argument('--search_size', type=int, default=10000,
                        help='size of search for finding training locations')
    parser.add_argument('--train_size', type=int, default=1000,
                        help='size of training set')
    parser.add_argument('--eval_size', type=int, default=200,
                        help='size of eval set')
    parser.add_argument('--inference_batch_size', type=int, default=10,
                        help='forward pass batch size')
    parser.add_argument('--train_batch_size', type=int, default=2,
                        help='backprop pass batch size')
    parser.add_argument('--train_update_freq', type=int, default=10,
                        help='number of batches for each training update')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='number of epochs of training')
    parser.add_argument('--l2_lambda', type=float, default=0.005,
                        help='l2 regularizer hyperparameter')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='reruns eval only on trained snapshots')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA usage')
    parser.add_argument('--no-cache', action='store_true', default=False,
                        help='disables reading of cache')
    parser.add_argument('--outdir', type=str, default=None,
                        help='dissection directory')
    parser.add_argument('--variant', type=str, default=None,
                        help='experiment variant')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    run_command(args)

def run_command(args):
    verbose_progress(True)
    progress = default_progress()
    classname = args.classname # 'door'
    layer = args.layer # 'layer4'
    num_eval_units = 20

    assert os.path.isfile(os.path.join(args.outdir, 'dissect.json')), (
            "Should be a dissection directory")

    if args.variant is None:
        args.variant = 'ace'

    if args.l2_lambda != 0.005:
        args.variant = '%s_reg%g' % (args.variant, args.l2_lambda)

    cachedir = os.path.join(args.outdir, safe_dir_name(layer), args.variant,
            classname)

    if pidfile_taken(os.path.join(cachedir, 'lock.pid'), True):
        sys.exit(0)

    # Take defaults for model constructor etc from dissect.json settings.
    with open(os.path.join(args.outdir, 'dissect.json')) as f:
        dissection = EasyDict(json.load(f))
    if args.model is None:
        args.model = dissection.settings.model
    if args.pthfile is None:
        args.pthfile = dissection.settings.pthfile
    if args.segmenter is None:
        args.segmenter = dissection.settings.segmenter
    # Default segmenter class
    if args.segmenter is None:
        args.segmenter = ("netdissect.segmenter.UnifiedParsingSegmenter(" +
                "segsizes=[256], segdiv='quad')")

    if (not args.no_cache and
        os.path.isfile(os.path.join(cachedir, 'snapshots', 'epoch-%d.npy' % (
            args.train_epochs - 1))) and
        os.path.isfile(os.path.join(cachedir, 'report.json'))):
        print('%s already done' % cachedir)
        sys.exit(0)

    os.makedirs(cachedir, exist_ok=True)

    # Instantiate generator
    model = create_instrumented_model(args, gen=True, edit=True,
            layers=[args.layer])
    if model is None:
        print('No model specified')
        sys.exit(1)
    # Instantiate segmenter
    segmenter = autoimport_eval(args.segmenter)
    labelnames, catname = segmenter.get_label_and_category_names()
    classnum = [i for i, (n, c) in enumerate(labelnames) if n == classname][0]
    num_classes = len(labelnames)
    with open(os.path.join(cachedir, 'labelnames.json'), 'w') as f:
        json.dump(labelnames, f, indent=1)

    # Sample sets for training.
    full_sample = netdissect.zdataset.z_sample_for_model(model,
            args.search_size, seed=10)
    second_sample = netdissect.zdataset.z_sample_for_model(model,
            args.search_size, seed=11)
    # Load any cached data.
    cache_filename = os.path.join(cachedir, 'corpus.npz')
    corpus = EasyDict()
    try:
        if not args.no_cache:
            corpus = EasyDict({k: torch.from_numpy(v)
                for k, v in numpy.load(cache_filename).items()})
    except:
        pass

    # The steps for the computation.
    compute_present_locations(args, corpus, cache_filename,
            model, segmenter, classnum, full_sample)
    compute_mean_present_features(args, corpus, cache_filename, model)
    compute_feature_quantiles(args, corpus, cache_filename, model, full_sample)
    compute_candidate_locations(args, corpus, cache_filename, model, segmenter,
            classnum, second_sample)
    # visualize_training_locations(args, corpus, cachedir, model)
    init_ablation = initial_ablation(args, args.outdir)
    scores = train_ablation(args, corpus, cache_filename,
            model, segmenter, classnum, init_ablation)
    summarize_scores(args, corpus, cachedir, layer, classname,
            args.variant, scores)
    if args.variant == 'ace':
        add_ace_ranking_to_dissection(args.outdir, layer, classname, scores)
    # TODO: do some evaluation.

class SaveImageWorker(WorkerBase):
    def work(self, data, filename):
        Image.fromarray(data).save(filename, optimize=True, quality=80)

def plot_heatmap(output_filename, data, size=256):
    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.imshow(data, cmap='hot', aspect='equal', interpolation='nearest',
              vmin=-1, vmax=1)
    canvas.print_figure(output_filename, format='png')


def draw_heatmap(output_filename, data, size=256):
    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.imshow(data, cmap='hot', aspect='equal', interpolation='nearest',
              vmin=-1, vmax=1)
    canvas.draw()       # draw the canvas, cache the renderer
    image = numpy.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
            (size, size, 3))
    return image

def compute_present_locations(args, corpus, cache_filename,
        model, segmenter, classnum, full_sample):
    # Phase 1.  Identify a set of locations where there are doorways.
    # Segment the image and find featuremap pixels that maximize the number
    # of doorway pixels under the featuremap pixel.
    if all(k in corpus for k in ['present_indices',
            'object_present_sample', 'object_present_location',
            'object_location_popularity', 'weighted_mean_present_feature']):
        return
    progress = default_progress()
    feature_shape = model.feature_shape[args.layer][2:]
    num_locations = numpy.prod(feature_shape).item()
    num_units = model.feature_shape[args.layer][1]
    with torch.no_grad():
        weighted_feature_sum = torch.zeros(num_units).cuda()
        object_presence_scores = []
        for [zbatch] in progress(
                torch.utils.data.DataLoader(TensorDataset(full_sample),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Object pool"):
            zbatch = zbatch.cuda()
            tensor_image = model(zbatch)
            segmented_image = segmenter.segment_batch(tensor_image,
                    downsample=2)
            mask = (segmented_image == classnum).max(1)[0]
            score = torch.nn.functional.adaptive_avg_pool2d(
                    mask.float(), feature_shape)
            object_presence_scores.append(score.cpu())
            feat = model.retained_layer(args.layer)
            weighted_feature_sum += (feat * score[:,None,:,:]).view(
                    feat.shape[0],feat.shape[1], -1).sum(2).sum(0)
        object_presence_at_feature = torch.cat(object_presence_scores)
        object_presence_at_image, object_location_in_image = (
                object_presence_at_feature.view(args.search_size, -1).max(1))
        best_presence_scores, best_presence_images = torch.sort(
                -object_presence_at_image)
        all_present_indices = torch.sort(
                best_presence_images[:(args.train_size+args.eval_size)])[0]
        corpus.present_indices = all_present_indices[:args.train_size]
        corpus.object_present_sample = full_sample[corpus.present_indices]
        corpus.object_present_location = object_location_in_image[
                corpus.present_indices]
        corpus.object_location_popularity = torch.bincount(
            corpus.object_present_location,
            minlength=num_locations)
        corpus.weighted_mean_present_feature = (weighted_feature_sum.cpu() / (
            1e-20 + object_presence_at_feature.view(-1).sum()))
        corpus.eval_present_indices = all_present_indices[-args.eval_size:]
        corpus.eval_present_sample = full_sample[corpus.eval_present_indices]
        corpus.eval_present_location = object_location_in_image[
                corpus.eval_present_indices]

    if cache_filename:
        numpy.savez(cache_filename, **corpus)

def compute_mean_present_features(args, corpus, cache_filename, model):
    # Phase 1.5.  Figure mean activations for every channel where there
    # is a doorway.
    if all(k in corpus for k in ['mean_present_feature']):
        return
    progress = default_progress()
    with torch.no_grad():
        total_present_feature = 0
        for [zbatch, featloc] in progress(
                torch.utils.data.DataLoader(TensorDataset(
                    corpus.object_present_sample,
                    corpus.object_present_location),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Mean activations"):
            zbatch = zbatch.cuda()
            featloc = featloc.cuda()
            tensor_image = model(zbatch)
            feat = model.retained_layer(args.layer)
            flatfeat = feat.view(feat.shape[0], feat.shape[1], -1)
            sum_feature_at_obj = flatfeat[
                    torch.arange(feat.shape[0]).to(feat.device), :, featloc
                    ].sum(0)
            total_present_feature = total_present_feature + sum_feature_at_obj
        corpus.mean_present_feature = (total_present_feature / len(
                corpus.object_present_sample)).cpu()
    if cache_filename:
        numpy.savez(cache_filename, **corpus)

def compute_feature_quantiles(args, corpus, cache_filename, model, full_sample):
    # Phase 1.6.  Figure the 99% and 99.9%ile of every feature.
    if all(k in corpus for k in ['feature_99', 'feature_999']):
        return
    progress = default_progress()
    with torch.no_grad():
        rq = RunningQuantile(resolution=10000) # 10x what's needed.
        for [zbatch] in progress(
                torch.utils.data.DataLoader(TensorDataset(full_sample),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Calculating 0.999 quantile"):
            zbatch = zbatch.cuda()
            tensor_image = model(zbatch)
            feat = model.retained_layer(args.layer)
            rq.add(feat.permute(0, 2, 3, 1
                ).contiguous().view(-1, feat.shape[1]))
        result = rq.quantiles([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])
        corpus.feature_001 = result[:, 0].cpu()
        corpus.feature_01 = result[:, 1].cpu()
        corpus.feature_10 = result[:, 2].cpu()
        corpus.feature_50 = result[:, 3].cpu()
        corpus.feature_90 = result[:, 4].cpu()
        corpus.feature_99 = result[:, 5].cpu()
        corpus.feature_999 = result[:, 6].cpu()
    numpy.savez(cache_filename, **corpus)

def compute_candidate_locations(args, corpus, cache_filename, model,
        segmenter, classnum, second_sample):
    # Phase 2.  Identify a set of candidate locations for doorways.
    # Place the median doorway activation in every location of an image
    # and identify where it can go that doorway pixels increase.
    if all(k in corpus for k in ['candidate_indices',
            'candidate_sample', 'candidate_score',
            'candidate_location', 'object_score_at_candidate',
            'candidate_location_popularity']):
        return
    progress = default_progress()
    feature_shape = model.feature_shape[args.layer][2:]
    num_locations = numpy.prod(feature_shape).item()
    with torch.no_grad():
        # Simplify - just treat all locations as possible
        possible_locations = numpy.arange(num_locations)

        # Speed up search for locations, by weighting probed locations
        # according to observed distribution.
        location_weights = (corpus.object_location_popularity).double()
        location_weights += (location_weights.mean()) / 10.0
        location_weights = location_weights / location_weights.sum()

        candidate_scores = []
        object_scores = []
        prng = numpy.random.RandomState(1)
        for [zbatch] in progress(
                torch.utils.data.DataLoader(TensorDataset(second_sample),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Candidate pool"):
            batch_scores = torch.zeros((len(zbatch),) + feature_shape).cuda()
            flat_batch_scores = batch_scores.view(len(zbatch), -1)
            zbatch = zbatch.cuda()
            tensor_image = model(zbatch)
            segmented_image = segmenter.segment_batch(tensor_image,
                    downsample=2)
            mask = (segmented_image == classnum).max(1)[0]
            object_score = torch.nn.functional.adaptive_avg_pool2d(
                    mask.float(), feature_shape)
            baseline_presence = mask.float().view(mask.shape[0], -1).sum(1)

            edit_mask = torch.zeros((1, 1) + feature_shape).cuda()
            if '_tcm' in args.variant:
                # variant: top-conditional-mean
                replace_vec = (corpus.mean_present_feature
                        [None,:,None,None].cuda())
            else: # default: weighted mean
                replace_vec = (corpus.weighted_mean_present_feature
                        [None,:,None,None].cuda())
            # Sample 10 random locations to examine.
            for loc in prng.choice(possible_locations, replace=False,
                    p=location_weights, size=5):
                edit_mask.zero_()
                edit_mask.view(-1)[loc] = 1
                model.edit_layer(args.layer,
                        ablation=edit_mask, replacement=replace_vec)
                tensor_image = model(zbatch)
                segmented_image = segmenter.segment_batch(tensor_image,
                    downsample=2)
                mask = (segmented_image == classnum).max(1)[0]
                modified_presence = mask.float().view(
                        mask.shape[0], -1).sum(1)
                flat_batch_scores[:,loc] = (
                        modified_presence - baseline_presence)
            candidate_scores.append(batch_scores.cpu())
            object_scores.append(object_score.cpu())

        object_scores = torch.cat(object_scores)
        candidate_scores = torch.cat(candidate_scores)
        # Eliminate candidates where the object is present.
        candidate_scores = candidate_scores * (object_scores == 0).float()
        candidate_score_at_image, candidate_location_in_image = (
                candidate_scores.view(args.search_size, -1).max(1))
        best_candidate_scores, best_candidate_images = torch.sort(
                -candidate_score_at_image)
        all_candidate_indices = torch.sort(
                best_candidate_images[:(args.train_size+args.eval_size)])[0]
        corpus.candidate_indices = all_candidate_indices[:args.train_size]
        corpus.candidate_sample = second_sample[corpus.candidate_indices]
        corpus.candidate_location = candidate_location_in_image[
                corpus.candidate_indices]
        corpus.candidate_score = candidate_score_at_image[
                corpus.candidate_indices]
        corpus.object_score_at_candidate = object_scores.view(
                len(object_scores), -1)[
                corpus.candidate_indices, corpus.candidate_location]
        corpus.candidate_location_popularity = torch.bincount(
            corpus.candidate_location,
            minlength=num_locations)
        corpus.eval_candidate_indices = all_candidate_indices[
                -args.eval_size:]
        corpus.eval_candidate_sample = second_sample[
                corpus.eval_candidate_indices]
        corpus.eval_candidate_location = candidate_location_in_image[
                corpus.eval_candidate_indices]
    numpy.savez(cache_filename, **corpus)

def visualize_training_locations(args, corpus, cachedir, model):
    # Phase 2.5 Create visualizations of the corpus images.
    progress = default_progress()
    feature_shape = model.feature_shape[args.layer][2:]
    num_locations = numpy.prod(feature_shape).item()
    with torch.no_grad():
        imagedir = os.path.join(cachedir, 'image')
        os.makedirs(imagedir, exist_ok=True)
        image_saver = WorkerPool(SaveImageWorker)
        for group, group_sample, group_location, group_indices in [
                ('present',
                    corpus.object_present_sample,
                    corpus.object_present_location,
                    corpus.present_indices),
                ('candidate',
                    corpus.candidate_sample,
                    corpus.candidate_location,
                    corpus.candidate_indices)]:
            for [zbatch, featloc, indices] in progress(
                    torch.utils.data.DataLoader(TensorDataset(
                        group_sample, group_location, group_indices),
                        batch_size=args.inference_batch_size, num_workers=10,
                        pin_memory=True),
                    desc="Visualize %s" % group):
                zbatch = zbatch.cuda()
                tensor_image = model(zbatch)
                feature_mask = torch.zeros((len(zbatch), 1) + feature_shape)
                feature_mask.view(len(zbatch), -1).scatter_(
                        1, featloc[:,None], 1)
                feature_mask = torch.nn.functional.adaptive_max_pool2d(
                        feature_mask.float(), tensor_image.shape[-2:]).cuda()
                yellow = torch.Tensor([1.0, 1.0, -1.0]
                        )[None, :, None, None].cuda()
                tensor_image = tensor_image * (1 - 0.5 * feature_mask) + (
                        0.5 * feature_mask * yellow)
                byte_image = (((tensor_image+1)/2)*255).clamp(0, 255).byte()
                numpy_image = byte_image.permute(0, 2, 3, 1).cpu().numpy()
                for i, index in enumerate(indices):
                    image_saver.add(numpy_image[i], os.path.join(imagedir,
                        '%s_%d.jpg' % (group, index)))
    image_saver.join()

def scale_summary(scale, lownums, highnums):
    value, order = (-(scale.detach())).cpu().sort(0)
    lowsum = ' '.join('%d: %.3g' % (o.item(), -v.item())
            for v, o in zip(value[:lownums], order[:lownums]))
    highsum = ' '.join('%d: %.3g' % (o.item(), -v.item())
            for v, o in zip(value[-highnums:], order[-highnums:]))
    return lowsum + ' ... ' + highsum

# Phase 3.  Given those two sets, now optimize a such that:
#   Door pred lost if we take 0 * a at a candidate (1)
#   Door pred gained If we take 99.9th activation * a at a candiate (1)
#

# ADE_au = E | on - E | off)
#       = cand-frac E_cand | on + nocand-frac E_cand | on
#        -  door-frac E_door | off + nodoor-frac E_nodoor | off
#       approx = cand-frac E_cand | on - door-frac E_door | off + K
# Each batch has both types, and minimizes
#     door-frac sum(s_c) when pixel off - cand-frac sum(s_c) when pixel on

def initial_ablation(args, dissectdir):
    # Load initialization from dissection, based on iou scores.
    with open(os.path.join(dissectdir, 'dissect.json')) as f:
        dissection = EasyDict(json.load(f))
    lrec = [l for l in dissection.layers if l.layer == args.layer][0]
    rrec = [r for r in lrec.rankings if r.name == '%s-iou' % args.classname
            ][0]
    init_scores = -torch.tensor(rrec.score)
    return init_scores / init_scores.max()

def ace_loss(segmenter, classnum, model, layer, high_replacement, ablation,
        pbatch, ploc, cbatch, cloc, run_backward=False,
        discrete_pixels=False,
        discrete_units=False,
        mixed_units=False,
        ablation_only=False,
        fullimage_measurement=False,
        fullimage_ablation=False,
        ):
    feature_shape = model.feature_shape[layer][2:]
    if discrete_units: # discretize ablation to the top N units
        assert discrete_units > 0
        d = torch.zeros_like(ablation)
        top_units = torch.topk(ablation.view(-1), discrete_units)[1]
        if mixed_units:
            d.view(-1)[top_units] = ablation.view(-1)[top_units]
        else:
            d.view(-1)[top_units] = 1
        ablation = d
    # First, ablate a sample of locations with positive presence
    # and see how much the presence is reduced.
    p_mask = torch.zeros((len(pbatch), 1) + feature_shape)
    if fullimage_ablation:
        p_mask[...] = 1
    else:
        p_mask.view(len(pbatch), -1).scatter_(1, ploc[:,None], 1)
    p_mask = p_mask.cuda()
    a_p_mask = (ablation * p_mask)
    model.edit_layer(layer, ablation=a_p_mask, replacement=None)
    tensor_images = model(pbatch.cuda())
    assert model._ablation[layer] is a_p_mask
    erase_effect, erased_mask = segmenter.predict_single_class(
            tensor_images, classnum, downsample=2)
    if discrete_pixels: # pixel loss: use mask instead of pred
        erase_effect = erased_mask.float()
    erase_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            erase_effect[:,None,:,:], feature_shape)[:,0,:,:]
    if fullimage_measurement:
        erase_loss = erase_downsampled.sum()
    else:
        erase_at_loc = erase_downsampled.view(len(erase_downsampled), -1
                )[torch.arange(len(erase_downsampled)), ploc]
        erase_loss = erase_at_loc.sum()
    if run_backward:
        erase_loss.backward()
    if ablation_only:
        return erase_loss
    # Second, activate a sample of locations that are candidates for
    # insertion and see how much the presence is increased.
    c_mask = torch.zeros((len(cbatch), 1) + feature_shape)
    c_mask.view(len(cbatch), -1).scatter_(1, cloc[:,None], 1)
    c_mask = c_mask.cuda()
    a_c_mask = (ablation * c_mask)
    model.edit_layer(layer, ablation=a_c_mask, replacement=high_replacement)
    tensor_images = model(cbatch.cuda())
    assert model._ablation[layer] is a_c_mask
    add_effect, added_mask = segmenter.predict_single_class(
            tensor_images, classnum, downsample=2)
    if discrete_pixels: # pixel loss: use mask instead of pred
        add_effect = added_mask.float()
    add_effect = -add_effect
    add_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            add_effect[:,None,:,:], feature_shape)[:,0,:,:]
    if fullimage_measurement:
        add_loss = add_downsampled.mean()
    else:
        add_at_loc = add_downsampled.view(len(add_downsampled), -1
                )[torch.arange(len(add_downsampled)), ploc]
        add_loss = add_at_loc.sum()
    if run_backward:
        add_loss.backward()
    return erase_loss + add_loss

def train_ablation(args, corpus, cachefile, model, segmenter, classnum,
        initial_ablation=None):
    progress = default_progress()
    cachedir = os.path.dirname(cachefile)
    snapdir = os.path.join(cachedir, 'snapshots')
    os.makedirs(snapdir, exist_ok=True)

    # high_replacement = corpus.feature_99[None,:,None,None].cuda()
    if '_h99' in args.variant:
        high_replacement = corpus.feature_99[None,:,None,None].cuda()
    elif '_tcm' in args.variant:
        # variant: top-conditional-mean
        high_replacement = (
                corpus.mean_present_feature[None,:,None,None].cuda())
    else: # default: weighted mean
        high_replacement = (
                corpus.weighted_mean_present_feature[None,:,None,None].cuda())
    fullimage_measurement = False
    ablation_only = False
    fullimage_ablation = False
    if '_fim' in args.variant:
        fullimage_measurement = True
    elif '_fia' in args.variant:
        fullimage_measurement = True
        ablation_only = True
        fullimage_ablation = True
    high_replacement.requires_grad = False
    for p in model.parameters():
        p.requires_grad = False

    ablation = torch.zeros(high_replacement.shape).cuda()
    if initial_ablation is not None:
        ablation.view(-1)[...] = initial_ablation
    ablation.requires_grad = True
    optimizer = torch.optim.Adam([ablation], lr=0.01)
    start_epoch = 0
    epoch = 0

    def eval_loss_and_reg():
        discrete_experiments = dict(
           # dpixel=dict(discrete_pixels=True),
           # dunits20=dict(discrete_units=20),
           # dumix20=dict(discrete_units=20, mixed_units=True),
           # dunits10=dict(discrete_units=10),
           # abonly=dict(ablation_only=True),
           # fimabl=dict(ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           dboth20=dict(discrete_units=20, discrete_pixels=True),
           # dbothm20=dict(discrete_units=20, mixed_units=True,
           #              discrete_pixels=True),
           # abdisc20=dict(discrete_units=20, discrete_pixels=True,
           #             ablation_only=True),
           # abdiscm20=dict(discrete_units=20, mixed_units=True,
           #             discrete_pixels=True,
           #             ablation_only=True),
           # fimadp=dict(discrete_pixels=True,
           #             ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           # fimadu10=dict(discrete_units=10,
           #             ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           # fimadb10=dict(discrete_units=10, discrete_pixels=True,
           #             ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           fimadbm10=dict(discrete_units=10, mixed_units=True,
                       discrete_pixels=True,
                       ablation_only=True,
                       fullimage_ablation=True,
                       fullimage_measurement=True),
           # fimadu20=dict(discrete_units=20,
           #             ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           # fimadb20=dict(discrete_units=20, discrete_pixels=True,
           #             ablation_only=True,
           #             fullimage_ablation=True,
           #             fullimage_measurement=True),
           fimadbm20=dict(discrete_units=20, mixed_units=True,
                       discrete_pixels=True,
                       ablation_only=True,
                       fullimage_ablation=True,
                       fullimage_measurement=True)
           )
        with torch.no_grad():
            total_loss = 0
            discrete_losses = {k: 0 for k in discrete_experiments}
            for [pbatch, ploc, cbatch, cloc] in progress(
                    torch.utils.data.DataLoader(TensorDataset(
                        corpus.eval_present_sample,
                        corpus.eval_present_location,
                        corpus.eval_candidate_sample,
                        corpus.eval_candidate_location),
                    batch_size=args.inference_batch_size, num_workers=10,
                    shuffle=False, pin_memory=True),
                    desc="Eval"):
                # First, put in zeros for the selected units.
                # Loss is amount of remaining object.
                total_loss = total_loss + ace_loss(segmenter, classnum,
                        model, args.layer, high_replacement, ablation,
                        pbatch, ploc, cbatch, cloc, run_backward=False,
                        ablation_only=ablation_only,
                        fullimage_measurement=fullimage_measurement)
                for k, config in discrete_experiments.items():
                    discrete_losses[k] = discrete_losses[k] + ace_loss(
                        segmenter, classnum,
                        model, args.layer, high_replacement, ablation,
                        pbatch, ploc, cbatch, cloc, run_backward=False,
                        **config)
            avg_loss = (total_loss / args.eval_size).item()
            avg_d_losses = {k: (d / args.eval_size).item()
                    for k, d in discrete_losses.items()}
            regularizer = (args.l2_lambda * ablation.pow(2).sum())
            print_progress('Epoch %d Loss %g Regularizer %g' %
                    (epoch, avg_loss, regularizer))
            print_progress(' '.join('%s: %g' % (k, d)
                    for k, d in avg_d_losses.items()))
            print_progress(scale_summary(ablation.view(-1), 10, 3))
            return avg_loss, regularizer, avg_d_losses

    if args.eval_only:
        # For eval_only, just load each snapshot and re-run validation eval
        # pass on each one.
        for epoch in range(-1, args.train_epochs):
            snapfile = os.path.join(snapdir, 'epoch-%d.pth' % epoch)
            if not os.path.exists(snapfile):
                data = {}
                if epoch >= 0:
                    print('No epoch %d' % epoch)
                    continue
            else:
                data = torch.load(snapfile)
                with torch.no_grad():
                    ablation[...] = data['ablation'].to(ablation.device)
                    optimizer.load_state_dict(data['optimizer'])
            avg_loss, regularizer, new_extra = eval_loss_and_reg()
            # Keep old values, and update any new ones.
            extra = {k: v for k, v in data.items()
                    if k not in ['ablation', 'optimizer', 'avg_loss']}
            extra.update(new_extra)
            torch.save(dict(ablation=ablation, optimizer=optimizer.state_dict(),
                avg_loss=avg_loss, **extra),
                os.path.join(snapdir, 'epoch-%d.pth' % epoch))
        # Return loaded ablation.
        return ablation.view(-1).detach().cpu().numpy()

    if not args.no_cache:
        for start_epoch in reversed(range(args.train_epochs)):
            snapfile = os.path.join(snapdir, 'epoch-%d.pth' % start_epoch)
            if os.path.exists(snapfile):
                data = torch.load(snapfile)
                with torch.no_grad():
                    ablation[...] = data['ablation'].to(ablation.device)
                    optimizer.load_state_dict(data['optimizer'])
                start_epoch += 1
                break

    if start_epoch < args.train_epochs:
        epoch = start_epoch - 1
        avg_loss, regularizer, extra = eval_loss_and_reg()
        if epoch == -1:
            torch.save(dict(ablation=ablation, optimizer=optimizer.state_dict(),
                avg_loss=avg_loss, **extra),
                os.path.join(snapdir, 'epoch-%d.pth' % epoch))

    update_size = args.train_update_freq * args.train_batch_size
    for epoch in range(start_epoch, args.train_epochs):
        candidate_shuffle = torch.randperm(len(corpus.candidate_sample))
        train_loss = 0
        for batch_num, [pbatch, ploc, cbatch, cloc] in enumerate(progress(
                torch.utils.data.DataLoader(TensorDataset(
                    corpus.object_present_sample,
                    corpus.object_present_location,
                    corpus.candidate_sample[candidate_shuffle],
                    corpus.candidate_location[candidate_shuffle]),
                batch_size=args.train_batch_size, num_workers=10,
                shuffle=True, pin_memory=True),
                desc="ACE opt epoch %d" % epoch)):
            if batch_num % args.train_update_freq == 0:
                optimizer.zero_grad()
            # First, put in zeros for the selected units.  Loss is amount
            # of remaining object.
            loss = ace_loss(segmenter, classnum,
                    model, args.layer, high_replacement, ablation,
                    pbatch, ploc, cbatch, cloc, run_backward=True,
                    ablation_only=ablation_only,
                    fullimage_measurement=fullimage_measurement)
            with torch.no_grad():
                train_loss = train_loss + loss
            if (batch_num + 1) % args.train_update_freq == 0:
                # Third, add some L2 loss to encourage sparsity.
                regularizer = (args.l2_lambda * update_size
                        * ablation.pow(2).sum())
                regularizer.backward()
                optimizer.step()
                with torch.no_grad():
                    ablation.clamp_(0, 1)
                    post_progress(l=(train_loss/update_size).item(),
                            r=(regularizer/update_size).item())
                    train_loss = 0

        avg_loss, regularizer, extra = eval_loss_and_reg()
        torch.save(dict(ablation=ablation, optimizer=optimizer.state_dict(),
            avg_loss=avg_loss, **extra),
            os.path.join(snapdir, 'epoch-%d.pth' % epoch))
        numpy.save(os.path.join(snapdir, 'epoch-%d.npy' % epoch),
                ablation.detach().cpu().numpy())

    # The output of this phase is this set of scores.
    return ablation.view(-1).detach().cpu().numpy()


def tensor_to_numpy_image_batch(tensor_image):
    byte_image = (((tensor_image+1)/2)*255).clamp(0, 255).byte()
    numpy_image = byte_image.permute(0, 2, 3, 1).cpu().numpy()
    return numpy_image

# Phase 4: evaluation of intervention

def evaluate_ablation(args, model, segmenter, eval_sample, classnum, layer,
        ordering):
    total_bincount = 0
    data_size = 0
    progress = default_progress()
    for l in model.ablation:
        model.ablation[l] = None
    feature_units = model.feature_shape[args.layer][1]
    feature_shape = model.feature_shape[args.layer][2:]
    repeats = len(ordering)
    total_scores = torch.zeros(repeats + 1)
    for i, batch in enumerate(progress(torch.utils.data.DataLoader(
                TensorDataset(eval_sample),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Evaluate interventions")):
        tensor_image = model(zbatch)
        segmented_image = segmenter.segment_batch(tensor_image,
                    downsample=2)
        mask = (segmented_image == classnum).max(1)[0]
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
            model.edit_layer(args.layer, ablation=(
                    imask.float()[:,None,:,:] * ibc[:,:,None,None]))
            _, seg, _, _, _ = (
                recovery.recover_im_seg_bc_and_features(
                    [ibz], model))
            mask = (seg == classnum).max(1)[0]
            downsampled_iseg = torch.nn.functional.adaptive_avg_pool2d(
                    mask.float()[:,None,:,:], feature_shape)[:,0,:,:]
            scores[j:j+batch_size] = downsampled_iseg[
                    (torch.arange(len(ibz)),) + tuple(ibl)]
        scores = scores.view(repeats, location_count).sum(1)
        total_scores[1:] += scores
    return total_scores

def evaluate_interventions(args, model, segmenter, eval_sample,
        classnum, layer, units):
    total_bincount = 0
    data_size = 0
    progress = default_progress()
    for l in model.ablation:
        model.ablation[l] = None
    feature_units = model.feature_shape[args.layer][1]
    feature_shape = model.feature_shape[args.layer][2:]
    repeats = len(ordering)
    total_scores = torch.zeros(repeats + 1)
    for i, batch in enumerate(progress(torch.utils.data.DataLoader(
                TensorDataset(eval_sample),
                batch_size=args.inference_batch_size, num_workers=10,
                pin_memory=True),
                desc="Evaluate interventions")):
        tensor_image = model(zbatch)
        segmented_image = segmenter.segment_batch(tensor_image,
                    downsample=2)
        mask = (segmented_image == classnum).max(1)[0]
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
            model.ablation[args.layer] = (
                    imask.float()[:,None,:,:] * ibc[:,:,None,None])
            _, seg, _, _, _ = (
                recovery.recover_im_seg_bc_and_features(
                    [ibz], model))
            mask = (seg == classnum).max(1)[0]
            downsampled_iseg = torch.nn.functional.adaptive_avg_pool2d(
                    mask.float()[:,None,:,:], feature_shape)[:,0,:,:]
            scores[j:j+batch_size] = downsampled_iseg[
                    (torch.arange(len(ibz)),) + tuple(ibl)]
        scores = scores.view(repeats, location_count).sum(1)
        total_scores[1:] += scores
    return total_scores


def add_ace_ranking_to_dissection(outdir, layer, classname, total_scores):
    source_filename = os.path.join(outdir, 'dissect.json')
    source_filename_bak = os.path.join(outdir, 'dissect.json.bak')

    # Back up the dissection (if not already backed up) before modifying
    if not os.path.exists(source_filename_bak):
        shutil.copy(source_filename, source_filename_bak)

    with open(source_filename) as f:
        dissection = EasyDict(json.load(f))

    ranking_name = '%s-ace' % classname

    # Remove any old ace ranking with the same name
    lrec = [l for l in dissection.layers if l.layer == layer][0]
    lrec.rankings = [r for r in lrec.rankings if r.name != ranking_name]

    # Now convert ace scores to rankings
    new_rankings = [dict(
        name=ranking_name,
        score=(-total_scores).flatten().tolist(),
        metric='ace')]

    # Prepend to list.
    lrec.rankings[2:2] = new_rankings

    # Replace the old dissect.json in-place
    with open(source_filename, 'w') as f:
        json.dump(dissection, f, indent=1)

def summarize_scores(args, corpus, cachedir, layer, classname, variant, scores):
    target_filename = os.path.join(cachedir, 'summary.json')

    ranking_name = '%s-%s' % (classname, variant)
    # Now convert ace scores to rankings
    new_rankings = [dict(
        name=ranking_name,
        score=(-scores).flatten().tolist(),
        metric=variant)]
    result = dict(layers=[dict(layer=layer, rankings=new_rankings)])

    # Replace the old dissect.json in-place
    with open(target_filename, 'w') as f:
        json.dump(result, f, indent=1)

if __name__ == '__main__':
    main()
