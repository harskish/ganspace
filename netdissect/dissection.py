'''
To run dissection:

1. Load up the convolutional model you wish to dissect, and wrap it in
   an InstrumentedModel; then call imodel.retain_layers([layernames,..])
   to instrument the layers of interest.
2. Load the segmentation dataset using the BrodenDataset class;
   use the transform_image argument to normalize images to be
   suitable for the model, or the size argument to truncate the dataset.
3. Choose a directory in which to write the output, and call
   dissect(outdir, model, dataset).

Example:

    from dissect import InstrumentedModel, dissect
    from broden import BrodenDataset

    model = InstrumentedModel(load_my_model())
    model.eval()
    model.cuda()
    model.retain_layers(['conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
    bds = BrodenDataset('dataset/broden1_227',
            transform_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STDEV)]),
            size=1000)
    dissect('result/dissect', model, bds,
            examples_per_unit=10)
'''

import torch, numpy, os, re, json, shutil, types, tempfile, torchvision
# import warnings
# warnings.simplefilter('error', UserWarning)
from PIL import Image
from xml.etree import ElementTree as et
from collections import OrderedDict, defaultdict
from .progress import verbose_progress, default_progress, print_progress
from .progress import desc_progress
from .runningstats import RunningQuantile, RunningTopK
from .runningstats import RunningCrossCovariance, RunningConditionalQuantile
from .sampler import FixedSubsetSampler
from .actviz import activation_visualization
from .segviz import segment_visualization, high_contrast
from .workerpool import WorkerBase, WorkerPool
from .segmenter import UnifiedParsingSegmenter

def dissect(outdir, model, dataset,
        segrunner=None,
        train_dataset=None,
        model_segmenter=None,
        quantile_threshold=0.005,
        iou_threshold=0.05,
        iqr_threshold=0.01,
        examples_per_unit=100,
        batch_size=100,
        num_workers=24,
        seg_batch_size=5,
        make_images=True,
        make_labels=True,
        make_maxiou=False,
        make_covariance=False,
        make_report=True,
        make_row_images=True,
        make_single_images=False,
        rank_all_labels=False,
        netname=None,
        meta=None,
        merge=None,
        settings=None,
        ):
    '''
    Runs net dissection in-memory, using pytorch, and saves visualizations
    and metadata into outdir.
    '''
    assert not model.training, 'Run model.eval() before dissection'
    if netname is None:
        netname = type(model).__name__
    if segrunner is None:
        segrunner = ClassifierSegRunner(dataset)
    if train_dataset is None:
        train_dataset = dataset
    make_iqr = (quantile_threshold == 'iqr')
    with torch.no_grad():
        device = next(model.parameters()).device
        levels = None
        labelnames, catnames = None, None
        maxioudata, iqrdata = None, None
        labeldata = None
        iqrdata, cov = None, None

        labelnames, catnames = segrunner.get_label_and_category_names()
        label_category = [catnames.index(c) if c in catnames else 0
                for l, c in labelnames]

        # First, always collect qunatiles and topk information.
        segloader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, num_workers=num_workers,
                pin_memory=(device.type == 'cuda'))
        quantiles, topk = collect_quantiles_and_topk(outdir, model,
            segloader, segrunner, k=examples_per_unit)

        # Thresholds can be automatically chosen by maximizing iqr
        if make_iqr:
            # Get thresholds based on an IQR optimization
            segloader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=1, num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            iqrdata = collect_iqr(outdir, model, segloader, segrunner)
            max_iqr, full_iqr_levels = iqrdata[:2]
            max_iqr_agreement = iqrdata[4]
            # qualified_iqr[max_iqr_quantile[layer] > 0.5] = 0
            levels = {layer: full_iqr_levels[layer][
                    max_iqr[layer].max(0)[1],
                    torch.arange(max_iqr[layer].shape[1])].to(device)
                    for layer in full_iqr_levels}
        else:
            levels = {k: qc.quantiles([1.0 - quantile_threshold])[:,0]
                      for k, qc in quantiles.items()}

        quantiledata = (topk, quantiles, levels, quantile_threshold)

        if make_images:
            segloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            generate_images(outdir, model, dataset, topk, levels, segrunner,
                    row_length=examples_per_unit, batch_size=seg_batch_size,
                    row_images=make_row_images,
                    single_images=make_single_images,
                    num_workers=num_workers)

        if make_maxiou:
            assert train_dataset, "Need training dataset for maxiou."
            segloader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=1, num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            maxioudata = collect_maxiou(outdir, model, segloader,
                    segrunner)

        if make_labels:
            segloader = torch.utils.data.DataLoader(dataset,
                    batch_size=1, num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            iou_scores, iqr_scores, tcs, lcs, ccs, ics = (
                    collect_bincounts(outdir, model, segloader,
                    levels, segrunner))
            labeldata = (iou_scores, iqr_scores, lcs, ccs, ics, iou_threshold,
                    iqr_threshold)

        if make_covariance:
            segloader = torch.utils.data.DataLoader(dataset,
                    batch_size=seg_batch_size,
                    num_workers=num_workers,
                    pin_memory=(device.type == 'cuda'))
            cov = collect_covariance(outdir, model, segloader, segrunner)

        if make_report:
            generate_report(outdir,
                    quantiledata=quantiledata,
                    labelnames=labelnames,
                    catnames=catnames,
                    labeldata=labeldata,
                    maxioudata=maxioudata,
                    iqrdata=iqrdata,
                    covariancedata=cov,
                    rank_all_labels=rank_all_labels,
                    netname=netname,
                    meta=meta,
                    mergedata=merge,
                    settings=settings)

        return quantiledata, labeldata

def generate_report(outdir, quantiledata, labelnames=None, catnames=None,
        labeldata=None, maxioudata=None, iqrdata=None, covariancedata=None,
        rank_all_labels=False, netname='Model', meta=None, settings=None,
        mergedata=None):
    '''
    Creates dissection.json reports and summary bargraph.svg files in the
    specified output directory, and copies a dissection.html interface
    to go along with it.
    '''
    all_layers = []
    # Current source code directory, for html to copy.
    srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Unpack arguments
    topk, quantiles, levels, quantile_threshold = quantiledata
    top_record = dict(
            netname=netname,
            meta=meta,
            default_ranking='unit',
            quantile_threshold=quantile_threshold)
    if settings is not None:
        top_record['settings'] = settings
    if labeldata is not None:
        iou_scores, iqr_scores, lcs, ccs, ics, iou_threshold, iqr_threshold = (
                labeldata)
        catorder = {'object': -7, 'scene': -6, 'part': -5,
                    'piece': -4,
                    'material': -3, 'texture': -2, 'color': -1}
        for i, cat in enumerate(c for c in catnames if c not in catorder):
            catorder[cat] = i
        catnumber = {n: i for i, n in enumerate(catnames)}
        catnumber['-'] = 0
        top_record['default_ranking'] = 'label'
        top_record['iou_threshold'] = iou_threshold
        top_record['iqr_threshold'] = iqr_threshold
        labelnumber = dict((name[0], num)
                for num, name in enumerate(labelnames))
    # Make a segmentation color dictionary
    segcolors = {}
    for i, name in enumerate(labelnames):
        key = ','.join(str(s) for s in high_contrast[i % len(high_contrast)])
        if key in segcolors:
            segcolors[key] += '/' + name[0]
        else:
            segcolors[key] = name[0]
    top_record['segcolors'] = segcolors
    for layer in topk.keys():
        units, rankings = [], []
        record = dict(layer=layer, units=units, rankings=rankings)
        # For every unit, we always have basic visualization information.
        topa, topi = topk[layer].result()
        lev = levels[layer]
        for u in range(len(topa)):
            units.append(dict(
                unit=u,
                interp=True,
                level=lev[u].item(),
                top=[dict(imgnum=i.item(), maxact=a.item())
                    for i, a in zip(topi[u], topa[u])],
                ))
        rankings.append(dict(name="unit", score=list([
            u for u in range(len(topa))])))
        # TODO: consider including stats and ranking based on quantiles,
        # variance, connectedness here.

        # if we have labeldata, then every unit also gets a bunch of other info
        if labeldata is not None:
            lscore, qscore, cc, ic = [dat[layer]
                    for dat in [iou_scores, iqr_scores, ccs, ics]]
            if iqrdata is not None:
                # If we have IQR thresholds, assign labels based on that
                max_iqr, max_iqr_level = iqrdata[:2]
                best_label = max_iqr[layer].max(0)[1]
                best_score = lscore[best_label, torch.arange(lscore.shape[1])]
                best_qscore = qscore[best_label, torch.arange(lscore.shape[1])]
            else:
                # Otherwise, assign labels based on max iou
                best_score, best_label = lscore.max(0)
                best_qscore = qscore[best_label, torch.arange(qscore.shape[1])]
            record['iou_threshold'] = iou_threshold,
            for u, urec in enumerate(units):
                score, qscore, label = (
                        best_score[u], best_qscore[u], best_label[u])
                urec.update(dict(
                    iou=score.item(),
                    iou_iqr=qscore.item(),
                    lc=lcs[label].item(),
                    cc=cc[catnumber[labelnames[label][1]], u].item(),
                    ic=ic[label, u].item(),
                    interp=(qscore.item() > iqr_threshold and
                        score.item() > iou_threshold),
                    iou_labelnum=label.item(),
                    iou_label=labelnames[label.item()][0],
                    iou_cat=labelnames[label.item()][1],
                    ))
        if maxioudata is not None:
            max_iou, max_iou_level, max_iou_quantile = maxioudata
            qualified_iou = max_iou[layer].clone()
            # qualified_iou[max_iou_quantile[layer] > 0.75] = 0
            best_score, best_label = qualified_iou.max(0)
            for u, urec in enumerate(units):
                urec.update(dict(
                    maxiou=best_score[u].item(),
                    maxiou_label=labelnames[best_label[u].item()][0],
                    maxiou_cat=labelnames[best_label[u].item()][1],
                    maxiou_level=max_iou_level[layer][best_label[u], u].item(),
                    maxiou_quantile=max_iou_quantile[layer][
                        best_label[u], u].item()))
        if iqrdata is not None:
            [max_iqr, max_iqr_level, max_iqr_quantile,
                    max_iqr_iou, max_iqr_agreement] = iqrdata
            qualified_iqr = max_iqr[layer].clone()
            qualified_iqr[max_iqr_quantile[layer] > 0.5] = 0
            best_score, best_label = qualified_iqr.max(0)
            for u, urec in enumerate(units):
                urec.update(dict(
                    iqr=best_score[u].item(),
                    iqr_label=labelnames[best_label[u].item()][0],
                    iqr_cat=labelnames[best_label[u].item()][1],
                    iqr_level=max_iqr_level[layer][best_label[u], u].item(),
                    iqr_quantile=max_iqr_quantile[layer][
                        best_label[u], u].item(),
                    iqr_iou=max_iqr_iou[layer][best_label[u], u].item()
                    ))
        if covariancedata is not None:
            score = covariancedata[layer].correlation()
            best_score, best_label = score.max(1)
            for u, urec in enumerate(units):
                urec.update(dict(
                    cor=best_score[u].item(),
                    cor_label=labelnames[best_label[u].item()][0],
                    cor_cat=labelnames[best_label[u].item()][1]
                    ))
        if mergedata is not None:
            # Final step: if the user passed any data to merge into the
            # units, merge them now.  This can be used, for example, to
            # indiate that a unit is not interpretable based on some
            # outside analysis of unit statistics.
            for lrec in mergedata.get('layers', []):
                if lrec['layer'] == layer:
                    break
            else:
                lrec = None
            for u, urec in enumerate(lrec.get('units', []) if lrec else []):
                units[u].update(urec)
        # After populating per-unit info, populate per-layer ranking info
        if labeldata is not None:
            # Collect all labeled units
            labelunits = defaultdict(list)
            all_labelunits = defaultdict(list)
            for u, urec in enumerate(units):
                if urec['interp']:
                    labelunits[urec['iou_labelnum']].append(u)
                all_labelunits[urec['iou_labelnum']].append(u)
            # Sort all units in order with most popular label first.
            label_ordering = sorted(units,
                # Sort by:
                key=lambda r: (-1 if r['interp'] else 0,  # interpretable
                    -len(labelunits[r['iou_labelnum']]),  # label freq, score
                    -max([units[u]['iou']
                        for u in labelunits[r['iou_labelnum']]], default=0),
                    r['iou_labelnum'],                    # label
                    -r['iou']))                           # unit score
            # Add label and iou ranking.
            rankings.append(dict(name="label", score=(numpy.argsort(list(
                ur['unit'] for ur in label_ordering))).tolist()))
            rankings.append(dict(name="max iou", metric="iou", score=list(
                -ur['iou'] for ur in units)))
            # Add ranking for top labels
            # for labelnum in [n for n in sorted(
            #     all_labelunits.keys(), key=lambda x:
            #         -len(all_labelunits[x])) if len(all_labelunits[n])]:
            #     label = labelnames[labelnum][0]
            #     rankings.append(dict(name="%s-iou" % label,
            #         concept=label, metric='iou',
            #         score=(-lscore[labelnum, :]).tolist()))
            # Collate labels by category then frequency.
            record['labels'] = [dict(
                        label=labelnames[label][0],
                        labelnum=label,
                        units=labelunits[label],
                        cat=labelnames[label][1])
                    for label in (sorted(labelunits.keys(),
                        # Sort by:
                        key=lambda l: (catorder.get(          # category
                            labelnames[l][1], 0),
                            -len(labelunits[l]),              # label freq
                            -max([units[u]['iou'] for u in labelunits[l]],
                                default=0) # score
                            ))) if len(labelunits[label])]
            # Total number of interpretable units.
            record['interpretable'] = sum(len(group['units'])
                    for group in record['labels'])
            # Make a bargraph of labels
            os.makedirs(os.path.join(outdir, safe_dir_name(layer)),
                    exist_ok=True)
            catgroups = OrderedDict()
            for _, cat in sorted([(v, k) for k, v in catorder.items()]):
                catgroups[cat] = []
            for rec in record['labels']:
                if rec['cat'] not in catgroups:
                    catgroups[rec['cat']] = []
                catgroups[rec['cat']].append(rec['label'])
            make_svg_bargraph(
                    [rec['label'] for rec in record['labels']],
                    [len(rec['units']) for rec in record['labels']],
                    [(cat, len(group)) for cat, group in catgroups.items()],
                    filename=os.path.join(outdir, safe_dir_name(layer),
                        'bargraph.svg'))
            # Only show the bargraph if it is non-empty.
            if len(record['labels']):
                record['bargraph'] = 'bargraph.svg'
        if maxioudata is not None:
            rankings.append(dict(name="max maxiou", metric="maxiou", score=list(
                    -ur['maxiou'] for ur in units)))
        if iqrdata is not None:
            rankings.append(dict(name="max iqr", metric="iqr", score=list(
                    -ur['iqr'] for ur in units)))
        if covariancedata is not None:
            rankings.append(dict(name="max cor", metric="cor", score=list(
                    -ur['cor'] for ur in units)))

        all_layers.append(record)
    # Now add the same rankings to every layer...
    all_labels = None
    if rank_all_labels:
        all_labels = [name for name, cat in labelnames]
    if labeldata is not None:
        # Count layers+quadrants with a given label, and sort by freq
        counted_labels = defaultdict(int)
        for label in [
                re.sub(r'-(?:t|b|l|r|tl|tr|bl|br)$', '', unitrec['iou_label'])
                for record in all_layers for unitrec in record['units']]:
            counted_labels[label] += 1
        if all_labels is None:
            all_labels = [label for count, label in sorted((-v, k)
                for k, v in counted_labels.items())]
        for record in all_layers:
            layer = record['layer']
            for label in all_labels:
                labelnum = labelnumber[label]
                record['rankings'].append(dict(name="%s-iou" % label,
                    concept=label, metric='iou',
                    score=(-iou_scores[layer][labelnum, :]).tolist()))

    if maxioudata is not None:
        if all_labels is None:
            counted_labels = defaultdict(int)
            for label in [
                    re.sub(r'-(?:t|b|l|r|tl|tr|bl|br)$', '',
                        unitrec['maxiou_label'])
                    for record in all_layers for unitrec in record['units']]:
                counted_labels[label] += 1
            all_labels = [label for count, label in sorted((-v, k)
                for k, v in counted_labels.items())]
        qualified_iou = max_iou[layer].clone()
        qualified_iou[max_iou_quantile[layer] > 0.5] = 0
        for record in all_layers:
            layer = record['layer']
            for label in all_labels:
                labelnum = labelnumber[label]
                record['rankings'].append(dict(name="%s-maxiou" % label,
                    concept=label, metric='maxiou',
                    score=(-qualified_iou[labelnum, :]).tolist()))

    if iqrdata is not None:
        if all_labels is None:
            counted_labels = defaultdict(int)
            for label in [
                    re.sub(r'-(?:t|b|l|r|tl|tr|bl|br)$', '',
                        unitrec['iqr_label'])
                    for record in all_layers for unitrec in record['units']]:
                counted_labels[label] += 1
            all_labels = [label for count, label in sorted((-v, k)
                for k, v in counted_labels.items())]
        # qualified_iqr[max_iqr_quantile[layer] > 0.5] = 0
        for record in all_layers:
            layer = record['layer']
            qualified_iqr = max_iqr[layer].clone()
            for label in all_labels:
                labelnum = labelnumber[label]
                record['rankings'].append(dict(name="%s-iqr" % label,
                    concept=label, metric='iqr',
                    score=(-qualified_iqr[labelnum, :]).tolist()))

    if covariancedata is not None:
        if all_labels is None:
            counted_labels = defaultdict(int)
            for label in [
                    re.sub(r'-(?:t|b|l|r|tl|tr|bl|br)$', '',
                        unitrec['cor_label'])
                    for record in all_layers for unitrec in record['units']]:
                counted_labels[label] += 1
            all_labels = [label for count, label in sorted((-v, k)
                for k, v in counted_labels.items())]
        for record in all_layers:
            layer = record['layer']
            score = covariancedata[layer].correlation()
            for label in all_labels:
                labelnum = labelnumber[label]
                record['rankings'].append(dict(name="%s-cor" % label,
                    concept=label, metric='cor',
                    score=(-score[:, labelnum]).tolist()))

    for record in all_layers:
        layer = record['layer']
        # Dump per-layer json inside per-layer directory
        record['dirname'] = '.'
        with open(os.path.join(outdir, safe_dir_name(layer), 'dissect.json'),
                'w') as jsonfile:
            top_record['layers'] = [record]
            json.dump(top_record, jsonfile, indent=1)
        # Copy the per-layer html
        shutil.copy(os.path.join(srcdir, 'dissect.html'),
                os.path.join(outdir, safe_dir_name(layer), 'dissect.html'))
        record['dirname'] = safe_dir_name(layer)

    # Dump all-layer json in parent directory
    with open(os.path.join(outdir, 'dissect.json'), 'w') as jsonfile:
        top_record['layers'] = all_layers
        json.dump(top_record, jsonfile, indent=1)
    # Copy the all-layer html
    shutil.copy(os.path.join(srcdir, 'dissect.html'),
            os.path.join(outdir, 'dissect.html'))
    shutil.copy(os.path.join(srcdir, 'edit.html'),
            os.path.join(outdir, 'edit.html'))


def generate_images(outdir, model, dataset, topk, levels,
        segrunner, row_length=None, gap_pixels=5,
        row_images=True, single_images=False, prefix='',
        batch_size=100, num_workers=24):
    '''
    Creates an image strip file for every unit of every retained layer
    of the model, in the format [outdir]/[layername]/[unitnum]-top.jpg.
    Assumes that the indexes of topk refer to the indexes of dataset.
    Limits each strip to the top row_length images.
    '''
    progress = default_progress()
    needed_images = {}
    if row_images is False:
        row_length = 1
    # Pass 1: needed_images lists all images that are topk for some unit.
    for layer in topk:
        topresult = topk[layer].result()[1].cpu()
        for unit, row in enumerate(topresult):
            for rank, imgnum in enumerate(row[:row_length]):
                imgnum = imgnum.item()
                if imgnum not in needed_images:
                    needed_images[imgnum] = []
                needed_images[imgnum].append((layer, unit, rank))
    levels = {k: v.cpu().numpy() for k, v in levels.items()}
    row_length = len(row[:row_length])
    needed_sample = FixedSubsetSampler(sorted(needed_images.keys()))
    device = next(model.parameters()).device
    segloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            sampler=needed_sample)
    vizgrid, maskgrid, origrid, seggrid = [{} for _ in range(4)]
    # Pass 2: populate vizgrid with visualizations of top units.
    pool = None
    for i, batch in enumerate(
            progress(segloader, desc='Making images')):
        # Reverse transformation to get the image in byte form.
        seg, _, byte_im, _ = segrunner.run_and_segment_batch(batch, model,
                want_rgb=True)
        torch_features = model.retained_features()
        scale_offset = getattr(model, 'scale_offset', None)
        if pool is None:
            # Distribute the work across processes: create shared mmaps.
            for layer, tf in torch_features.items():
                [vizgrid[layer], maskgrid[layer], origrid[layer],
                        seggrid[layer]] = [
                    create_temp_mmap_grid((tf.shape[1],
                        byte_im.shape[1], row_length,
                        byte_im.shape[2] + gap_pixels, depth),
                        dtype='uint8',
                        fill=255)
                    for depth in [3, 4, 3, 3]]
            # Pass those mmaps to worker processes.
            pool = WorkerPool(worker=VisualizeImageWorker,
                    memmap_grid_info=[
                        {layer: (g.filename, g.shape, g.dtype)
                            for layer, g in grid.items()}
                        for grid in [vizgrid, maskgrid, origrid, seggrid]])
        byte_im = byte_im.cpu().numpy()
        numpy_seg = seg.cpu().numpy()
        features = {}
        for index in range(len(byte_im)):
            imgnum = needed_sample.samples[index + i*segloader.batch_size]
            for layer, unit, rank in needed_images[imgnum]:
                if layer not in features:
                    features[layer] = torch_features[layer].cpu().numpy()
                pool.add(layer, unit, rank,
                        byte_im[index],
                        features[layer][index, unit],
                        levels[layer][unit],
                        scale_offset[layer] if scale_offset else None,
                        numpy_seg[index])
    pool.join()
    # Pass 3: save image strips as [outdir]/[layer]/[unitnum]-[top/orig].jpg
    pool = WorkerPool(worker=SaveImageWorker)
    for layer, vg in progress(vizgrid.items(), desc='Saving images'):
        os.makedirs(os.path.join(outdir, safe_dir_name(layer),
            prefix + 'image'), exist_ok=True)
        if single_images:
           os.makedirs(os.path.join(outdir, safe_dir_name(layer),
               prefix + 's-image'), exist_ok=True)
        og, sg, mg = origrid[layer], seggrid[layer], maskgrid[layer]
        for unit in progress(range(len(vg)), desc='Units'):
            for suffix, grid in [('top.jpg', vg), ('orig.jpg', og),
                    ('seg.png', sg), ('mask.png', mg)]:
                strip = grid[unit].reshape(
                        (grid.shape[1], grid.shape[2] * grid.shape[3],
                            grid.shape[4]))
                if row_images:
                    filename = os.path.join(outdir, safe_dir_name(layer),
                            prefix + 'image', '%d-%s' % (unit, suffix))
                    pool.add(strip[:,:-gap_pixels,:].copy(), filename)
                    # Image.fromarray(strip[:,:-gap_pixels,:]).save(filename,
                    #        optimize=True, quality=80)
                if single_images:
                    single_filename = os.path.join(outdir, safe_dir_name(layer),
                        prefix + 's-image', '%d-%s' % (unit, suffix))
                    pool.add(strip[:,:strip.shape[1] // row_length
                        - gap_pixels,:].copy(), single_filename)
                    # Image.fromarray(strip[:,:strip.shape[1] // row_length
                    #     - gap_pixels,:]).save(single_filename,
                    #             optimize=True, quality=80)
    pool.join()
    # Delete the shared memory map files
    clear_global_shared_files([g.filename
        for grid in [vizgrid, maskgrid, origrid, seggrid]
        for g in grid.values()])

global_shared_files = {}
def create_temp_mmap_grid(shape, dtype, fill):
    dtype = numpy.dtype(dtype)
    filename = os.path.join(tempfile.mkdtemp(), 'temp-%s-%s.mmap' %
            ('x'.join('%d' % s for s in shape), dtype.name))
    fid = open(filename, mode='w+b')
    original = numpy.memmap(fid, dtype=dtype, mode='w+', shape=shape)
    original.fid = fid
    original[...] = fill
    global_shared_files[filename] = original
    return original

def shared_temp_mmap_grid(filename, shape, dtype):
    if filename not in global_shared_files:
        global_shared_files[filename] = numpy.memmap(
                filename, dtype=dtype, mode='r+', shape=shape)
    return global_shared_files[filename]

def clear_global_shared_files(filenames):
    for fn in filenames:
        if fn in global_shared_files:
            del global_shared_files[fn]
        try:
            os.unlink(fn)
        except OSError:
            pass

class VisualizeImageWorker(WorkerBase):
    def setup(self, memmap_grid_info):
        self.vizgrid, self.maskgrid, self.origrid, self.seggrid = [
                {layer: shared_temp_mmap_grid(*info)
                    for layer, info in grid.items()}
                for grid in memmap_grid_info]
    def work(self, layer, unit, rank,
            byte_im, acts, level, scale_offset, seg):
        self.origrid[layer][unit,:,rank,:byte_im.shape[0],:] = byte_im
        [self.vizgrid[layer][unit,:,rank,:byte_im.shape[0],:],
         self.maskgrid[layer][unit,:,rank,:byte_im.shape[0],:]] = (
                    activation_visualization(
                        byte_im,
                        acts,
                        level,
                        scale_offset=scale_offset,
                        return_mask=True))
        self.seggrid[layer][unit,:,rank,:byte_im.shape[0],:] = (
                    segment_visualization(seg, byte_im.shape[0:2]))

class SaveImageWorker(WorkerBase):
    def work(self, data, filename):
        Image.fromarray(data).save(filename, optimize=True, quality=80)

def score_tally_stats(label_category, tc, truth, cc, ic):
    pred = cc[label_category]
    total = tc[label_category][:, None]
    truth = truth[:, None]
    epsilon = 1e-20 # avoid division-by-zero
    union = pred + truth - ic
    iou = ic.double() / (union.double() + epsilon)
    arr = torch.empty(size=(2, 2) + ic.shape, dtype=ic.dtype, device=ic.device)
    arr[0, 0] = ic
    arr[0, 1] = pred - ic
    arr[1, 0] = truth - ic
    arr[1, 1] = total - union
    arr = arr.double() / total.double()
    mi = mutual_information(arr)
    je = joint_entropy(arr)
    iqr = mi / je
    iqr[torch.isnan(iqr)] = 0 # Zero out any 0/0
    return iou, iqr

def collect_quantiles_and_topk(outdir, model, segloader,
        segrunner, k=100, resolution=1024):
    '''
    Collects (estimated) quantile information and (exact) sorted top-K lists
    for every channel in the retained layers of the model.  Returns
    a map of quantiles (one RunningQuantile for each layer) along with
    a map of topk (one RunningTopK for each layer).
    '''
    device = next(model.parameters()).device
    features = model.retained_features()
    cached_quantiles = {
            layer: load_quantile_if_present(os.path.join(outdir,
                safe_dir_name(layer)), 'quantiles.npz',
                device=torch.device('cpu'))
            for layer in features }
    cached_topks = {
            layer: load_topk_if_present(os.path.join(outdir,
                safe_dir_name(layer)), 'topk.npz',
                device=torch.device('cpu'))
            for layer in features }
    if (all(value is not None for value in cached_quantiles.values()) and
        all(value is not None for value in cached_topks.values())):
        return cached_quantiles, cached_topks

    layer_batch_size = 8
    all_layers = list(features.keys())
    layer_batches = [all_layers[i:i+layer_batch_size]
            for i in range(0, len(all_layers), layer_batch_size)]

    quantiles, topks = {}, {}
    progress = default_progress()
    for layer_batch in layer_batches:
        for i, batch in enumerate(progress(segloader, desc='Quantiles')):
            # We don't actually care about the model output.
            model(batch[0].to(device))
            features = model.retained_features()
            # We care about the retained values
            for key in layer_batch:
                value = features[key]
                if topks.get(key, None) is None:
                    topks[key] = RunningTopK(k)
                if quantiles.get(key, None) is None:
                    quantiles[key] = RunningQuantile(resolution=resolution)
                topvalue = value
                if len(value.shape) > 2:
                    topvalue, _ = value.view(*(value.shape[:2] + (-1,))).max(2)
                    # Put the channel index last.
                    value = value.permute(
                            (0,) + tuple(range(2, len(value.shape))) + (1,)
                            ).contiguous().view(-1, value.shape[1])
                quantiles[key].add(value)
                topks[key].add(topvalue)
        # Save GPU memory
        for key in layer_batch:
            quantiles[key].to_(torch.device('cpu'))
            topks[key].to_(torch.device('cpu'))
    for layer in quantiles:
        save_state_dict(quantiles[layer],
                os.path.join(outdir, safe_dir_name(layer), 'quantiles.npz'))
        save_state_dict(topks[layer],
                os.path.join(outdir, safe_dir_name(layer), 'topk.npz'))
    return quantiles, topks

def collect_bincounts(outdir, model, segloader, levels, segrunner):
    '''
    Returns label_counts, category_activation_counts, and intersection_counts,
    across the data set, counting the pixels of intersection between upsampled,
    thresholded model featuremaps, with segmentation classes in the segloader.

    label_counts (independent of model): pixels across the data set that
        are labeled with the given label.
    category_activation_counts (one per layer): for each feature channel,
        pixels across the dataset where the channel exceeds the level
        threshold.  There is one count per category: activations only
        contribute to the categories for which any category labels are
        present on the images.
    intersection_counts (one per layer): for each feature channel and
        label, pixels across the dataset where the channel exceeds
        the level, and the labeled segmentation class is also present.

    This is a performance-sensitive function.  Best performance is
    achieved with a counting scheme which assumes a segloader with
    batch_size 1.
    '''
    # Load cached data if present
    (iou_scores, iqr_scores,
            total_counts, label_counts, category_activation_counts,
            intersection_counts) = {}, {}, None, None, {}, {}
    found_all = True
    for layer in model.retained_features():
        filename = os.path.join(outdir, safe_dir_name(layer), 'bincounts.npz')
        if os.path.isfile(filename):
            data = numpy.load(filename)
            iou_scores[layer] = torch.from_numpy(data['iou_scores'])
            iqr_scores[layer] = torch.from_numpy(data['iqr_scores'])
            total_counts = torch.from_numpy(data['total_counts'])
            label_counts = torch.from_numpy(data['label_counts'])
            category_activation_counts[layer] = torch.from_numpy(
                    data['category_activation_counts'])
            intersection_counts[layer] = torch.from_numpy(
                    data['intersection_counts'])
        else:
            found_all = False
    if found_all:
        return (iou_scores, iqr_scores,
            total_counts, label_counts, category_activation_counts,
            intersection_counts)

    device = next(model.parameters()).device
    labelcat, categories = segrunner.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0
                for l, c in labelcat]
    num_labels, num_categories = (len(n) for n in [labelcat, categories])

    # One-hot vector of category for each label
    labelcat = torch.zeros(num_labels, num_categories,
            dtype=torch.long, device=device)
    labelcat.scatter_(1, torch.from_numpy(numpy.array(label_category,
        dtype='int64')).to(device)[:,None], 1)
    # Running bincounts
    # activation_counts = {}
    assert segloader.batch_size == 1 # category_activation_counts needs this.
    category_activation_counts = {}
    intersection_counts = {}
    label_counts = torch.zeros(num_labels, dtype=torch.long, device=device)
    total_counts = torch.zeros(num_categories, dtype=torch.long, device=device)
    progress = default_progress()
    scale_offset_map = getattr(model, 'scale_offset', None)
    upsample_grids = {}
    # total_batch_categories = torch.zeros(
    #         labelcat.shape[1], dtype=torch.long, device=device)
    for i, batch in enumerate(progress(segloader, desc='Bincounts')):
        seg, batch_label_counts, _, imshape = segrunner.run_and_segment_batch(
                batch, model, want_bincount=True, want_rgb=True)
        bc = batch_label_counts.cpu()
        batch_label_counts = batch_label_counts.to(device)
        seg = seg.to(device)
        features = model.retained_features()
        # Accumulate bincounts and identify nonzeros
        label_counts += batch_label_counts[0]
        batch_labels = bc[0].nonzero()[:,0]
        batch_categories = labelcat[batch_labels].max(0)[0]
        total_counts += batch_categories * (
                seg.shape[0] * seg.shape[2] * seg.shape[3])
        for key, value in features.items():
            if key not in upsample_grids:
                upsample_grids[key] = upsample_grid(value.shape[2:],
                        seg.shape[2:], imshape,
                        scale_offset=scale_offset_map.get(key, None)
                            if scale_offset_map is not None else None,
                        dtype=value.dtype, device=value.device)
            upsampled = torch.nn.functional.grid_sample(value,
                    upsample_grids[key], padding_mode='border')
            amask = (upsampled > levels[key][None,:,None,None].to(
                upsampled.device))
            ac = amask.int().view(amask.shape[1], -1).sum(1)
            # if key not in activation_counts:
            #     activation_counts[key] = ac
            # else:
            #     activation_counts[key] += ac
            # The fastest approach: sum over each label separately!
            for label in batch_labels.tolist():
                if label == 0:
                    continue  # ignore the background label
                imask = amask * ((seg == label).max(dim=1, keepdim=True)[0])
                ic = imask.int().view(imask.shape[1], -1).sum(1)
                if key not in intersection_counts:
                    intersection_counts[key] = torch.zeros(num_labels,
                            amask.shape[1], dtype=torch.long, device=device)
                intersection_counts[key][label] += ic
            # Count activations within images that have category labels.
            # Note: This only makes sense with batch-size one
            # total_batch_categories += batch_categories
            cc = batch_categories[:,None] * ac[None,:]
            if key not in category_activation_counts:
                category_activation_counts[key] = cc
            else:
                category_activation_counts[key] += cc
    iou_scores = {}
    iqr_scores = {}
    for k in intersection_counts:
        iou_scores[k], iqr_scores[k] = score_tally_stats(
            label_category, total_counts, label_counts,
            category_activation_counts[k], intersection_counts[k])
    for k in intersection_counts:
        numpy.savez(os.path.join(outdir, safe_dir_name(k), 'bincounts.npz'),
                iou_scores=iou_scores[k].cpu().numpy(),
                iqr_scores=iqr_scores[k].cpu().numpy(),
                total_counts=total_counts.cpu().numpy(),
                label_counts=label_counts.cpu().numpy(),
                category_activation_counts=category_activation_counts[k]
                    .cpu().numpy(),
                intersection_counts=intersection_counts[k].cpu().numpy(),
                levels=levels[k].cpu().numpy())
    return (iou_scores, iqr_scores,
            total_counts, label_counts, category_activation_counts,
            intersection_counts)

def collect_cond_quantiles(outdir, model, segloader, segrunner):
    '''
    Returns maxiou and maxiou_level across the data set, one per layer.

    This is a performance-sensitive function.  Best performance is
    achieved with a counting scheme which assumes a segloader with
    batch_size 1.
    '''
    device = next(model.parameters()).device
    cached_cond_quantiles = {
            layer: load_conditional_quantile_if_present(os.path.join(outdir,
                safe_dir_name(layer)), 'cond_quantiles.npz') # on cpu
            for layer in model.retained_features() }
    label_fracs = load_npy_if_present(outdir, 'label_fracs.npy', 'cpu')
    if label_fracs is not None and all(
            value is not None for value in cached_cond_quantiles.values()):
        return cached_cond_quantiles, label_fracs

    labelcat, categories = segrunner.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0
                for l, c in labelcat]
    num_labels, num_categories = (len(n) for n in [labelcat, categories])

    # One-hot vector of category for each label
    labelcat = torch.zeros(num_labels, num_categories,
            dtype=torch.long, device=device)
    labelcat.scatter_(1, torch.from_numpy(numpy.array(label_category,
        dtype='int64')).to(device)[:,None], 1)
    # Running maxiou
    assert segloader.batch_size == 1 # category_activation_counts needs this.
    conditional_quantiles = {}
    label_counts = torch.zeros(num_labels, dtype=torch.long, device=device)
    pixel_count = 0
    progress = default_progress()
    scale_offset_map = getattr(model, 'scale_offset', None)
    upsample_grids = {}
    common_conditions = set()
    if label_fracs is None or label_fracs is 0:
        for i, batch in enumerate(progress(segloader, desc='label fracs')):
            seg, batch_label_counts, im, _ = segrunner.run_and_segment_batch(
                    batch, model, want_bincount=True, want_rgb=True)
            batch_label_counts = batch_label_counts.to(device)
            features = model.retained_features()
            # Accumulate bincounts and identify nonzeros
            label_counts += batch_label_counts[0]
            pixel_count += seg.shape[2] * seg.shape[3]
        label_fracs = (label_counts.cpu().float() / pixel_count)[:, None, None]
        numpy.save(os.path.join(outdir, 'label_fracs.npy'), label_fracs)

    skip_threshold = 1e-4
    skip_labels = set(i.item()
        for i in (label_fracs.view(-1) < skip_threshold).nonzero().view(-1))

    for layer in progress(model.retained_features().keys(), desc='CQ layers'):
        if cached_cond_quantiles.get(layer, None) is not None:
            conditional_quantiles[layer] = cached_cond_quantiles[layer]
            continue

        for i, batch in enumerate(progress(segloader, desc='Condquant')):
            seg, batch_label_counts, _, imshape = (
                    segrunner.run_and_segment_batch(
                         batch, model, want_bincount=True, want_rgb=True))
            bc = batch_label_counts.cpu()
            batch_label_counts = batch_label_counts.to(device)
            features = model.retained_features()
            # Accumulate bincounts and identify nonzeros
            label_counts += batch_label_counts[0]
            pixel_count += seg.shape[2] * seg.shape[3]
            batch_labels = bc[0].nonzero()[:,0]
            batch_categories = labelcat[batch_labels].max(0)[0]
            cpu_seg = None
            value = features[layer]
            if layer not in upsample_grids:
                upsample_grids[layer] = upsample_grid(value.shape[2:],
                        seg.shape[2:], imshape,
                        scale_offset=scale_offset_map.get(layer, None)
                            if scale_offset_map is not None else None,
                        dtype=value.dtype, device=value.device)
            if layer not in conditional_quantiles:
                conditional_quantiles[layer] = RunningConditionalQuantile(
                        resolution=2048)
            upsampled = torch.nn.functional.grid_sample(value,
                    upsample_grids[layer], padding_mode='border').view(
                            value.shape[1], -1)
            conditional_quantiles[layer].add(('all',), upsampled.t())
            cpu_upsampled = None
            for label in batch_labels.tolist():
                if label in skip_labels:
                    continue
                label_key = ('label', label)
                if label_key in common_conditions:
                    imask = (seg == label).max(dim=1)[0].view(-1)
                    intersected = upsampled[:, imask]
                    conditional_quantiles[layer].add(('label', label),
                            intersected.t())
                else:
                    if cpu_seg is None:
                        cpu_seg = seg.cpu()
                    if cpu_upsampled is None:
                        cpu_upsampled = upsampled.cpu()
                    imask = (cpu_seg == label).max(dim=1)[0].view(-1)
                    intersected = cpu_upsampled[:, imask]
                    conditional_quantiles[layer].add(('label', label),
                            intersected.t())
            if num_categories > 1:
                for cat in batch_categories.nonzero()[:,0]:
                    conditional_quantiles[layer].add(('cat', cat.item()),
                            upsampled.t())
            # Move the most common conditions to the GPU.
            if i and not i & (i - 1):  # if i is a power of 2:
                cq = conditional_quantiles[layer]
                common_conditions = set(cq.most_common_conditions(64))
                cq.to_('cpu', [k for k in cq.running_quantiles.keys()
                        if k not in common_conditions])
        # When a layer is done, get it off the GPU
        conditional_quantiles[layer].to_('cpu')

    label_fracs = (label_counts.cpu().float() / pixel_count)[:, None, None]

    for cq in conditional_quantiles.values():
        cq.to_('cpu')

    for layer in conditional_quantiles:
        save_state_dict(conditional_quantiles[layer],
            os.path.join(outdir, safe_dir_name(layer), 'cond_quantiles.npz'))
    numpy.save(os.path.join(outdir, 'label_fracs.npy'), label_fracs)

    return conditional_quantiles, label_fracs


def collect_maxiou(outdir, model, segloader, segrunner):
    '''
    Returns maxiou and maxiou_level across the data set, one per layer.

    This is a performance-sensitive function.  Best performance is
    achieved with a counting scheme which assumes a segloader with
    batch_size 1.
    '''
    device = next(model.parameters()).device
    conditional_quantiles, label_fracs = collect_cond_quantiles(
            outdir, model, segloader, segrunner)

    labelcat, categories = segrunner.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0
                for l, c in labelcat]
    num_labels, num_categories = (len(n) for n in [labelcat, categories])

    label_list = [('label', i) for i in range(num_labels)]
    category_list = [('all',)] if num_categories <= 1 else (
            [('cat', i) for i in range(num_categories)])
    max_iou, max_iou_level, max_iou_quantile = {}, {}, {}
    fracs = torch.logspace(-3, 0, 100)
    progress = default_progress()
    for layer, cq in progress(conditional_quantiles.items(), desc='Maxiou'):
        levels = cq.conditional(('all',)).quantiles(1 - fracs)
        denoms = 1 - cq.collected_normalize(category_list, levels)
        isects = (1 - cq.collected_normalize(label_list, levels)) * label_fracs
        unions = label_fracs + denoms[label_category, :, :] - isects
        iou = isects / unions
        # TODO: erase any for which threshold is bad
        max_iou[layer], level_bucket = iou.max(2)
        max_iou_level[layer] = levels[
                torch.arange(levels.shape[0])[None,:], level_bucket]
        max_iou_quantile[layer] = fracs[level_bucket]
    for layer in model.retained_features():
        numpy.savez(os.path.join(outdir, safe_dir_name(layer), 'max_iou.npz'),
            max_iou=max_iou[layer].cpu().numpy(),
            max_iou_level=max_iou_level[layer].cpu().numpy(),
            max_iou_quantile=max_iou_quantile[layer].cpu().numpy())
    return (max_iou, max_iou_level, max_iou_quantile)

def collect_iqr(outdir, model, segloader, segrunner):
    '''
    Returns iqr and iqr_level.

    This is a performance-sensitive function.  Best performance is
    achieved with a counting scheme which assumes a segloader with
    batch_size 1.
    '''
    max_iqr, max_iqr_level, max_iqr_quantile, max_iqr_iou  = {}, {}, {}, {}
    max_iqr_agreement = {}
    found_all = True
    for layer in model.retained_features():
        filename = os.path.join(outdir, safe_dir_name(layer), 'iqr.npz')
        if os.path.isfile(filename):
            data = numpy.load(filename)
            max_iqr[layer] = torch.from_numpy(data['max_iqr'])
            max_iqr_level[layer] = torch.from_numpy(data['max_iqr_level'])
            max_iqr_quantile[layer] = torch.from_numpy(data['max_iqr_quantile'])
            max_iqr_iou[layer] = torch.from_numpy(data['max_iqr_iou'])
            max_iqr_agreement[layer] = torch.from_numpy(
                    data['max_iqr_agreement'])
        else:
            found_all = False
    if found_all:
        return (max_iqr, max_iqr_level, max_iqr_quantile, max_iqr_iou,
            max_iqr_agreement)


    device = next(model.parameters()).device
    conditional_quantiles, label_fracs = collect_cond_quantiles(
            outdir, model, segloader, segrunner)

    labelcat, categories = segrunner.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0
                for l, c in labelcat]
    num_labels, num_categories = (len(n) for n in [labelcat, categories])

    label_list = [('label', i) for i in range(num_labels)]
    category_list = [('all',)] if num_categories <= 1 else (
            [('cat', i) for i in range(num_categories)])
    full_mi, full_je, full_iqr = {}, {}, {}
    fracs = torch.logspace(-3, 0, 100)
    progress = default_progress()
    for layer, cq in progress(conditional_quantiles.items(), desc='IQR'):
        levels = cq.conditional(('all',)).quantiles(1 - fracs)
        truth = label_fracs.to(device)
        preds = (1 - cq.collected_normalize(category_list, levels)
                )[label_category, :, :].to(device)
        cond_isects = 1 - cq.collected_normalize(label_list, levels).to(device)
        isects = cond_isects * truth
        unions = truth + preds - isects
        arr = torch.empty(size=(2, 2) + isects.shape, dtype=isects.dtype,
                device=device)
        arr[0, 0] = isects
        arr[0, 1] = preds - isects
        arr[1, 0] = truth - isects
        arr[1, 1] = 1 - unions
        arr.clamp_(0, 1)
        mi = mutual_information(arr)
        mi[:,:,-1] = 0  # at the 1.0 quantile should be no MI.
        # Don't trust mi when less than label_frac is less than 1e-3,
        # because our samples are too small.
        mi[label_fracs.view(-1) < 1e-3, :, :] = 0
        je = joint_entropy(arr)
        iqr = mi / je
        iqr[torch.isnan(iqr)] = 0 # Zero out any 0/0
        full_mi[layer] = mi.cpu()
        full_je[layer] = je.cpu()
        full_iqr[layer] = iqr.cpu()
        del mi, je
        agreement = isects + arr[1, 1]
        # When optimizing, maximize only over those pairs where the
        # unit is positively correlated with the label, and where the
        # threshold level is positive
        positive_iqr = iqr
        positive_iqr[agreement <= 0.8] = 0
        positive_iqr[(levels <= 0.0)[None, :, :].expand(positive_iqr.shape)] = 0
        # TODO: erase any for which threshold is bad
        maxiqr, level_bucket = positive_iqr.max(2)
        max_iqr[layer] = maxiqr.cpu()
        max_iqr_level[layer] = levels.to(device)[
                torch.arange(levels.shape[0])[None,:], level_bucket].cpu()
        max_iqr_quantile[layer] = fracs.to(device)[level_bucket].cpu()
        max_iqr_agreement[layer] = agreement[
                torch.arange(agreement.shape[0])[:, None],
                torch.arange(agreement.shape[1])[None, :],
                level_bucket].cpu()

        # Compute the iou that goes with each maximized iqr
        matching_iou = (isects[
                torch.arange(isects.shape[0])[:, None],
                torch.arange(isects.shape[1])[None, :],
                level_bucket] /
            unions[
                torch.arange(unions.shape[0])[:, None],
                torch.arange(unions.shape[1])[None, :],
                level_bucket])
        matching_iou[torch.isnan(matching_iou)] = 0
        max_iqr_iou[layer] = matching_iou.cpu()
    for layer in model.retained_features():
        numpy.savez(os.path.join(outdir, safe_dir_name(layer), 'iqr.npz'),
            max_iqr=max_iqr[layer].cpu().numpy(),
            max_iqr_level=max_iqr_level[layer].cpu().numpy(),
            max_iqr_quantile=max_iqr_quantile[layer].cpu().numpy(),
            max_iqr_iou=max_iqr_iou[layer].cpu().numpy(),
            max_iqr_agreement=max_iqr_agreement[layer].cpu().numpy(),
            full_mi=full_mi[layer].cpu().numpy(),
            full_je=full_je[layer].cpu().numpy(),
            full_iqr=full_iqr[layer].cpu().numpy())
    return (max_iqr, max_iqr_level, max_iqr_quantile, max_iqr_iou,
            max_iqr_agreement)

def mutual_information(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j,k]
            ind = arr[j,:].sum(dim=0) * arr[:,k].sum(dim=0)
            term = joint * (joint / ind).log()
            term[torch.isnan(term)] = 0
            total += term
    return total.clamp_(0)

def joint_entropy(arr):
    total = 0
    for j in range(arr.shape[0]):
        for k in range(arr.shape[1]):
            joint = arr[j,k]
            term = joint * joint.log()
            term[torch.isnan(term)] = 0
            total += term
    return (-total).clamp_(0)

def information_quality_ratio(arr):
    iqr = mutual_information(arr) / joint_entropy(arr)
    iqr[torch.isnan(iqr)] = 0
    return iqr

def collect_covariance(outdir, model, segloader, segrunner):
    '''
    Returns label_mean, label_variance, unit_mean, unit_variance,
    and cross_covariance across the data set.

    label_mean, label_variance (independent of model):
        treating the label as a one-hot, each label's mean and variance.
    unit_mean, unit_variance (one per layer): for each feature channel,
        the mean and variance of the activations in that channel.
    cross_covariance (one per layer): the cross covariance between the
        labels and the units in the layer.
    '''
    device = next(model.parameters()).device
    cached_covariance = {
            layer: load_covariance_if_present(os.path.join(outdir,
                safe_dir_name(layer)), 'covariance.npz', device=device)
            for layer in model.retained_features() }
    if all(value is not None for value in cached_covariance.values()):
        return cached_covariance
    labelcat, categories = segrunner.get_label_and_category_names()
    label_category = [categories.index(c) if c in categories else 0
                for l, c in labelcat]
    num_labels, num_categories = (len(n) for n in [labelcat, categories])

    # Running covariance
    cov = {}
    progress = default_progress()
    scale_offset_map = getattr(model, 'scale_offset', None)
    upsample_grids = {}
    for i, batch in enumerate(progress(segloader, desc='Covariance')):
        seg, _, _, imshape = segrunner.run_and_segment_batch(batch, model,
                want_rgb=True)
        features = model.retained_features()
        ohfeats = multilabel_onehot(seg, num_labels, ignore_index=0)
        # Accumulate bincounts and identify nonzeros
        for key, value in features.items():
            if key not in upsample_grids:
                upsample_grids[key] = upsample_grid(value.shape[2:],
                        seg.shape[2:], imshape,
                        scale_offset=scale_offset_map.get(key, None)
                            if scale_offset_map is not None else None,
                        dtype=value.dtype, device=value.device)
            upsampled = torch.nn.functional.grid_sample(value,
                    upsample_grids[key].expand(
                        (value.shape[0],) + upsample_grids[key].shape[1:]),
                    padding_mode='border')
            if key not in cov:
                cov[key] = RunningCrossCovariance()
            cov[key].add(upsampled, ohfeats)
    for layer in cov:
        save_state_dict(cov[layer],
                os.path.join(outdir, safe_dir_name(layer), 'covariance.npz'))
    return cov

def multilabel_onehot(labels, num_labels, dtype=None, ignore_index=None):
    '''
    Converts a multilabel tensor into a onehot tensor.

    The input labels is a tensor of shape (samples, multilabels, y, x).
    The output is a tensor of shape (samples, num_labels, y, x).
    If ignore_index is specified, labels with that index are ignored.
    Each x in labels should be 0 <= x < num_labels, or x == ignore_index.
    '''
    assert ignore_index is None or ignore_index <= 0
    if dtype is None:
        dtype = torch.float
    device = labels.device
    chans = num_labels + (-ignore_index if ignore_index else 0)
    outshape = (labels.shape[0], chans) + labels.shape[2:]
    result = torch.zeros(outshape, device=device, dtype=dtype)
    if ignore_index and ignore_index < 0:
        labels = labels + (-ignore_index)
    result.scatter_(1, labels, 1)
    if ignore_index and ignore_index < 0:
        result = result[:, -ignore_index:]
    elif ignore_index is not None:
        result[:, ignore_index] = 0
    return result

def load_npy_if_present(outdir, filename, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        return torch.from_numpy(data).to(device)
    return 0

def load_npz_if_present(outdir, filename, varnames, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        numpy_result = [data[n] for n in varnames]
        return tuple(torch.from_numpy(data).to(device) for data in numpy_result)
    return None

def load_quantile_if_present(outdir, filename, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        result = RunningQuantile(state=data)
        result.to_(device)
        return result
    return None

def load_conditional_quantile_if_present(outdir, filename):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        result = RunningConditionalQuantile(state=data)
        return result
    return None

def load_topk_if_present(outdir, filename, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        result = RunningTopK(state=data)
        result.to_(device)
        return result
    return None

def load_covariance_if_present(outdir, filename, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        result = RunningCrossCovariance(state=data)
        result.to_(device)
        return result
    return None

def save_state_dict(obj, filepath):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)
    dic = obj.state_dict()
    numpy.savez(filepath, **dic)

def upsample_grid(data_shape, target_shape, input_shape=None,
        scale_offset=None, dtype=torch.float, device=None):
    '''Prepares a grid to use with grid_sample to upsample a batch of
    features in data_shape to the target_shape. Can use scale_offset
    and input_shape to center the grid in a nondefault way: scale_offset
    maps feature pixels to input_shape pixels, and it is assumed that
    the target_shape is a uniform downsampling of input_shape.'''
    # Default is that nothing is resized.
    if target_shape is None:
        target_shape = data_shape
    # Make a default scale_offset to fill the image if there isn't one
    if scale_offset is None:
        scale = tuple(float(ts) / ds
                for ts, ds in zip(target_shape, data_shape))
        offset = tuple(0.5 * s - 0.5 for s in scale)
    else:
        scale, offset = (v for v in zip(*scale_offset))
        # Handle downsampling for different input vs target shape.
        if input_shape is not None:
            scale = tuple(s * (ts - 1) / (ns - 1)
                    for s, ns, ts in zip(scale, input_shape, target_shape))
            offset = tuple(o * (ts - 1) / (ns - 1)
                    for o, ns, ts in zip(offset, input_shape, target_shape))
    # Pytorch needs target coordinates in terms of source coordinates [-1..1]
    ty, tx = (((torch.arange(ts, dtype=dtype, device=device) - o)
                  * (2 / (s * (ss - 1))) - 1)
        for ts, ss, s, o, in zip(target_shape, data_shape, scale, offset))
    # Whoa, note that grid_sample reverses the order y, x -> x, y.
    grid = torch.stack(
        (tx[None,:].expand(target_shape), ty[:,None].expand(target_shape)),2
       )[None,:,:,:].expand((1, target_shape[0], target_shape[1], 2))
    return grid

def safe_dir_name(filename):
    keepcharacters = (' ','.','_','-')
    return ''.join(c
            for c in filename if c.isalnum() or c in keepcharacters).rstrip()

bargraph_palette = [
    ('#4B4CBF', '#B6B6F2'),
    ('#55B05B', '#B6F2BA'),
    ('#50BDAC', '#A5E5DB'),
    ('#81C679', '#C0FF9B'),
    ('#F0883B', '#F2CFB6'),
    ('#D4CF24', '#F2F1B6'),
    ('#D92E2B', '#F2B6B6'),
    ('#AB6BC6', '#CFAAFF'),
]

def make_svg_bargraph(labels, heights, categories,
        barheight=100, barwidth=12, show_labels=True, filename=None):
    # if len(labels) == 0:
    #     return # Nothing to do
    unitheight = float(barheight) / max(max(heights, default=1), 1)
    textheight = barheight if show_labels else 0
    labelsize = float(barwidth)
    gap = float(barwidth) / 4
    textsize = barwidth + gap
    rollup = max(heights, default=1)
    textmargin = float(labelsize) * 2 / 3
    leftmargin = 32
    rightmargin = 8
    svgwidth = len(heights) * (barwidth + gap) + 2 * leftmargin + rightmargin
    svgheight = barheight + textheight

    # create an SVG XML element
    svg = et.Element('svg', width=str(svgwidth), height=str(svgheight),
            version='1.1', xmlns='http://www.w3.org/2000/svg')

    # Draw the bar graph
    basey = svgheight - textheight
    x = leftmargin
    # Add units scale on left
    if len(heights):
        for h in [1, (max(heights) + 1) // 2, max(heights)]:
            et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;' +
                'text-anchor:end;alignment-baseline:hanging;' +
                'transform:translate(%dpx, %dpx);') %
                (textsize, x - gap, basey - h * unitheight)).text = str(h)
        et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;' +
                'text-anchor:middle;' +
                'transform:translate(%dpx, %dpx) rotate(-90deg)') %
                (textsize, x - gap - textsize, basey - h * unitheight / 2)
                ).text = 'units'
    # Draw big category background rectangles
    for catindex, (cat, catcount) in enumerate(categories):
        if not catcount:
            continue
        et.SubElement(svg, 'rect', x=str(x), y=str(basey - rollup * unitheight),
                width=(str((barwidth + gap) * catcount - gap)),
                height = str(rollup*unitheight),
                fill=bargraph_palette[catindex % len(bargraph_palette)][1])
        x += (barwidth + gap) * catcount
    # Draw small bars as well as 45degree text labels
    x = leftmargin
    catindex = -1
    catcount = 0
    for label, height in zip(labels, heights):
        while not catcount and catindex <= len(categories):
            catindex += 1
            catcount = categories[catindex][1]
            color = bargraph_palette[catindex % len(bargraph_palette)][0]
        et.SubElement(svg, 'rect', x=str(x), y=str(basey-(height * unitheight)),
                width=str(barwidth), height=str(height * unitheight),
                fill=color)
        x += barwidth
        if show_labels:
            et.SubElement(svg, 'text', x='0', y='0',
                style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
                'transform:translate(%dpx, %dpx) rotate(-45deg);') %
                (labelsize, x, basey + textmargin)).text = readable(label)
        x += gap
        catcount -= 1
    # Text labels for each category
    x = leftmargin
    for cat, catcount in categories:
        if not catcount:
            continue
        et.SubElement(svg, 'text', x='0', y='0',
            style=('font-family:sans-serif;font-size:%dpx;text-anchor:end;'+
            'transform:translate(%dpx, %dpx) rotate(-90deg);') %
            (textsize, x + (barwidth + gap) * catcount - gap,
                basey - rollup * unitheight + gap)).text = '%d %s' % (
                    catcount, readable(cat + ('s' if catcount != 1 else '')))
        x += (barwidth + gap) * catcount
    # Output - this is the bare svg.
    result = et.tostring(svg)
    if filename:
        f = open(filename, 'wb')
        # When writing to a file a special header is needed.
        f.write(''.join([
            '<?xml version=\"1.0\" standalone=\"no\"?>\n',
            '<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n',
            '\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n']
            ).encode('utf-8'))
        f.write(result)
        f.close()
    return result

readable_replacements = [(re.compile(r[0]), r[1]) for r in [
    (r'-[sc]$', ''),
    (r'_', ' '),
    ]]

def readable(label):
    for pattern, subst in readable_replacements:
        label= re.sub(pattern, subst, label)
    return label

def reverse_normalize_from_transform(transform):
    '''
    Crawl around the transforms attached to a dataset looking for a
    Normalize transform, and return it a corresponding ReverseNormalize,
    or None if no normalization is found.
    '''
    if isinstance(transform, torchvision.transforms.Normalize):
        return ReverseNormalize(transform.mean, transform.std)
    t = getattr(transform, 'transform', None)
    if t is not None:
        return reverse_normalize_from_transform(t)
    transforms = getattr(transform, 'transforms', None)
    if transforms is not None:
        for t in reversed(transforms):
            result = reverse_normalize_from_transform(t)
            if result is not None:
                return result
    return None

class ReverseNormalize:
    '''
    Applies the reverse of torchvision.transforms.Normalize.
    '''
    def __init__(self, mean, stdev):
        mean = numpy.array(mean)
        stdev = numpy.array(stdev)
        self.mean = torch.from_numpy(mean)[None,:,None,None].float()
        self.stdev = torch.from_numpy(stdev)[None,:,None,None].float()
    def __call__(self, data):
        device = data.device
        return data.mul(self.stdev.to(device)).add_(self.mean.to(device))

class ImageOnlySegRunner:
    def __init__(self, dataset, recover_image=None):
        if recover_image is None:
            recover_image = reverse_normalize_from_transform(dataset)
        self.recover_image = recover_image
        self.dataset = dataset
    def get_label_and_category_names(self):
        return [('-', '-')], ['-']
    def run_and_segment_batch(self, batch, model,
            want_bincount=False, want_rgb=False):
        [im] = batch
        device = next(model.parameters()).device
        if want_rgb:
            rgb = self.recover_image(im.clone()
                ).permute(0, 2, 3, 1).mul_(255).clamp(0, 255).byte()
        else:
            rgb = None
        # Stubs for seg and bc
        seg = torch.zeros(im.shape[0], 1, 1, 1, dtype=torch.long)
        bc = torch.ones(im.shape[0], 1, dtype=torch.long)
        # Run the model.
        model(im.to(device))
        return seg, bc, rgb, im.shape[2:]

class ClassifierSegRunner:
    def __init__(self, dataset, recover_image=None):
        # The dataset contains explicit segmentations
        if recover_image is None:
            recover_image = reverse_normalize_from_transform(dataset)
        self.recover_image = recover_image
        self.dataset = dataset
    def get_label_and_category_names(self):
        catnames = self.dataset.categories
        label_and_cat_names = [(readable(label),
            catnames[self.dataset.label_category[i]])
                for i, label in enumerate(self.dataset.labels)]
        return label_and_cat_names, catnames
    def run_and_segment_batch(self, batch, model,
            want_bincount=False, want_rgb=False):
        '''
        Runs the dissected model on one batch of the dataset, and
        returns a multilabel semantic segmentation for the data.
        Given a batch of size (n, c, y, x) the segmentation should
        be a (long integer) tensor of size (n, d, y//r, x//r) where
        d is the maximum number of simultaneous labels given to a pixel,
        and where r is some (optional) resolution reduction factor.
        In the segmentation returned, the label `0` is reserved for
        the background "no-label".

        In addition to the segmentation, bc, rgb, and shape are returned
        where bc is a per-image bincount counting returned label pixels,
        rgb is a viewable (n, y, x, rgb) byte image tensor for the data
        for visualizations (reversing normalizations, for example), and
        shape is the (y, x) size of the data.  If want_bincount or
        want_rgb are False, those return values may be None.
        '''
        im, seg, bc = batch
        device = next(model.parameters()).device
        if want_rgb:
            rgb = self.recover_image(im.clone()
                ).permute(0, 2, 3, 1).mul_(255).clamp(0, 255).byte()
        else:
            rgb = None
        # Run the model.
        model(im.to(device))
        return seg, bc, rgb, im.shape[2:]

class GeneratorSegRunner:
    def __init__(self, segmenter):
        # The segmentations are given by an algorithm
        if segmenter is None:
            segmenter = UnifiedParsingSegmenter(segsizes=[256], segdiv='quad')
        self.segmenter = segmenter
        self.num_classes = len(segmenter.get_label_and_category_names()[0])
    def get_label_and_category_names(self):
        return self.segmenter.get_label_and_category_names()
    def run_and_segment_batch(self, batch, model,
            want_bincount=False, want_rgb=False):
        '''
        Runs the dissected model on one batch of the dataset, and
        returns a multilabel semantic segmentation for the data.
        Given a batch of size (n, c, y, x) the segmentation should
        be a (long integer) tensor of size (n, d, y//r, x//r) where
        d is the maximum number of simultaneous labels given to a pixel,
        and where r is some (optional) resolution reduction factor.
        In the segmentation returned, the label `0` is reserved for
        the background "no-label".

        In addition to the segmentation, bc, rgb, and shape are returned
        where bc is a per-image bincount counting returned label pixels,
        rgb is a viewable (n, y, x, rgb) byte image tensor for the data
        for visualizations (reversing normalizations, for example), and
        shape is the (y, x) size of the data.  If want_bincount or
        want_rgb are False, those return values may be None.
        '''
        device = next(model.parameters()).device
        z_batch = batch[0]
        tensor_images = model(z_batch.to(device))
        seg = self.segmenter.segment_batch(tensor_images, downsample=2)
        if want_bincount:
            index = torch.arange(z_batch.shape[0],
                    dtype=torch.long, device=device)
            bc = (seg + index[:, None, None, None] * self.num_classes).view(-1
                ).bincount(minlength=z_batch.shape[0] * self.num_classes)
            bc = bc.view(z_batch.shape[0], self.num_classes)
        else:
            bc = None
        if want_rgb:
            images = ((tensor_images + 1) / 2 * 255)
            rgb = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
        else:
            rgb = None
        return seg, bc, rgb, tensor_images.shape[2:]
