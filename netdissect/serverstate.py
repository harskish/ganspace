import os, torch, numpy, base64, json, re, threading, random
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from netdissect.easydict import EasyDict
from netdissect.modelconfig import create_instrumented_model
from netdissect.runningstats import RunningQuantile
from netdissect.dissection import safe_dir_name
from netdissect.zdataset import z_sample_for_model
from PIL import Image
from io import BytesIO

class DissectionProject:
    '''
    DissectionProject understand how to drive a GanTester within a
    dissection project directory structure: it caches data in files,
    creates image files, and translates data between plain python data
    types and the pytorch-specific tensors required by GanTester.
    '''
    def __init__(self, config, project_dir, path_url, public_host):
        print('config done', project_dir)
        self.use_cuda = torch.cuda.is_available()
        self.dissect = config
        self.project_dir = project_dir
        self.path_url = path_url
        self.public_host = public_host
        self.cachedir = os.path.join(self.project_dir, 'cache')
        self.tester = GanTester(
                config.settings, dissectdir=project_dir,
                device=torch.device('cuda') if self.use_cuda
                     else torch.device('cpu'))
        self.stdz = []

    def get_zs(self, size):
        if size <= len(self.stdz):
            return self.stdz[:size].tolist()
        z_tensor = self.tester.standard_z_sample(size)
        numpy_z = z_tensor.cpu().numpy()
        self.stdz = numpy_z
        return self.stdz.tolist()

    def get_z(self, id):
        if id < len(self.stdz):
            return self.stdz[id]
        return self.get_zs((id + 1) * 2)[id]

    def get_zs_for_ids(self, ids):
        max_id = max(ids)
        if max_id >= len(self.stdz):
            self.get_z(max_id)
        return self.stdz[ids]

    def get_layers(self):
        result = []
        layer_shapes = self.tester.layer_shapes()
        for layer in self.tester.layers:
            shape = layer_shapes[layer]
            result.append(dict(
                layer=layer,
                channels=shape[1],
                shape=[shape[2], shape[3]]))
        return result

    def get_units(self, layer):
        try:
            dlayer = [dl for dl in self.dissect['layers']
                    if dl['layer'] == layer][0]
        except:
            return None

        dunits = dlayer['units']
        result = [dict(unit=unit_num,
                       img='/%s/%s/s-image/%d-top.jpg' %
                        (self.path_url, layer, unit_num),
                       label=unit['iou_label'])
                  for unit_num, unit in enumerate(dunits)]
        return result

    def get_rankings(self, layer):
        try:
            dlayer = [dl for dl in self.dissect['layers']
                    if dl['layer'] == layer][0]
        except:
            return None
        result = [dict(name=ranking['name'],
                       metric=ranking.get('metric', None),
                       scores=ranking['score'])
                  for ranking in dlayer['rankings']]
        return result

    def get_levels(self, layer, quantiles):
        levels = self.tester.levels(
                layer, torch.from_numpy(numpy.array(quantiles)))
        return levels.cpu().numpy().tolist()

    def generate_images(self, zs, ids, interventions, return_urls=False):
        if ids is not None:
            assert zs is None
            zs = self.get_zs_for_ids(ids)
            if not interventions:
                # Do file caching when ids are given (and no ablations).
                imgdir = os.path.join(self.cachedir, 'img', 'id')
                os.makedirs(imgdir, exist_ok=True)
                exist = set(os.listdir(imgdir))
                unfinished = [('%d.jpg' % id) not in exist for id in ids]
                needed_z_tensor = torch.tensor(zs[unfinished]).float().to(
                        self.tester.device)
                needed_ids = numpy.array(ids)[unfinished]
                # Generate image files for just the needed images.
                if len(needed_z_tensor):
                    imgs = self.tester.generate_images(needed_z_tensor
                            ).cpu().numpy()
                    for i, img in zip(needed_ids, imgs):
                         Image.fromarray(img.transpose(1, 2, 0)).save(
                                 os.path.join(imgdir, '%d.jpg' % i), 'jpeg',
                                 quality=99, optimize=True, progressive=True)
                # Assemble a response.
                imgurls = ['/%s/cache/img/id/%d.jpg'
                      % (self.path_url, i) for i in ids]
                return [dict(id=i, d=d) for i, d in zip(ids, imgurls)]
        # No file caching when ids are not given (or ablations are applied)
        z_tensor = torch.tensor(zs).float().to(self.tester.device)
        imgs = self.tester.generate_images(z_tensor,
                intervention=decode_intervention_array(interventions,
                    self.tester.layer_shapes()),
                ).cpu().numpy()
        numpy_z = z_tensor.cpu().numpy()
        if return_urls:
            randdir = '%03d' % random.randrange(1000)
            imgdir = os.path.join(self.cachedir, 'img', 'uniq', randdir)
            os.makedirs(imgdir, exist_ok=True)
            startind = random.randrange(100000)
            imgurls = []
            for i, img in enumerate(imgs):
                filename = '%d.jpg' % (i + startind)
                Image.fromarray(img.transpose(1, 2, 0)).save(
                         os.path.join(imgdir, filename), 'jpeg',
                         quality=99, optimize=True, progressive=True)
                image_url_path = ('/%s/cache/img/uniq/%s/%s'
                      % (self.path_url, randdir, filename))
                imgurls.append(image_url_path)
                tweet_filename = 'tweet-%d.html' % (i + startind)
                tweet_url_path = ('/%s/cache/img/uniq/%s/%s'
                      % (self.path_url, randdir, tweet_filename))
                with open(os.path.join(imgdir, tweet_filename), 'w') as f:
                    f.write(twitter_card(image_url_path, tweet_url_path,
                        self.public_host))
            return [dict(d=d) for d in imgurls]
        imgurls = [img2base64(img.transpose(1, 2, 0)) for img in imgs]
        return [dict(d=d) for d in imgurls]

    def get_features(self, ids, masks, layers, interventions):
        zs = self.get_zs_for_ids(ids)
        z_tensor = torch.tensor(zs).float().to(self.tester.device)
        t_masks = torch.stack(
                [torch.from_numpy(mask_to_numpy(mask)) for mask in masks]
                )[:,None,:,:].to(self.tester.device)
        t_features = self.tester.feature_stats(z_tensor, t_masks,
                decode_intervention_array(interventions,
                    self.tester.layer_shapes()), layers)
        # Convert torch arrays to plain python lists before returning.
        return { layer: { key: value.cpu().numpy().tolist()
                          for key, value in feature.items() }
                 for layer, feature in t_features.items() }

    def get_featuremaps(self, ids, layers, interventions):
        zs = self.get_zs_for_ids(ids)
        z_tensor = torch.tensor(zs).float().to(self.tester.device)
        # Quantilized features are returned.
        q_features = self.tester.feature_maps(z_tensor,
                decode_intervention_array(interventions,
                    self.tester.layer_shapes()), layers)
        # Scale them 0-255 and return them.
        # TODO: turn them into pngs for returning.
        return { layer: [
            value.clamp(0, 1).mul(255).byte().cpu().numpy().tolist()
            for value in valuelist ]
            for layer, valuelist in q_features.items()
            if (not layers) or (layer in layers) }

    def get_recipes(self):
        recipedir = os.path.join(self.project_dir, 'recipe')
        if not os.path.isdir(recipedir):
            return []
        result = []
        for filename in os.listdir(recipedir):
            with open(os.path.join(recipedir, filename)) as f:
                result.append(json.load(f))
        return result




class GanTester:
    '''
    GanTester holds on to a specific model to test.

    (1) loads and instantiates the GAN;
    (2) instruments it at every layer so that units can be ablated
    (3) precomputes z dimensionality, and output image dimensions.
    '''
    def __init__(self, args, dissectdir=None, device=None):
        self.cachedir = os.path.join(dissectdir, 'cache')
        self.device = device if device is not None else torch.device('cpu')
        self.dissectdir = dissectdir
        self.modellock = threading.Lock()

        # Load the generator from the pth file.
        args_copy = EasyDict(args)
        args_copy.edit = True
        model = create_instrumented_model(args_copy)
        model.eval()
        self.model = model

        # Get the set of layers of interest.
        # Default: all shallow children except last.
        self.layers = sorted(model.retained_features().keys())

        # Move it to CUDA if wanted.
        model.to(device)

        self.quantiles = {
            layer: load_quantile_if_present(os.path.join(self.dissectdir,
                safe_dir_name(layer)), 'quantiles.npz',
                device=torch.device('cpu'))
            for layer in self.layers }

    def layer_shapes(self):
        return self.model.feature_shape

    def standard_z_sample(self, size=100, seed=1, device=None):
        '''
        Generate a standard set of random Z as a (size, z_dimension) tensor.
        With the same random seed, it always returns the same z (e.g.,
        the first one is always the same regardless of the size.)
        '''
        result = z_sample_for_model(self.model, size)
        if device is not None:
            result = result.to(device)
        return result

    def reset_intervention(self):
        self.model.remove_edits()

    def apply_intervention(self, intervention):
        '''
        Applies an ablation recipe of the form [(layer, unit, alpha)...].
        '''
        self.reset_intervention()
        if not intervention:
            return
        for layer, (a, v) in intervention.items():
            self.model.edit_layer(layer, ablation=a, replacement=v)

    def generate_images(self, z_batch, intervention=None):
        '''
        Makes some images.
        '''
        with torch.no_grad(), self.modellock:
            batch_size = 10
            self.apply_intervention(intervention)
            test_loader = DataLoader(TensorDataset(z_batch[:,:,None,None]),
                batch_size=batch_size,
                pin_memory=('cuda' == self.device.type
                            and z_batch.device.type == 'cpu'))
            result_img = torch.zeros(
                    *((len(z_batch), 3) + self.model.output_shape[2:]),
                    dtype=torch.uint8, device=self.device)
            for batch_num, [batch_z,] in enumerate(test_loader):
                batch_z = batch_z.to(self.device)
                out = self.model(batch_z)
                result_img[batch_num*batch_size:
                        batch_num*batch_size+len(batch_z)] = (
                                (((out + 1) / 2) * 255).clamp(0, 255).byte())
            return result_img

    def get_layers(self):
        return self.layers

    def feature_stats(self, z_batch,
            masks=None, intervention=None, layers=None):
        feature_stat = defaultdict(dict)
        with torch.no_grad(), self.modellock:
            batch_size = 10
            self.apply_intervention(intervention)
            if masks is None:
                masks = torch.ones(z_batch.size(0), 1, 1, 1,
                        device=z_batch.device, dtype=z_batch.dtype)
            else:
                assert masks.shape[0] == z_batch.shape[0]
                assert masks.shape[1] == 1
            test_loader = DataLoader(
                TensorDataset(z_batch[:,:,None,None], masks),
                batch_size=batch_size,
                pin_memory=('cuda' == self.device.type
                    and z_batch.device.type == 'cpu'))
            processed = 0
            for batch_num, [batch_z, batch_m] in enumerate(test_loader):
                batch_z, batch_m = [
                        d.to(self.device) for d in [batch_z, batch_m]]
                # Run model but disregard output
                self.model(batch_z)
                processing = batch_z.shape[0]
                for layer, feature in self.model.retained_features().items():
                    if layers is not None:
                        if layer not in layers:
                            continue
                    # Compute max features touching mask
                    resized_max = torch.nn.functional.adaptive_max_pool2d(
                            batch_m,
                            (feature.shape[2], feature.shape[3]))
                    max_feature = (feature * resized_max).view(
                            feature.shape[0], feature.shape[1], -1
                            ).max(2)[0].max(0)[0]
                    if 'max' not in feature_stat[layer]:
                        feature_stat[layer]['max'] = max_feature
                    else:
                        torch.max(feature_stat[layer]['max'], max_feature,
                                    out=feature_stat[layer]['max'])
                    # Compute mean features weighted by overlap with mask
                    resized_mean = torch.nn.functional.adaptive_avg_pool2d(
                            batch_m,
                            (feature.shape[2], feature.shape[3]))
                    mean_feature = (feature * resized_mean).view(
                            feature.shape[0], feature.shape[1], -1
                            ).sum(2).sum(0) / (resized_mean.sum() + 1e-15)
                    if 'mean' not in feature_stat[layer]:
                        feature_stat[layer]['mean'] = mean_feature
                    else:
                        feature_stat[layer]['mean'] = (
                                processed * feature_mean[layer]['mean']
                                + processing * mean_feature) / (
                                        processed + processing)
                processed += processing
            # After summaries are done, also compute quantile stats
            for layer, stats in feature_stat.items():
                if self.quantiles.get(layer, None) is not None:
                    for statname in ['max', 'mean']:
                        stats['%s_quantile' % statname] = (
                            self.quantiles[layer].normalize(stats[statname]))
        return feature_stat

    def levels(self, layer, quantiles):
        return self.quantiles[layer].quantiles(quantiles)

    def feature_maps(self, z_batch, intervention=None, layers=None,
            quantiles=True):
        feature_map = defaultdict(list)
        with torch.no_grad(), self.modellock:
            batch_size = 10
            self.apply_intervention(intervention)
            test_loader = DataLoader(
                TensorDataset(z_batch[:,:,None,None]),
                batch_size=batch_size,
                pin_memory=('cuda' == self.device.type
                    and z_batch.device.type == 'cpu'))
            processed = 0
            for batch_num, [batch_z] in enumerate(test_loader):
                batch_z = batch_z.to(self.device)
                # Run model but disregard output
                self.model(batch_z)
                processing = batch_z.shape[0]
                for layer, feature in self.model.retained_features().items():
                    for single_featuremap in feature:
                        if quantiles:
                            feature_map[layer].append(self.quantiles[layer]
                                    .normalize(single_featuremap))
                        else:
                            feature_map[layer].append(single_featuremap)
        return feature_map

def load_quantile_if_present(outdir, filename, device):
    filepath = os.path.join(outdir, filename)
    if os.path.isfile(filepath):
        data = numpy.load(filepath)
        result = RunningQuantile(state=data)
        result.to_(device)
        return result
    return None

if __name__ == '__main__':
    test_main()

def mask_to_numpy(mask_record):
    # Detect a png image mask.
    bitstring = mask_record['bitstring']
    bitnumpy = None
    default_shape = (256, 256)
    if 'image/png;base64,' in bitstring:
        bitnumpy = base642img(bitstring)
        default_shape = bitnumpy.shape[:2]
    # Set up results
    shape = mask_record.get('shape', None)
    if not shape: # None or empty []
        shape = default_shape
    result = numpy.zeros(shape=shape, dtype=numpy.float32)
    bitbounds = mask_record.get('bitbounds', None)
    if not bitbounds: # None or empty []
        bitbounds = ([0] * len(result.shape)) + list(result.shape)
    start = bitbounds[:len(result.shape)]
    end = bitbounds[len(result.shape):]
    if bitnumpy is not None:
        if bitnumpy.shape[2] == 4:
            # Mask is any nontransparent bits in the alpha channel if present
            result[start[0]:end[0], start[1]:end[1]] = (bitnumpy[:,:,3] > 0)
        else:
            # Or any nonwhite pixels in the red channel if no alpha.
            result[start[0]:end[0], start[1]:end[1]] = (bitnumpy[:,:,0] < 255)
        return result
    else:
        # Or bitstring can be just ones and zeros.
        indexes = start.copy()
        bitindex = 0
        while True:
            result[tuple(indexes)] = (bitstring[bitindex] != '0')
            for ii in range(len(indexes) - 1, -1, -1):
                if indexes[ii] < end[ii] - 1:
                    break
                indexes[ii] = start[ii]
            else:
                assert (bitindex + 1) == len(bitstring)
                return result
            indexes[ii] += 1
            bitindex += 1

def decode_intervention_array(interventions, layer_shapes):
    result = {}
    for channels in [decode_intervention(intervention, layer_shapes)
            for intervention in (interventions or [])]:
        for layer, channel in channels.items():
            if layer not in result:
                result[layer] = channel
                continue
            accum = result[layer]
            newalpha = 1 - (1 - channel[:1]) * (1 - accum[:1])
            newvalue = (accum[1:] * accum[:1] * (1 - channel[:1]) +
                    channel[1:] * channel[:1]) / (newalpha + 1e-40)
            accum[:1] = newalpha
            accum[1:] = newvalue
    return result

def decode_intervention(intervention, layer_shapes):
    # Every plane of an intervention is a solid choice of activation
    # over a set of channels, with a mask applied to alpha-blended channels
    # (when the mask resolution is different from the feature map, it can
    # be either a max-pooled or average-pooled to the proper resolution).
    # This can be reduced to a single alpha-blended featuremap.
    if intervention is None:
        return None
    mask = intervention.get('mask', None)
    if mask:
        mask = torch.from_numpy(mask_to_numpy(mask))
    maskpooling = intervention.get('maskpooling', 'max')
    channels = {}  # layer -> ([alpha, val], c)
    for arec in intervention.get('ablations', []):
        unit = arec['unit']
        layer = arec['layer']
        alpha = arec.get('alpha', 1.0)
        if alpha is None:
            alpha = 1.0
        value = arec.get('value', 0.0)
        if value is None:
            value = 0.0
        if alpha != 0.0 or value != 0.0:
            if layer not in channels:
                channels[layer] = torch.zeros(2, *layer_shapes[layer][1:])
            channels[layer][0, unit] = alpha
            channels[layer][1, unit] = value
    if mask is not None:
        for layer in channels:
            layer_shape = layer_shapes[layer][2:]
            if maskpooling == 'mean':
                layer_mask = torch.nn.functional.adaptive_avg_pool2d(
                    mask[None,None,...], layer_shape)[0]
            else:
                layer_mask = torch.nn.functional.adaptive_max_pool2d(
                    mask[None,None,...], layer_shape)[0]
            channels[layer][0] *= layer_mask
    return channels

def img2base64(imgarray, for_html=True, image_format='jpeg'):
    '''
    Converts a numpy array to a jpeg base64 url
    '''
    input_image_buff = BytesIO()
    Image.fromarray(imgarray).save(input_image_buff, image_format,
            quality=99, optimize=True, progressive=True)
    res = base64.b64encode(input_image_buff.getvalue()).decode('ascii')
    if for_html:
        return 'data:image/' + image_format + ';base64,' + res
    else:
        return res

def base642img(stringdata):
    stringdata = re.sub('^(?:data:)?image/\w+;base64,', '', stringdata)
    im = Image.open(BytesIO(base64.b64decode(stringdata)))
    return numpy.array(im)

def twitter_card(image_path, tweet_path, public_host):
    return '''\
<!doctype html>
<html>
<head>
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="Painting with GANs from MIT-IBM Watson AI Lab" />
<meta name="twitter:description" content="This demo lets you modify a selection of meaningful GAN units for a generated image by simply painting." />
<meta name="twitter:image" content="http://{public_host}{image_path}" />
<meta name="twitter:url" content="http://{public_host}{tweet_path}" />
<meta http-equiv="refresh" content="10; url=http://bit.ly/ganpaint">
</head>
<style>
body {{ font: 12px Arial, sans-serif; }}
</style>
<body>
<center>
<h1>Painting with GANs from MIT-IBM Watson AI Lab</h1>
<p>This demo lets you modify a selection of meatningful GAN units for a generated image by simply painting.</p>
<img src="{image_path}">
<p>Redirecting to
<a href="http://bit.ly/ganpaint">GANPaint</a>
</p>
</center>
</body>
'''.format(
        image_path=image_path,
        tweet_path=tweet_path,
        public_host=public_host)
