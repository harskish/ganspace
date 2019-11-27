# Usage as a simple differentiable segmenter base class

import os, torch, numpy, json, glob
import skimage.morphology
from collections import OrderedDict
from netdissect import upsegmodel
from netdissect import segmodel as segmodel_module
from netdissect.easydict import EasyDict
from urllib.request import urlretrieve

class BaseSegmenter:
    def get_label_and_category_names(self):
        '''
        Returns two lists: first, a list of tuples [(label, category), ...]
        where the label and category are human-readable strings indicating
        the meaning of a segmentation class.  The 0th segmentation class
        should be reserved for a label ('-') that means "no prediction."
        The second list should just be a list of [category,...] listing
        all categories in a canonical order.
        '''
        raise NotImplemented()

    def segment_batch(self, tensor_images, downsample=1):
        '''
        Returns a multilabel segmentation for the given batch of (RGB [-1...1])
        images.  Each pixel of the result is a torch.long indicating a
        predicted class number.  Multiple classes can be predicted for
        the same pixel: output shape is (n, multipred, y, x), where
        multipred is 3, 5, or 6, for how many different predicted labels can
        be given for each pixel (depending on whether subdivision is being
        used).  If downsample is specified, then the output y and x dimensions
        are downsampled from the original image.
        '''
        raise NotImplemented()

    def predict_single_class(self, tensor_images, classnum, downsample=1):
        '''
        Given a batch of images (RGB, normalized to [-1...1]) and
        a specific segmentation class number, returns a tuple with
           (1) a differentiable ([0..1]) prediction score for the class
               at every pixel of the input image.
           (2) a binary mask showing where in the input image the
               specified class is the best-predicted label for the pixel.
        Does not work on subdivided labels.
        '''
        raise NotImplemented()

class UnifiedParsingSegmenter(BaseSegmenter):
    '''
    This is a wrapper for a more complicated multi-class segmenter,
    as described in https://arxiv.org/pdf/1807.10221.pdf, and as
    released in https://github.com/CSAILVision/unifiedparsing.
    For our purposes and to simplify processing, we do not use
    whole-scene predictions, and we only consume part segmentations
    for the three largest object classes (sky, building, person).
    '''

    def __init__(self, segsizes=None, segdiv=None):
        # Create a segmentation model
        if segsizes is None:
            segsizes = [256]
        if segdiv == None:
            segdiv = 'undivided'
        segvocab = 'upp'
        segarch = ('resnet50', 'upernet')
        epoch = 40
        segmodel = load_unified_parsing_segmentation_model(
                segarch, segvocab, epoch)
        segmodel.cuda()
        self.segmodel = segmodel
        self.segsizes = segsizes
        self.segdiv = segdiv
        mult = 1
        if self.segdiv == 'quad':
            mult = 5
        self.divmult = mult
        # Assign class numbers for parts.
        first_partnumber = (
                (len(segmodel.labeldata['object']) - 1) * mult + 1 +
                (len(segmodel.labeldata['material']) - 1))
        # We only use parts for these three types of objects, for efficiency.
        partobjects = ['sky', 'building', 'person']
        partnumbers = {}
        partnames = []
        objectnumbers = {k: v
                for v, k in enumerate(segmodel.labeldata['object'])}
        part_index_translation = []
        # We merge some classes.  For example "door" is both an object
        # and a part of a building.  To avoid confusion, we just count
        # such classes as objects, and add part scores to the same index.
        for owner in partobjects:
            part_list = segmodel.labeldata['object_part'][owner]
            numeric_part_list = []
            for part in part_list:
                if part in objectnumbers:
                    numeric_part_list.append(objectnumbers[part])
                elif part in partnumbers:
                    numeric_part_list.append(partnumbers[part])
                else:
                    partnumbers[part] = len(partnames) + first_partnumber
                    partnames.append(part)
                    numeric_part_list.append(partnumbers[part])
            part_index_translation.append(torch.tensor(numeric_part_list))
        self.objects_with_parts = [objectnumbers[obj] for obj in partobjects]
        self.part_index = part_index_translation
        self.part_names = partnames
        # For now we'll just do object and material labels.
        self.num_classes = 1 + (
                len(segmodel.labeldata['object']) - 1) * mult + (
                len(segmodel.labeldata['material']) - 1) + len(partnames)
        self.num_object_classes = len(self.segmodel.labeldata['object']) - 1

    def get_label_and_category_names(self, dataset=None):
        '''
        Lists label and category names.
        '''
        # Labels are ordered as follows:
        # 0, [object labels] [divided object labels] [materials] [parts]
        # The zero label is reserved to mean 'no prediction'.
        if self.segdiv == 'quad':
            suffixes = ['t', 'l', 'b', 'r']
        else:
            suffixes = []
        divided_labels = []
        for suffix in suffixes:
            divided_labels.extend([('%s-%s' % (label, suffix), 'part')
                for label in self.segmodel.labeldata['object'][1:]])
        # Create the whole list of labels
        labelcats = (
                [(label, 'object')
                    for label in self.segmodel.labeldata['object']] +
                divided_labels +
                [(label, 'material')
                    for label in self.segmodel.labeldata['material'][1:]] +
                [(label, 'part') for label in self.part_names])
        return labelcats, ['object', 'part', 'material']

    def raw_seg_prediction(self, tensor_images, downsample=1):
        '''
        Generates a segmentation by applying multiresolution voting on
        the segmentation model, using (rounded to 32 pixels) a set of
        resolutions in the example benchmark code.
        '''
        y, x = tensor_images.shape[2:]
        b = len(tensor_images)
        tensor_images = (tensor_images + 1) / 2 * 255
        tensor_images = torch.flip(tensor_images, (1,)) # BGR!!!?
        tensor_images -= torch.tensor([102.9801, 115.9465, 122.7717]).to(
                   dtype=tensor_images.dtype, device=tensor_images.device
                   )[None,:,None,None]
        seg_shape = (y // downsample, x // downsample)
        # We want these to be multiples of 32 for the model.
        sizes = [(s, s) for s in self.segsizes]
        pred = {category: torch.zeros(
            len(tensor_images), len(self.segmodel.labeldata[category]),
            seg_shape[0], seg_shape[1]).cuda()
            for category in ['object', 'material']}
        part_pred = {partobj_index: torch.zeros(
            len(tensor_images), len(partindex),
            seg_shape[0], seg_shape[1]).cuda()
            for partobj_index, partindex in enumerate(self.part_index)}
        for size in sizes:
            if size == tensor_images.shape[2:]:
                resized = tensor_images
            else:
                resized = torch.nn.AdaptiveAvgPool2d(size)(tensor_images)
            r_pred = self.segmodel(
                dict(img=resized), seg_size=seg_shape)
            for k in pred:
                pred[k] += r_pred[k]
            for k in part_pred:
                part_pred[k] += r_pred['part'][k]
        return pred, part_pred

    def segment_batch(self, tensor_images, downsample=1):
        '''
        Returns a multilabel segmentation for the given batch of (RGB [-1...1])
        images.  Each pixel of the result is a torch.long indicating a
        predicted class number.  Multiple classes can be predicted for
        the same pixel: output shape is (n, multipred, y, x), where
        multipred is 3, 5, or 6, for how many different predicted labels can
        be given for each pixel (depending on whether subdivision is being
        used).  If downsample is specified, then the output y and x dimensions
        are downsampled from the original image.
        '''
        pred, part_pred = self.raw_seg_prediction(tensor_images,
                downsample=downsample)
        piece_channels = 2 if self.segdiv == 'quad' else 0
        y, x = tensor_images.shape[2:]
        seg_shape = (y // downsample, x // downsample)
        segs = torch.zeros(len(tensor_images), 3 + piece_channels,
                seg_shape[0], seg_shape[1],
                dtype=torch.long, device=tensor_images.device)
        _, segs[:,0] = torch.max(pred['object'], dim=1)
        # Get materials and translate to shared numbering scheme
        _, segs[:,1] = torch.max(pred['material'], dim=1)
        maskout = (segs[:,1] == 0)
        segs[:,1] += (len(self.segmodel.labeldata['object']) - 1) * self.divmult
        segs[:,1][maskout] = 0
        # Now deal with subparts of sky, buildings, people
        for i, object_index in enumerate(self.objects_with_parts):
            trans = self.part_index[i].to(segs.device)
            # Get the argmax, and then translate to shared numbering scheme
            seg = trans[torch.max(part_pred[i], dim=1)[1]]
            # Only trust the parts where the prediction also predicts the
            # owning object.
            mask = (segs[:,0] == object_index)
            segs[:,2][mask] = seg[mask]

        if self.segdiv == 'quad':
            segs = self.expand_segment_quad(segs, self.segdiv)
        return segs

    def predict_single_class(self, tensor_images, classnum, downsample=1):
        '''
        Given a batch of images (RGB, normalized to [-1...1]) and
        a specific segmentation class number, returns a tuple with
           (1) a differentiable ([0..1]) prediction score for the class
               at every pixel of the input image.
           (2) a binary mask showing where in the input image the
               specified class is the best-predicted label for the pixel.
        Does not work on subdivided labels.
        '''
        result = 0
        pred, part_pred = self.raw_seg_prediction(tensor_images,
                downsample=downsample)
        material_offset = (len(self.segmodel.labeldata['object']) - 1
            ) * self.divmult
        if material_offset < classnum < material_offset + len(
                self.segmodel.labeldata['material']):
            return (
                pred['material'][:, classnum - material_offset],
                pred['material'].max(dim=1)[1] == classnum - material_offset)
        mask = None
        if classnum < len(self.segmodel.labeldata['object']):
            result = pred['object'][:, classnum]
            mask = (pred['object'].max(dim=1)[1] == classnum)
        # Some objects, like 'door', are also a part of other objects,
        # so add the part prediction also.
        for i, object_index in enumerate(self.objects_with_parts):
            local_index = (self.part_index[i] == classnum).nonzero()
            if len(local_index) == 0:
                continue
            local_index = local_index.item()
            # Ignore part predictions outside the mask. (We could pay
            # atttention to and penalize such predictions.)
            mask2 = (pred['object'].max(dim=1)[1] == object_index) * (
                    part_pred[i].max(dim=1)[1] == local_index)
            if mask is None:
                mask = mask2
            else:
                mask = torch.max(mask, mask2)
            result = result + (part_pred[i][:, local_index])
        assert result is not 0, 'unrecognized class %d' % classnum
        return result, mask

    def expand_segment_quad(self, segs, segdiv='quad'):
        shape = segs.shape
        segs[:,3:] = segs[:,0:1] # start by copying the object channel
        num_seg_labels = self.num_object_classes
        # For every connected component present (using generator)
        for i, mask in component_masks(segs[:,0:1]):
            # Figure the bounding box of the label
            top, bottom = mask.any(dim=1).nonzero()[[0, -1], 0]
            left, right = mask.any(dim=0).nonzero()[[0, -1], 0]
            # Chop the bounding box into four parts
            vmid = (top + bottom + 1) // 2
            hmid = (left + right + 1) // 2
            # Construct top, bottom, right, left masks
            quad_mask = mask[None,:,:].repeat(4, 1, 1)
            quad_mask[0, vmid:, :] = 0   # top
            quad_mask[1, :, hmid:] = 0   # right
            quad_mask[2, :vmid, :] = 0   # bottom
            quad_mask[3, :, :hmid] = 0   # left
            quad_mask = quad_mask.long()
            # Modify extra segmentation labels by offsetting
            segs[i,3,:,:] += quad_mask[0] * num_seg_labels
            segs[i,4,:,:] += quad_mask[1] * (2 * num_seg_labels)
            segs[i,3,:,:] += quad_mask[2] * (3 * num_seg_labels)
            segs[i,4,:,:] += quad_mask[3] * (4 * num_seg_labels)
        # remove any components that were too small to subdivide
        mask = segs[:,3:] <= self.num_object_classes
        segs[:,3:][mask] = 0
        return segs

class SemanticSegmenter(BaseSegmenter):
    def __init__(self, modeldir=None, segarch=None, segvocab=None,
            segsizes=None, segdiv=None, epoch=None):
        # Create a segmentation model
        if modeldir == None:
            modeldir = 'dataset/segmodel'
        if segvocab == None:
            segvocab = 'baseline'
        if segarch == None:
            segarch = ('resnet50_dilated8', 'ppm_bilinear_deepsup')
        if segdiv == None:
            segdiv = 'undivided'
        elif isinstance(segarch, str):
            segarch = segarch.split(',')
        segmodel = load_segmentation_model(modeldir, segarch, segvocab, epoch)
        if segsizes is None:
            segsizes = getattr(segmodel.meta, 'segsizes', [256])
        self.segsizes = segsizes
        # Verify segmentation model to has every out_channel labeled.
        assert len(segmodel.meta.labels) == list(c for c in segmodel.modules()
            if isinstance(c, torch.nn.Conv2d))[-1].out_channels
        segmodel.cuda()
        self.segmodel = segmodel
        self.segdiv = segdiv
        # Image normalization
        self.bgr = (segmodel.meta.imageformat.byteorder == 'BGR')
        self.imagemean = torch.tensor(segmodel.meta.imageformat.mean)
        self.imagestd = torch.tensor(segmodel.meta.imageformat.stdev)
        # Map from labels to external indexes, and labels to channel sets.
        self.labelmap = {'-': 0}
        self.channelmap = {'-': []}
        self.labels = [('-', '-')]
        num_labels = 1
        self.num_underlying_classes = len(segmodel.meta.labels)
        # labelmap maps names to external indexes.
        for i, label in enumerate(segmodel.meta.labels):
            if label.name not in self.channelmap:
                self.channelmap[label.name] = []
            self.channelmap[label.name].append(i)
            if getattr(label, 'internal', None) or label.name in self.labelmap:
                continue
            self.labelmap[label.name] = num_labels
            num_labels += 1
            self.labels.append((label.name, label.category))
        # Each category gets its own independent softmax.
        self.category_indexes = { category.name:
                [i for i, label in enumerate(segmodel.meta.labels)
                   if label.category == category.name] 
                for category in segmodel.meta.categories }
        # catindexmap maps names to category internal indexes
        self.catindexmap = {}
        for catname, indexlist in self.category_indexes.items():
            for index, i in enumerate(indexlist):
                self.catindexmap[segmodel.meta.labels[i].name] = (
                        (catname, index))
        # After the softmax, each category is mapped to external indexes.
        self.category_map = { catname:
                torch.tensor([
                    self.labelmap.get(segmodel.meta.labels[ind].name, 0)
                    for ind in catindex])
                for catname, catindex in self.category_indexes.items()}
        self.category_rules = segmodel.meta.categories
        # Finally, naive subdivision can be applied.
        mult = 1
        if self.segdiv == 'quad':
            mult = 5
            suffixes = ['t', 'l', 'b', 'r']
            divided_labels = []
            for suffix in suffixes:
                divided_labels.extend([('%s-%s' % (label, suffix), cat)
                    for label, cat in self.labels[1:]])
                self.channelmap.update({
                    '%s-%s' % (label, suffix): self.channelmap[label]
                    for label, cat in self.labels[1:] })
            self.labels.extend(divided_labels)
        # For examining a single class
        self.channellist = [self.channelmap[name] for name, _ in self.labels]

    def get_label_and_category_names(self, dataset=None):
        return self.labels, self.segmodel.categories

    def segment_batch(self, tensor_images, downsample=1):
        return self.raw_segment_batch(tensor_images, downsample)[0]

    def raw_segment_batch(self, tensor_images, downsample=1):
        pred = self.raw_seg_prediction(tensor_images, downsample)
        catsegs = {}
        for catkey, catindex in self.category_indexes.items():
            _, segs = torch.max(pred[:, catindex], dim=1)
            catsegs[catkey] = segs
        masks = {}
        segs = torch.zeros(len(tensor_images), len(self.category_rules),
                pred.shape[2], pred.shape[2], device=pred.device,
                dtype=torch.long)
        for i, cat in enumerate(self.category_rules):
            catmap = self.category_map[cat.name].to(pred.device)
            translated = catmap[catsegs[cat.name]]
            if getattr(cat, 'mask', None) is not None:
                if cat.mask not in masks:
                    maskcat, maskind = self.catindexmap[cat.mask]
                    masks[cat.mask] = (catsegs[maskcat] == maskind)
                translated *= masks[cat.mask].long()
            segs[:,i] = translated
        if self.segdiv == 'quad':
            segs = self.expand_segment_quad(segs,
                    self.num_underlying_classes, self.segdiv)
        return segs, pred

    def raw_seg_prediction(self, tensor_images, downsample=1):
        '''
        Generates a segmentation by applying multiresolution voting on
        the segmentation model, using (rounded to 32 pixels) a set of
        resolutions in the example benchmark code.
        '''
        y, x = tensor_images.shape[2:]
        b = len(tensor_images)
        # Flip the RGB order if specified.
        if self.bgr:
           tensor_images = torch.flip(tensor_images, (1,))
        # Transform from our [-1..1] range to torch standard [0..1] range
        # and then apply normalization.
        tensor_images = ((tensor_images + 1) / 2
                ).sub_(self.imagemean[None,:,None,None].to(tensor_images.device)
                ).div_(self.imagestd[None,:,None,None].to(tensor_images.device))
        # Output shape can be downsampled.
        seg_shape = (y // downsample, x // downsample)
        # We want these to be multiples of 32 for the model.
        sizes = [(s, s) for s in self.segsizes]
        pred = torch.zeros(
            len(tensor_images), (self.num_underlying_classes),
            seg_shape[0], seg_shape[1]).cuda()
        for size in sizes:
            if size == tensor_images.shape[2:]:
                resized = tensor_images
            else:
                resized = torch.nn.AdaptiveAvgPool2d(size)(tensor_images)
            raw_pred = self.segmodel(
                dict(img_data=resized), segSize=seg_shape)
            softmax_pred = torch.empty_like(raw_pred)
            for catindex in self.category_indexes.values():
                softmax_pred[:, catindex] = torch.nn.functional.softmax(
                        raw_pred[:, catindex], dim=1)
            pred += softmax_pred
        return pred

    def expand_segment_quad(self, segs, num_seg_labels, segdiv='quad'):
        shape = segs.shape
        output = segs.repeat(1, 3, 1, 1)
        # For every connected component present (using generator)
        for i, mask in component_masks(segs):
            # Figure the bounding box of the label
            top, bottom = mask.any(dim=1).nonzero()[[0, -1], 0]
            left, right = mask.any(dim=0).nonzero()[[0, -1], 0]
            # Chop the bounding box into four parts
            vmid = (top + bottom + 1) // 2
            hmid = (left + right + 1) // 2
            # Construct top, bottom, right, left masks
            quad_mask = mask[None,:,:].repeat(4, 1, 1)
            quad_mask[0, vmid:, :] = 0   # top
            quad_mask[1, :, hmid:] = 0   # right
            quad_mask[2, :vmid, :] = 0   # bottom
            quad_mask[3, :, :hmid] = 0   # left
            quad_mask = quad_mask.long()
            # Modify extra segmentation labels by offsetting
            output[i,1,:,:] += quad_mask[0] * num_seg_labels
            output[i,2,:,:] += quad_mask[1] * (2 * num_seg_labels)
            output[i,1,:,:] += quad_mask[2] * (3 * num_seg_labels)
            output[i,2,:,:] += quad_mask[3] * (4 * num_seg_labels)
        return output

    def predict_single_class(self, tensor_images, classnum, downsample=1):
        '''
        Given a batch of images (RGB, normalized to [-1...1]) and
        a specific segmentation class number, returns a tuple with
           (1) a differentiable ([0..1]) prediction score for the class
               at every pixel of the input image.
           (2) a binary mask showing where in the input image the
               specified class is the best-predicted label for the pixel.
        Does not work on subdivided labels.
        '''
        seg, pred = self.raw_segment_batch(tensor_images,
                downsample=downsample)
        result = pred[:,self.channellist[classnum]].sum(dim=1)
        mask = (seg == classnum).max(1)[0]
        return result, mask

def component_masks(segmentation_batch):
    '''
    Splits connected components into regions (slower, requires cpu).
    '''
    npbatch = segmentation_batch.cpu().numpy()
    for i in range(segmentation_batch.shape[0]):
        labeled, num = skimage.morphology.label(npbatch[i][0], return_num=True)
        labeled = torch.from_numpy(labeled).to(segmentation_batch.device)
        for label in range(1, num):
            yield i, (labeled == label)

def load_unified_parsing_segmentation_model(segmodel_arch, segvocab, epoch):
    segmodel_dir = 'dataset/segmodel/%s-%s-%s' % ((segvocab,) + segmodel_arch)
    # Load json of class names and part/object structure
    with open(os.path.join(segmodel_dir, 'labels.json')) as f:
        labeldata = json.load(f)
    nr_classes={k: len(labeldata[k])
                for k in ['object', 'scene', 'material']}
    nr_classes['part'] = sum(len(p) for p in labeldata['object_part'].values())
    # Create a segmentation model
    segbuilder = upsegmodel.ModelBuilder()
    # example segmodel_arch = ('resnet101', 'upernet')
    seg_encoder = segbuilder.build_encoder(
            arch=segmodel_arch[0],
            fc_dim=2048,
            weights=os.path.join(segmodel_dir, 'encoder_epoch_%d.pth' % epoch))
    seg_decoder = segbuilder.build_decoder(
            arch=segmodel_arch[1],
            fc_dim=2048, use_softmax=True,
            nr_classes=nr_classes,
            weights=os.path.join(segmodel_dir, 'decoder_epoch_%d.pth' % epoch))
    segmodel = upsegmodel.SegmentationModule(
            seg_encoder, seg_decoder, labeldata)
    segmodel.categories = ['object', 'part', 'material']
    segmodel.eval()
    return segmodel

def load_segmentation_model(modeldir, segmodel_arch, segvocab, epoch=None):
    # Load csv of class names
    segmodel_dir = 'dataset/segmodel/%s-%s-%s' % ((segvocab,) + segmodel_arch)
    with open(os.path.join(segmodel_dir, 'labels.json')) as f:
        labeldata = EasyDict(json.load(f))
    # Automatically pick the last epoch available.
    if epoch is None:
        choices = [os.path.basename(n)[14:-4] for n in
                glob.glob(os.path.join(segmodel_dir, 'encoder_epoch_*.pth'))]
        epoch = max([int(c) for c in choices if c.isdigit()])
    # Create a segmentation model
    segbuilder = segmodel_module.ModelBuilder()
    # example segmodel_arch = ('resnet101', 'upernet')
    seg_encoder = segbuilder.build_encoder(
            arch=segmodel_arch[0],
            fc_dim=2048,
            weights=os.path.join(segmodel_dir, 'encoder_epoch_%d.pth' % epoch))
    seg_decoder = segbuilder.build_decoder(
            arch=segmodel_arch[1],
            fc_dim=2048, inference=True, num_class=len(labeldata.labels),
            weights=os.path.join(segmodel_dir, 'decoder_epoch_%d.pth' % epoch))
    segmodel = segmodel_module.SegmentationModule(seg_encoder, seg_decoder,
                                  torch.nn.NLLLoss(ignore_index=-1))
    segmodel.categories = [cat.name for cat in labeldata.categories]
    segmodel.labels = [label.name for label in labeldata.labels]
    categories = OrderedDict()
    label_category = numpy.zeros(len(segmodel.labels), dtype=int)
    for i, label in enumerate(labeldata.labels):
        label_category[i] = segmodel.categories.index(label.category)
    segmodel.meta = labeldata
    segmodel.eval()
    return segmodel

def ensure_upp_segmenter_downloaded(directory):
    baseurl = 'http://netdissect.csail.mit.edu/data/segmodel'
    dirname = 'upp-resnet50-upernet'
    files = ['decoder_epoch_40.pth', 'encoder_epoch_40.pth', 'labels.json']
    download_dir = os.path.join(directory, dirname)
    os.makedirs(download_dir, exist_ok=True)
    for fn in files:
        if os.path.isfile(os.path.join(download_dir, fn)):
            continue # Skip files already downloaded
        url = '%s/%s/%s' % (baseurl, dirname, fn)
        print('Downloading %s' % url)
        urlretrieve(url, os.path.join(download_dir, fn))
    assert os.path.isfile(os.path.join(directory, dirname, 'labels.json'))

def test_main():
    '''
    Test the unified segmenter.
    '''
    from PIL import Image
    testim = Image.open('script/testdata/test_church_242.jpg')
    tensor_im = (torch.from_numpy(numpy.asarray(testim)).permute(2, 0, 1)
            .float() / 255 * 2 - 1)[None, :, :, :].cuda()
    segmenter = UnifiedParsingSegmenter()
    seg = segmenter.segment_batch(tensor_im)
    bc = torch.bincount(seg.view(-1))
    labels, cats = segmenter.get_label_and_category_names()
    for label in bc.nonzero()[:,0]:
        if label.item():
            # What is the prediction for this class?
            pred, mask = segmenter.predict_single_class(tensor_im, label.item())
            assert mask.sum().item() == bc[label].item()
            assert len(((seg == label).max(1)[0] - mask).nonzero()) == 0
            inside_pred = pred[mask].mean().item()
            outside_pred = pred[~mask].mean().item()
            print('%s (%s, #%d): %d pixels, pred %.2g inside %.2g outside' %
                (labels[label.item()] + (label.item(), bc[label].item(),
                    inside_pred, outside_pred)))

if __name__ == '__main__':
    test_main()
