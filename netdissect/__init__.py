'''
Netdissect package.

To run dissection:

1. Load up the convolutional model you wish to dissect, and wrap it
   in an InstrumentedModel.  Call imodel.retain_layers([layernames,..])
   to analyze a specified set of layers.
2. Load the segmentation dataset using the BrodenDataset class;
   use the transform_image argument to normalize images to be
   suitable for the model, or the size argument to truncate the dataset.
3. Write a function to recover the original image (with RGB scaled to
   [0...1]) given a normalized dataset image; ReverseNormalize in this
   package inverts transforms.Normalize for this purpose.
4. Choose a directory in which to write the output, and call
   dissect(outdir, model, dataset).

Example:

    from netdissect import InstrumentedModel, dissect
    from netdissect import BrodenDataset, ReverseNormalize

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
            recover_image=ReverseNormalize(IMAGE_MEAN, IMAGE_STDEV),
            examples_per_unit=10)
'''

from .dissection import dissect, ReverseNormalize
from .dissection import ClassifierSegRunner, GeneratorSegRunner
from .dissection import ImageOnlySegRunner
from .broden import BrodenDataset, ScaleSegmentation, scatter_batch
from .segdata import MultiSegmentDataset
from .nethook import InstrumentedModel
from .zdataset import z_dataset_for_model, z_sample_for_model, standard_z_sample
from . import actviz
from . import progress
from . import runningstats
from . import sampler

__all__ = [
    'dissect', 'ReverseNormalize',
    'ClassifierSegRunner', 'GeneratorSegRunner', 'ImageOnlySegRunner',
    'BrodenDataset', 'ScaleSegmentation', 'scatter_batch',
    'MultiSegmentDataset',
    'InstrumentedModel',
    'z_dataset_for_model', 'z_sample_for_model', 'standard_z_sample'
    'actviz',
    'progress',
    'runningstats',
    'sampler'
]
