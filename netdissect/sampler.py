'''
A sampler is just a list of integer listing the indexes of the
inputs in a data set to sample.  For reproducibility, the
FixedRandomSubsetSampler uses a seeded prng to produce the same
sequence always.  FixedSubsetSampler is just a wrapper for an
explicit list of integers.

coordinate_sample solves another sampling problem: when testing
convolutional outputs, we can reduce data explosing by sampling
random points of the feature map rather than the entire feature map.
coordinate_sample does this in a deterministic way that is also
resolution-independent.
'''

import numpy
import random
from torch.utils.data.sampler import Sampler

class FixedSubsetSampler(Sampler):
    """Represents a fixed sequence of data set indices.
    Subsets can be created by specifying a subset of output indexes.
    """
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def subset(self, new_subset):
        return FixedSubsetSampler(self.dereference(new_subset))

    def dereference(self, indices):
        '''
        Translate output sample indices (small numbers indexing the sample)
        to input sample indices (larger number indexing the original full set)
        '''
        return [self.samples[i] for i in indices]


class FixedRandomSubsetSampler(FixedSubsetSampler):
    """Samples a fixed number of samples from the dataset, deterministically.
    Arguments:
        data_source,
        sample_size,
        seed (optional)
    """
    def __init__(self, data_source, start=None, end=None, seed=1):
        rng = random.Random(seed)
        shuffled = list(range(len(data_source)))
        rng.shuffle(shuffled)
        self.data_source = data_source
        super(FixedRandomSubsetSampler, self).__init__(shuffled[start:end])

    def class_subset(self, class_filter):
        '''
        Returns only the subset matching the given rule.
        '''
        if isinstance(class_filter, int):
            rule = lambda d: d[1] == class_filter
        else:
            rule = class_filter
        return self.subset([i for i, j in enumerate(self.samples)
                if rule(self.data_source[j])])

def coordinate_sample(shape, sample_size, seeds, grid=13, seed=1, flat=False):
    '''
    Returns a (end-start) sets of sample_size grid points within
    the shape given.  If the shape dimensions are a multiple of 'grid',
    then sampled points within the same row will never be duplicated.
    '''
    if flat:
        sampind = numpy.zeros((len(seeds), sample_size), dtype=int)
    else:
        sampind = numpy.zeros((len(seeds), 2, sample_size), dtype=int)
    assert sample_size <= grid
    for j, seed in enumerate(seeds):
        rng = numpy.random.RandomState(seed)
        # Shuffle the 169 random grid squares, and pick :sample_size.
        square_count = grid ** len(shape)
        square = numpy.stack(numpy.unravel_index(
            rng.choice(square_count, square_count)[:sample_size],
            (grid,) * len(shape)))
        # Then add a random offset to each x, y and put in the range [0...1)
        # Notice this selects the same locations regardless of resolution.
        uniform = (square + rng.uniform(size=square.shape)) / grid
        # TODO: support affine scaling so that we can align receptive field
        # centers exactly when sampling neurons in different layers.
        coords = (uniform * numpy.array(shape)[:,None]).astype(int)
        # Now take sample_size without replacement.  We do this in a way
        # such that if sample_size is decreased or increased up to 'grid',
        # the selected points become a subset, not totally different points.
        if flat:
            sampind[j] = numpy.ravel_multi_index(coords, dims=shape)
        else:
            sampind[j] = coords
    return sampind

if __name__ == '__main__':
    from numpy.testing import assert_almost_equal
    # Test that coordinate_sample is deterministic, in-range, and scalable.
    assert_almost_equal(coordinate_sample((26, 26), 10, range(101, 102)),
            [[[14,  0, 12, 11,  8, 13, 11, 20,  7, 20],
              [ 9, 22,  7, 11, 23, 18, 21, 15,  2,  5]]])
    assert_almost_equal(coordinate_sample((13, 13), 10, range(101, 102)),
            [[[ 7,  0,  6,  5,  4,  6,  5, 10,  3, 20 // 2],
              [ 4, 11,  3,  5, 11,  9, 10,  7,  1,  5 // 2]]])
    assert_almost_equal(coordinate_sample((13, 13), 10, range(100, 102),
        flat=True),
            [[  8,  24,  67, 103,  87,  79, 138,  94,  98,  53],
             [ 95,  11,  81,  70,  63,  87,  75, 137,  40, 2+10*13]])
    assert_almost_equal(coordinate_sample((13, 13), 10, range(101, 103),
        flat=True),
            [[ 95,  11,  81,  70,  63,  87,  75, 137,  40, 132],
             [  0,  78, 114, 111,  66,  45,  72,  73,  79, 135]])
    assert_almost_equal(coordinate_sample((26, 26), 10, range(101, 102),
        flat=True),
            [[373,  22, 319, 297, 231, 356, 307, 535, 184, 5+20*26]])
    # Test FixedRandomSubsetSampler
    fss = FixedRandomSubsetSampler(range(10))
    assert len(fss) == 10
    assert_almost_equal(list(fss), [8, 0, 3, 4, 5, 2, 9, 6, 7, 1])
    fss = FixedRandomSubsetSampler(range(10), 3, 8)
    assert len(fss) == 5
    assert_almost_equal(list(fss), [4, 5, 2, 9, 6])
    fss = FixedRandomSubsetSampler([(i, i % 3) for i in range(10)],
            class_filter=1)
    assert len(fss) == 3
    assert_almost_equal(list(fss), [4, 7, 1])
