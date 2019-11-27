import os, numpy, torch, json
from .parallelfolder import ParallelImageFolders
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, normalize

class FieldDef(object):
    def __init__(self, field, index, bitshift, bitmask, labels):
        self.field = field
        self.index = index
        self.bitshift = bitshift
        self.bitmask = bitmask
        self.labels = labels

class MultiSegmentDataset(object):
    '''
    Just like ClevrMulticlassDataset, but the second stream is a one-hot
    segmentation tensor rather than a flat one-hot presence vector.

    MultiSegmentDataset('dataset/clevrseg',
        imgdir='images/train/positive',
        segdir='images/train/segmentation')
    '''
    def __init__(self, directory, transform=None,
            imgdir='img', segdir='seg', val=False, size=None):
        self.segdataset = ParallelImageFolders(
                [os.path.join(directory, imgdir),
                 os.path.join(directory, segdir)],
                transform=transform)
        self.fields = []
        with open(os.path.join(directory, 'labelnames.json'), 'r') as f:
            for defn in json.load(f):
                self.fields.append(FieldDef(
                    defn['field'], defn['index'], defn['bitshift'],
                    defn['bitmask'], defn['label']))
        self.labels = ['-'] # Reserve label 0 to mean "no label"
        self.categories = []
        self.label_category = [0]
        for fieldnum, f in enumerate(self.fields):
            self.categories.append(f.field)
            f.firstchannel = len(self.labels)
            f.channels = len(f.labels) - 1
            for lab in f.labels[1:]:
                self.labels.append(lab)
                self.label_category.append(fieldnum)
        # Reserve 25% of the dataset for validation.
        first_val = int(len(self.segdataset) * 0.75)
        self.val = val
        self.first = first_val if val else 0
        self.length = len(self.segdataset) - first_val if val else first_val
        # Truncate the dataset if requested.
        if size:
            self.length = min(size, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img, segimg = self.segdataset[index + self.first]
        segin = numpy.array(segimg, numpy.uint8, copy=False)
        segout = torch.zeros(len(self.categories),
                segin.shape[0], segin.shape[1], dtype=torch.int64)
        for i, field in enumerate(self.fields):
            fielddata = ((torch.from_numpy(segin[:, :, field.index])
                    >> field.bitshift) & field.bitmask)
            segout[i] = field.firstchannel + fielddata - 1
        bincount = numpy.bincount(segout.flatten(),
                minlength=len(self.labels))
        return img, segout, bincount

if __name__ == '__main__':
    ds = MultiSegmentDataset('dataset/clevrseg')
    print(ds[0])
    import pdb; pdb.set_trace()

