'''
Utilities for dealing with simple state dicts as npz files instead of pth files.
'''

import torch
from collections.abc import MutableMapping, Mapping

def load_from_numpy_dict(model, numpy_dict, prefix='', examples=None):
    '''
    Loads a model from numpy_dict using load_state_dict.
    Converts numpy types to torch types using the current state_dict
    of the model to determine types and devices for the tensors.
    Supports loading a subdict by prepending the given prefix to all keys.
    '''
    if prefix:
        if not prefix.endswith('.'):
            prefix = prefix + '.'
        numpy_dict = PrefixSubDict(numpy_dict, prefix)
    if examples is None:
        exampels = model.state_dict()
    torch_state_dict = TorchTypeMatchingDict(numpy_dict, examples)
    model.load_state_dict(torch_state_dict)

def save_to_numpy_dict(model, numpy_dict, prefix=''):
    '''
    Saves a model by copying tensors to numpy_dict.
    Converts torch types to numpy types using `t.detach().cpu().numpy()`.
    Supports saving a subdict by prepending the given prefix to all keys.
    '''
    if prefix:
        if not prefix.endswith('.'):
            prefix = prefix + '.'
    for k, v in model.numpy_dict().items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        numpy_dict[prefix + k] = v

class TorchTypeMatchingDict(Mapping):
    '''
    Provides a view of a dict of numpy values as torch tensors, where the
    types are converted to match the types and devices in the given
    dict of examples.
    '''
    def __init__(self, data, examples):
        self.data = data
        self.examples = examples
        self.cached_data = {}
    def __getitem__(self, key):
        if key in self.cached_data:
            return self.cached_data[key]
        val = self.data[key]
        if key not in self.examples:
            return val
        example = self.examples.get(key, None)
        example_type = type(example)
        if example is not None and type(val) != example_type:
            if isinstance(example, torch.Tensor):
                val = torch.from_numpy(val)
            else:
                val = example_type(val)
        if isinstance(example, torch.Tensor):
            val = val.to(dtype=example.dtype, device=example.device)
        self.cached_data[key] = val
        return val
    def __iter__(self):
        return self.data.keys()
    def __len__(self):
        return len(self.data)

class PrefixSubDict(MutableMapping):
    '''
    Provides a view of the subset of a dict where string keys begin with
    the given prefix.  The prefix is stripped from all keys of the view.
    '''
    def __init__(self, data, prefix=''):
        self.data = data
        self.prefix = prefix
        self._cached_keys = None
    def __getitem__(self, key):
        return self.data[self.prefix + key]
    def __setitem__(self, key, value):
        pkey = self.prefix + key
        if self._cached_keys is not None and pkey not in self.data:
            self._cached_keys = None
        self.data[pkey] = value
    def __delitem__(self, key):
        pkey = self.prefix + key
        if self._cached_keys is not None and pkey in self.data:
            self._cached_keys = None
        del self.data[pkey]
    def __cached_keys(self):
        if self._cached_keys is None:
            plen = len(self.prefix)
            self._cached_keys = list(k[plen:] for k in self.data
                    if k.startswith(self.prefix))
        return self._cached_keys
    def __iter__(self):
        return iter(self.__cached_keys())
    def __len__(self):
        return len(self.__cached_keys())
