'''
Utilities for instrumenting a torch model.

InstrumentedModel will wrap a pytorch model and allow hooking
arbitrary layers to monitor or modify their output directly.

Modified by Erik Härkönen:
- 29.11.2019: Unhooking bugfix
- 25.01.2020: Offset edits, removed old API
'''

import torch, numpy, types
from collections import OrderedDict

class InstrumentedModel(torch.nn.Module):
    '''
    A wrapper for hooking, probing and intervening in pytorch Modules.
    Example usage:

    ```
    model = load_my_model()
    with inst as InstrumentedModel(model):
        inst.retain_layer(layername)
        inst.edit_layer(layername, 0.5, target_features)
        inst.edit_layer(layername, offset=offset_tensor)
        inst(inputs)
        original_features = inst.retained_layer(layername)
    ```
    '''

    def __init__(self, model):
        super(InstrumentedModel, self).__init__()
        self.model = model
        self._retained = OrderedDict()
        self._ablation = {}
        self._replacement = {}
        self._offset = {}
        self._hooked_layer = {}
        self._old_forward = {}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def retain_layer(self, layername):
        '''
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and retain its output each time the model is run.
        A pair (layername, aka) can be provided, and the aka will be used
        as the key for the retained value instead of the layername.
        '''
        self.retain_layers([layername])

    def retain_layers(self, layernames):
        '''
        Retains a list of a layers at once.
        '''
        self.add_hooks(layernames)
        for layername in layernames:
            aka = layername
            if not isinstance(aka, str):
                layername, aka = layername
            if aka not in self._retained:
                self._retained[aka] = None

    def retained_features(self):
        '''
        Returns a dict of all currently retained features.
        '''
        return OrderedDict(self._retained)

    def retained_layer(self, aka=None, clear=False):
        '''
        Retrieve retained data that was previously hooked by retain_layer.
        Call this after the model is run.  If clear is set, then the
        retained value will return and also cleared.
        '''
        if aka is None:
            # Default to the first retained layer.
            aka = next(self._retained.keys().__iter__())
        result = self._retained[aka]
        if clear:
            self._retained[aka] = None
        return result

    def edit_layer(self, layername, ablation=None, replacement=None, offset=None):
        '''
        Pass a fully-qualified layer name (E.g., module.submodule.conv3)
        to hook that layer and modify its output each time the model is run.
        The output of the layer will be modified to be a convex combination
        of the replacement and x interpolated according to the ablation, i.e.:
        `output = x * (1 - a) + (r * a)`.
        Additionally or independently, an offset can be added to the output.
        '''
        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername

        # The default ablation if a replacement is specified is 1.0.
        if ablation is None and replacement is not None:
            ablation = 1.0
        self.add_hooks([(layername, aka)])
        if ablation is not None:
            self._ablation[aka] = ablation
        if replacement is not None:
            self._replacement[aka] = replacement
        if offset is not None:
            self._offset[aka] = offset
        # If needed, could add an arbitrary postprocessing lambda here.

    def remove_edits(self, layername=None, remove_offset=True, remove_replacement=True):
        '''
        Removes edits at the specified layer, or removes edits at all layers
        if no layer name is specified.
        '''
        if layername is None:
            if remove_replacement:
                self._ablation.clear()
                self._replacement.clear()
            if remove_offset:
                self._offset.clear()
            return

        if not isinstance(layername, str):
            layername, aka = layername
        else:
            aka = layername
        if remove_replacement and aka in self._ablation:
            del self._ablation[aka]
        if remove_replacement and aka in self._replacement:
            del self._replacement[aka]
        if remove_offset and aka in self._offset:
            del self._offset[aka]

    def add_hooks(self, layernames):
        '''
        Sets up a set of layers to be hooked.

        Usually not called directly: use edit_layer or retain_layer instead.
        '''
        needed = set()
        aka_map = {}
        for name in layernames:
            aka = name
            if not isinstance(aka, str):
                name, aka = name
            if self._hooked_layer.get(aka, None) != name:
                aka_map[name] = aka
                needed.add(name)
        if not needed:
            return
        for name, layer in self.model.named_modules():
            if name in aka_map:
                needed.remove(name)
                aka = aka_map[name]
                self._hook_layer(layer, name, aka)
        for name in needed:
            raise ValueError('Layer %s not found in model' % name)

    def _hook_layer(self, layer, layername, aka):
        '''
        Internal method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        '''
        if aka in self._hooked_layer:
            raise ValueError('Layer %s already hooked' % aka)
        if layername in self._old_forward:
            raise ValueError('Layer %s already hooked' % layername)
        self._hooked_layer[aka] = layername
        self._old_forward[layername] = (layer, aka,
                layer.__dict__.get('forward', None))
        editor = self
        original_forward = layer.forward
        def new_forward(self, *inputs, **kwargs):
            original_x = original_forward(*inputs, **kwargs)
            x = editor._postprocess_forward(original_x, aka)
            return x
        layer.forward = types.MethodType(new_forward, layer)

    def _unhook_layer(self, aka):
        '''
        Internal method to remove a hook, restoring the original forward method.
        '''
        if aka not in self._hooked_layer:
            return
        layername = self._hooked_layer[aka]
        layer, check, old_forward = self._old_forward[layername]
        assert check == aka
        if old_forward is None:
            if 'forward' in layer.__dict__:
                del layer.__dict__['forward']
        else:
            layer.forward = old_forward
        del self._old_forward[layername]
        del self._hooked_layer[aka]
        if aka in self._ablation:
            del self._ablation[aka]
        if aka in self._replacement:
            del self._replacement[aka]
        if aka in self._offset:
            del self._offset[aka]
        if aka in self._retained:
            del self._retained[aka]

    def _postprocess_forward(self, x, aka):
        '''
        The internal method called by the hooked layers after they are run.
        '''
        # Retain output before edits, if desired.
        if aka in self._retained:
            self._retained[aka] = x.detach()
        
        # Apply replacement edit
        a = make_matching_tensor(self._ablation, aka, x)
        if a is not None:
            x = x * (1 - a)
            v = make_matching_tensor(self._replacement, aka, x)
            if v is not None:
                x += (v * a)
        
        # Apply offset edit
        b = make_matching_tensor(self._offset, aka, x)
        if b is not None:
            x = x + b
        
        return x

    def close(self):
        '''
        Unhooks all hooked layers in the model.
        '''
        for aka in list(self._old_forward.keys()):
            self._unhook_layer(aka)
        assert len(self._old_forward) == 0


def make_matching_tensor(valuedict, name, data):
    '''
    Converts `valuedict[name]` to be a tensor with the same dtype, device,
    and dimension count as `data`, and caches the converted tensor.
    '''
    v = valuedict.get(name, None)
    if v is None:
        return None
    if not isinstance(v, torch.Tensor):
        # Accept non-torch data.
        v = torch.from_numpy(numpy.array(v))
        valuedict[name] = v
    if not v.device == data.device or not v.dtype == data.dtype:
        # Ensure device and type matches.
        assert not v.requires_grad, '%s wrong device or type' % (name)
        v = v.to(device=data.device, dtype=data.dtype)
        valuedict[name] = v
    if len(v.shape) < len(data.shape):
        # Ensure dimensions are unsqueezed as needed.
        assert not v.requires_grad, '%s wrong dimensions' % (name)
        v = v.view((1,) + tuple(v.shape) +
                (1,) * (len(data.shape) - len(v.shape) - 1))
        valuedict[name] = v
    return v
