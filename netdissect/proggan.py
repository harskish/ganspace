import torch, numpy, itertools
import torch.nn as nn
from collections import OrderedDict


def print_network(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if verbose:
        print(net)
    print('Total number of parameters: {:3.3f} M'.format(num_params / 1e6))


def from_pth_file(filename):
    '''
    Instantiate from a pth file.
    '''
    state_dict = torch.load(filename)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # Convert old version of parameter names
    if 'features.0.conv.weight' in state_dict:
        state_dict = state_dict_from_old_pt_dict(state_dict)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result

###############################################################################
# Modules
###############################################################################

class ProgressiveGenerator(nn.Sequential):
    def __init__(self, resolution=None, sizes=None, modify_sequence=None,
            output_tanh=False):
        '''
        A pytorch progessive GAN generator that can be converted directly
        from either a tensorflow model or a theano model.  It consists of
        a sequence of convolutional layers, organized in pairs, with an
        upsampling and reduction of channels at every other layer; and
        then finally followed by an output layer that reduces it to an
        RGB [-1..1] image.

        The network can be given more layers to increase the output
        resolution.  The sizes argument indicates the fieature depth at
        each upsampling, starting with the input z: [input-dim, 4x4-depth,
        8x8-depth, 16x16-depth...].  The output dimension is 2 * 2**len(sizes)

        Some default architectures can be selected by supplying the
        resolution argument instead.

        The optional modify_sequence function can be used to transform the
        sequence of layers before the network is constructed.

        If output_tanh is set to True, the network applies a tanh to clamp
        the output to [-1,1] before output; otherwise the output is unclamped.
        '''
        assert (resolution is None) != (sizes is None)
        if sizes is None:
            sizes = {
                    8: [512, 512, 512],
                    16: [512, 512, 512, 512],
                    32: [512, 512, 512, 512, 256],
                    64: [512, 512, 512, 512, 256, 128],
                    128: [512, 512, 512, 512, 256, 128, 64],
                    256: [512, 512, 512, 512, 256, 128, 64, 32],
                    1024: [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
                }[resolution]
        # Follow the schedule of upsampling given by sizes.
        # layers are called: layer1, layer2, etc; then output_128x128
        sequence = []
        def add_d(layer, name=None):
            if name is None:
                name = 'layer%d' % (len(sequence) + 1)
            sequence.append((name, layer))
        add_d(NormConvBlock(sizes[0], sizes[1], kernel_size=4, padding=3))
        add_d(NormConvBlock(sizes[1], sizes[1], kernel_size=3, padding=1))
        for i, (si, so) in enumerate(zip(sizes[1:-1], sizes[2:])):
            add_d(NormUpscaleConvBlock(si, so, kernel_size=3, padding=1))
            add_d(NormConvBlock(so, so, kernel_size=3, padding=1))
        # Create an output layer.  During training, the progressive GAN
        # learns several such output layers for various resolutions; we
        # just include the last (highest resolution) one.
        dim = 4 * (2 ** (len(sequence) // 2 - 1))
        add_d(OutputConvBlock(sizes[-1], tanh=output_tanh),
                name='output_%dx%d' % (dim, dim))
        # Allow the sequence to be modified
        if modify_sequence is not None:
            sequence = modify_sequence(sequence)
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        # Convert vector input to 1x1 featuremap.
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return super().forward(x)

class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class DoubleResolutionLayer(nn.Module):
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x

class WScaleLayer(nn.Module):
    def __init__(self, size, fan_in, gain=numpy.sqrt(2)):
        super(WScaleLayer, self).__init__()
        self.scale = gain / numpy.sqrt(fan_in) # No longer a parameter
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])
        return x

class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels,
                gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x

class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = DoubleResolutionLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels,
                gain=numpy.sqrt(2) / kernel_size)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = self.relu(self.wscale(x))
        return x

class OutputConvBlock(nn.Module):
    def __init__(self, in_channels, tanh=False):
        super().__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
                in_channels, 3, kernel_size=1, padding=0, bias=False)
        self.wscale = WScaleLayer(3, in_channels, gain=1)
        self.clamp = nn.Hardtanh() if tanh else (lambda x: x)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.wscale(x)
        x = self.clamp(x)
        return x

###############################################################################
# Conversion
###############################################################################

def from_tf_parameters(parameters):
    '''
    Instantiate from tensorflow variables.
    '''
    state_dict = state_dict_from_tf_parameters(parameters)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result

def from_old_pt_dict(parameters):
    '''
    Instantiate from old pytorch state dict.
    '''
    state_dict = state_dict_from_old_pt_dict(parameters)
    sizes = sizes_from_state_dict(state_dict)
    result = ProgressiveGenerator(sizes=sizes)
    result.load_state_dict(state_dict)
    return result

def sizes_from_state_dict(params):
    '''
    In a progressive GAN, the number of channels can change after each
    upsampling.  This function reads the state dict to figure the
    number of upsamplings and the channel depth of each filter.
    '''
    sizes = []
    for i in itertools.count():
        pt_layername = 'layer%d' % (i + 1)
        try:
            weight = params['%s.conv.weight' % pt_layername]
        except KeyError:
            break
        if i == 0:
            sizes.append(weight.shape[1])
        if i % 2 == 0:
            sizes.append(weight.shape[0])
    return sizes

def state_dict_from_tf_parameters(parameters):
    '''
    Conversion from tensorflow parameters
    '''
    def torch_from_tf(data):
        return torch.from_numpy(data.eval())

    params = dict(parameters)
    result = {}
    sizes = []
    for i in itertools.count():
        resolution = 4 * (2 ** (i // 2))
        # Translate parameter names.  For example:
        # 4x4/Dense/weight -> layer1.conv.weight
        # 32x32/Conv0_up/weight -> layer7.conv.weight
        # 32x32/Conv1/weight -> layer8.conv.weight
        tf_layername = '%dx%d/%s' % (resolution, resolution,
                'Dense' if i == 0 else 'Conv' if i == 1 else
                'Conv0_up' if i % 2 == 0 else 'Conv1')
        pt_layername = 'layer%d' % (i + 1)
        # Stop looping when we run out of parameters.
        try:
            weight = torch_from_tf(params['%s/weight' % tf_layername])
        except KeyError:
            break
        # Transpose convolution weights into pytorch format.
        if i == 0:
            # Convert dense layer to 4x4 convolution
            weight = weight.view(weight.shape[0], weight.shape[1] // 16,
                   4, 4).permute(1, 0, 2, 3).flip(2, 3)
            sizes.append(weight.shape[0])
        elif i % 2 == 0:
            # Convert inverse convolution to convolution
            weight = weight.permute(2, 3, 0, 1).flip(2, 3)
        else:
            # Ordinary Conv2d conversion.
            weight = weight.permute(3, 2, 0, 1)
            sizes.append(weight.shape[1])
        result['%s.conv.weight' % (pt_layername)] = weight
        # Copy bias vector.
        bias = torch_from_tf(params['%s/bias' % tf_layername])
        result['%s.wscale.b' % (pt_layername)] = bias
    # Copy just finest-grained ToRGB output layers.  For example:
    # ToRGB_lod0/weight -> output.conv.weight
    i -= 1
    resolution = 4 * (2 ** (i // 2))
    tf_layername = 'ToRGB_lod0'
    pt_layername = 'output_%dx%d' % (resolution, resolution)
    result['%s.conv.weight' % pt_layername] = torch_from_tf(
            params['%s/weight' % tf_layername]).permute(3, 2, 0, 1)
    result['%s.wscale.b' % pt_layername] = torch_from_tf(
            params['%s/bias' % tf_layername])
    # Return parameters
    return result

def state_dict_from_old_pt_dict(params):
    '''
    Conversion from the old pytorch model layer names.
    '''
    result = {}
    sizes = []
    for i in itertools.count():
        old_layername = 'features.%d' % i
        pt_layername = 'layer%d' % (i + 1)
        try:
            weight = params['%s.conv.weight' % (old_layername)]
        except KeyError:
            break
        if i == 0:
            sizes.append(weight.shape[0])
        if i % 2 == 0:
            sizes.append(weight.shape[1])
        result['%s.conv.weight' % (pt_layername)] = weight
        result['%s.wscale.b' % (pt_layername)] = params[
                '%s.wscale.b' % (old_layername)]
    # Copy the output layers.
    i -= 1
    resolution = 4 * (2 ** (i // 2))
    pt_layername = 'output_%dx%d' % (resolution, resolution)
    result['%s.conv.weight' % pt_layername] = params['output.conv.weight']
    result['%s.wscale.b' % pt_layername] = params['output.wscale.b']
    # Return parameters and also network architecture sizes.
    return result

