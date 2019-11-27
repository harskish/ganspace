import os
import numpy
from scipy.interpolate import RectBivariateSpline

def activation_visualization(image, data, level, alpha=0.5, source_shape=None,
        crop=False, zoom=None, border=2, negate=False, return_mask=False,
        **kwargs):
    """
    Makes a visualiztion image of activation data overlaid on the image.
    Params:
        image The original image.
        data The single channel feature map.
        alpha The darkening to apply in inactive regions of the image.
        level The threshold of activation levels to highlight.
    """
    if len(image.shape) == 2:
        # Puff up grayscale image to RGB.
        image = image[:,:,None] * numpy.array([[[1, 1, 1]]])
    surface = activation_surface(data, target_shape=image.shape[:2],
            source_shape=source_shape, **kwargs)
    if negate:
        surface = -surface
        level = -level
    if crop:
        # crop to source_shape
        if source_shape is not None:
            ch, cw = ((t - s) // 2 for s, t in zip(
                source_shape, image.shape[:2]))
            image = image[ch:ch+source_shape[0], cw:cw+source_shape[1]]
            surface = surface[ch:ch+source_shape[0], cw:cw+source_shape[1]]
        if crop is True:
            crop = surface.shape
        elif not hasattr(crop, '__len__'):
            crop = (crop, crop)
        if zoom is not None:
            source_rect = best_sub_rect(surface >= level, crop, zoom,
                    pad=border)
        else:
            source_rect = (0, surface.shape[0], 0, surface.shape[1])
        image = zoom_image(image, source_rect, crop)
        surface = zoom_image(surface, source_rect, crop)
    mask = (surface >= level)
    # Add a yellow border at the edge of the mask for contrast
    result = (mask[:, :, None] * (1 - alpha) + alpha) * image
    if border:
        edge = mask_border(mask)[:,:,None]
        result = numpy.maximum(edge * numpy.array([[[200, 200, 0]]]), result)
    if not return_mask:
        return result
    mask_image = (1 - mask[:, :, None]) * numpy.array(
            [[[0, 0, 0, 255 * (1 - alpha)]]], dtype=numpy.uint8)
    if border:
        mask_image = numpy.maximum(edge * numpy.array([[[200, 200, 0, 255]]]),
                mask_image)
    return result, mask_image

def activation_surface(data, target_shape=None, source_shape=None,
        scale_offset=None, deg=1, pad=True):
    """
    Generates an upsampled activation sample.
    Params:
        target_shape Shape of the output array.
        source_shape The centered shape of the output to match with data
            when upscaling.  Defaults to the whole target_shape.
        scale_offset The amount by which to scale, then offset data
            dimensions to end up with target dimensions.  A pair of pairs.
        deg Degree of interpolation to apply (1 = linear, etc).
        pad True to zero-pad the edge instead of doing a funny edge interp.
    """
    # Default is that nothing is resized.
    if target_shape is None:
        target_shape = data.shape
    # Make a default scale_offset to fill the image if there isn't one
    if scale_offset is None:
        scale = tuple(float(ts) / ds
                for ts, ds in zip(target_shape, data.shape))
        offset = tuple(0.5 * s - 0.5 for s in scale)
    else:
        scale, offset = (v for v in zip(*scale_offset))
    # Now we adjust offsets to take into account cropping and so on
    if source_shape is not None:
        offset = tuple(o + (ts - ss) / 2.0
                for o, ss, ts in zip(offset, source_shape, target_shape))
    # Pad the edge with zeros for sensible edge behavior
    if pad:
        zeropad = numpy.zeros(
                (data.shape[0] + 2, data.shape[1] + 2), dtype=data.dtype)
        zeropad[1:-1, 1:-1] = data
        data = zeropad
        offset = tuple((o - s) for o, s in zip(offset, scale))
    # Upsample linearly
    ty, tx = (numpy.arange(ts) for ts in target_shape)
    sy, sx = (numpy.arange(ss) * s + o
            for ss, s, o in zip(data.shape, scale, offset))
    levels = RectBivariateSpline(
            sy, sx, data, kx=deg, ky=deg)(ty, tx, grid=True)
    # Return the mask.
    return levels

def mask_border(mask, border=2):
    """Given a mask computes a border mask"""
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(2, 2)
    erosion = numpy.ones((mask.shape[0] + 10, mask.shape[1] + 10), dtype='int')
    erosion[5:5+mask.shape[0], 5:5+mask.shape[1]] = ~mask
    for _ in range(border):
        erosion = ndimage.binary_erosion(erosion, struct)
    return ~mask ^ erosion[5:5+mask.shape[0], 5:5+mask.shape[1]]

def bounding_rect(mask, pad=0):
    """Returns (r, b, l, r) boundaries so that all nonzero pixels in mask
    have locations (i, j) with  t <= i < b, and l <= j < r."""
    nz = mask.nonzero()
    if len(nz[0]) == 0:
        # print('no pixels')
        return (0, mask.shape[0], 0, mask.shape[1])
    (t, b), (l, r) = [(max(0, p.min() - pad), min(s, p.max() + 1 + pad))
            for p, s in zip(nz, mask.shape)]
    return (t, b, l, r)

def best_sub_rect(mask, shape, max_zoom=None, pad=2):
    """Finds the smallest subrectangle containing all the nonzeros of mask,
    matching the aspect ratio of shape, and where the zoom-up ratio is no
    more than max_zoom"""
    t, b, l, r = bounding_rect(mask, pad=pad)
    height = max(b - t, int(round(float(shape[0]) * (r - l) / shape[1])))
    if max_zoom is not None:
        height = int(max(round(float(shape[0]) / max_zoom), height))
    width = int(round(float(shape[1]) * height / shape[0]))
    nt = min(mask.shape[0] - height, max(0, (b + t - height) // 2))
    nb = nt + height
    nl = min(mask.shape[1] - width, max(0, (r + l - width) // 2))
    nr = nl + width
    return (nt, nb, nl, nr)

def zoom_image(img, source_rect, target_shape=None):
    """Zooms pixels from the source_rect of img to target_shape."""
    import warnings
    from scipy.ndimage import zoom
    if target_shape is None:
        target_shape = img.shape
    st, sb, sl, sr = source_rect
    source = img[st:sb, sl:sr]
    if source.shape == target_shape:
        return source
    zoom_tuple = tuple(float(t) / s
            for t, s in zip(target_shape, source.shape[:2])
            ) + (1,) * (img.ndim - 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning) # "output shape of zoom"
        target = zoom(source, zoom_tuple)
    assert target.shape[:2] == target_shape, (target.shape, target_shape)
    return target

def scale_offset(dilations):
    if len(dilations) == 0:
        return (1, 0)
    scale, offset = scale_offset(dilations[1:])
    kernel, stride, padding = dilations[0]
    scale *= stride
    offset *= stride
    offset += (kernel - 1) / 2.0 - padding
    return scale, offset

def choose_level(feature_map, percentile=0.8):
    '''
    Chooses the top 80% level (or whatever the level chosen).
    '''
    data_range = numpy.sort(feature_map.flatten())
    return numpy.interp(
            percentile, numpy.linspace(0, 1, len(data_range)), data_range)

def dilations(modulelist):
    result = []
    for module in modulelist:
        settings = tuple(getattr(module, n, d)
            for n, d in (('kernel_size', 1), ('stride', 1), ('padding', 0)))
        settings = (((s, s) if not isinstance(s, tuple) else s)
            for s in settings)
        if settings != ((1, 1), (1, 1), (0, 0)):
            result.append(zip(*settings))
    return zip(*result)

def grid_scale_offset(modulelist):
    '''Returns (yscale, yoffset), (xscale, xoffset) given a list of modules'''
    return tuple(scale_offset(d) for d in dilations(modulelist))

