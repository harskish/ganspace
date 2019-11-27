import matplotlib.pyplot as plt
import numpy

def plot_tensor_images(data, **kwargs):
    data = ((data + 1) / 2 * 255).permute(0, 2, 3, 1).byte().cpu().numpy()
    width = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    height = int(numpy.ceil(data.shape[0] / float(width)))
    kwargs = dict(kwargs)
    margin = 0.01
    if 'figsize' not in kwargs:
        # Size figure to one display pixel per data pixel
        dpi = plt.rcParams['figure.dpi']
        kwargs['figsize'] = (
                (1 + margin) * (width * data.shape[2] / dpi),
                (1 + margin) * (height * data.shape[1] / dpi))
    f, axarr = plt.subplots(height, width, **kwargs)
    if len(numpy.shape(axarr)) == 0:
        axarr = numpy.array([[axarr]])
    if len(numpy.shape(axarr)) == 1:
        axarr = axarr[None,:]
    for i, im in enumerate(data):
        ax = axarr[i // width, i % width]
        ax.imshow(data[i])
        ax.axis('off')
    for i in range(i, width * height):
        ax = axarr[i // width, i % width]
        ax.axis('off')
    plt.subplots_adjust(wspace=margin, hspace=margin,
            left=0, right=1, bottom=0, top=1)
    plt.show()

def plot_max_heatmap(data, shape=None, **kwargs):
    if shape is None:
        shape = data.shape[2:]
    data = data.max(1)[0].cpu().numpy()
    vmin = data.min()
    vmax = data.max()
    width = int(numpy.ceil(numpy.sqrt(data.shape[0])))
    height = int(numpy.ceil(data.shape[0] / float(width)))
    kwargs = dict(kwargs)
    margin = 0.01
    if 'figsize' not in kwargs:
        # Size figure to one display pixel per data pixel
        dpi = plt.rcParams['figure.dpi']
        kwargs['figsize'] = (
                width * shape[1] / dpi, height * shape[0] / dpi)
    f, axarr = plt.subplots(height, width, **kwargs)
    if len(numpy.shape(axarr)) == 0:
        axarr = numpy.array([[axarr]])
    if len(numpy.shape(axarr)) == 1:
        axarr = axarr[None,:]
    for i, im in enumerate(data):
        ax = axarr[i // width, i % width]
        img = ax.imshow(data[i], vmin=vmin, vmax=vmax, cmap='hot')
        ax.axis('off')
    for i in range(i, width * height):
        ax = axarr[i // width, i % width]
        ax.axis('off')
    plt.subplots_adjust(wspace=margin, hspace=margin,
            left=0, right=1, bottom=0, top=1)
    plt.show()
