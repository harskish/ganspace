import os, torch, numpy
from torch.utils.data import TensorDataset

def z_dataset_for_model(model, size=100, seed=1):
    return TensorDataset(z_sample_for_model(model, size, seed))

def z_sample_for_model(model, size=100, seed=1):
    # If the model is marked with an input shape, use it.
    if hasattr(model, 'input_shape'):
        sample = standard_z_sample(size, model.input_shape[1], seed=seed).view(
                (size,) + model.input_shape[1:])
        return sample
    # Examine first conv in model to determine input feature size.
    first_layer = [c for c in model.modules()
            if isinstance(c, (torch.nn.Conv2d, torch.nn.ConvTranspose2d,
                torch.nn.Linear))][0]
    # 4d input if convolutional, 2d input if first layer is linear.
    if isinstance(first_layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        sample = standard_z_sample(
                size, first_layer.in_channels, seed=seed)[:,:,None,None]
    else:
        sample = standard_z_sample(
                size, first_layer.in_features, seed=seed)
    return sample

def standard_z_sample(size, depth, seed=1, device=None):
	'''
	Generate a standard set of random Z as a (size, z_dimension) tensor.
	With the same random seed, it always returns the same z (e.g.,
	the first one is always the same regardless of the size.)
	'''
	# Use numpy RandomState since it can be done deterministically
	# without affecting global state
	rng = numpy.random.RandomState(seed)
	result = torch.from_numpy(
			rng.standard_normal(size * depth)
			.reshape(size, depth)).float()
	if device is not None:
		result = result.to(device)
	return result

