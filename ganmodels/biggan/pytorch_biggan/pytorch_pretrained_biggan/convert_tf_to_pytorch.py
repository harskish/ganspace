# coding: utf-8
"""
Convert a TF Hub model for BigGAN in a PT one.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

from itertools import chain

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from .model import BigGAN, WEIGHTS_NAME, CONFIG_NAME
from .config import BigGANConfig

logger = logging.getLogger(__name__)


def extract_batch_norm_stats(tf_model_path, batch_norm_stats_path=None):
    try:
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError("Loading a TensorFlow models in PyTorch, requires TensorFlow and TF Hub to be installed. "
                          "Please see https://www.tensorflow.org/install/ for installation instructions for TensorFlow. "
                          "And see https://github.com/tensorflow/hub for installing Hub. "
                          "Probably pip install tensorflow tensorflow-hub")
    tf.reset_default_graph()
    logger.info('Loading BigGAN module from: {}'.format(tf_model_path))
    module = hub.Module(tf_model_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
              for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    stacks = sum(((i*10 + 1, i*10 + 3, i*10 + 6, i*10 + 8) for i in range(50)), ())
    numpy_stacks = []
    for i in stacks:
        logger.info("Retrieving module_apply_default/stack_{}".format(i))
        try:
            stack_var = tf.get_default_graph().get_tensor_by_name("module_apply_default/stack_%d:0" % i)
        except KeyError:
            break  # We have all the stats
        numpy_stacks.append(sess.run(stack_var))

    if batch_norm_stats_path is not None:
        torch.save(numpy_stacks, batch_norm_stats_path)
    else:
        return numpy_stacks


def build_tf_to_pytorch_map(model, config):
    """ Build a map from TF variables to PyTorch modules. """
    tf_to_pt_map = {}

    # Embeddings and GenZ
    tf_to_pt_map.update({'linear/w/ema_0.9999': model.embeddings.weight,
                         'Generator/GenZ/G_linear/b/ema_0.9999': model.generator.gen_z.bias,
                         'Generator/GenZ/G_linear/w/ema_0.9999': model.generator.gen_z.weight_orig,
                         'Generator/GenZ/G_linear/u0': model.generator.gen_z.weight_u})

    # GBlock blocks
    model_layer_idx = 0
    for i, (up, in_channels, out_channels) in enumerate(config.layers):
        if i == config.attention_layer_position:
            model_layer_idx += 1
        layer_str = "Generator/GBlock_%d/" % i if i > 0 else "Generator/GBlock/"
        layer_pnt = model.generator.layers[model_layer_idx]
        for i in range(4):  #  Batchnorms
            batch_str = layer_str + ("BatchNorm_%d/" % i if i > 0 else "BatchNorm/")
            batch_pnt = getattr(layer_pnt, 'bn_%d' % i)
            for name in ('offset', 'scale'):
                sub_module_str = batch_str + name + "/"
                sub_module_pnt = getattr(batch_pnt, name)
                tf_to_pt_map.update({sub_module_str + "w/ema_0.9999": sub_module_pnt.weight_orig,
                                     sub_module_str + "u0": sub_module_pnt.weight_u})
        for i in range(4):  # Convolutions
            conv_str = layer_str + "conv%d/" % i
            conv_pnt = getattr(layer_pnt, 'conv_%d' % i)
            tf_to_pt_map.update({conv_str + "b/ema_0.9999": conv_pnt.bias,
                                 conv_str + "w/ema_0.9999": conv_pnt.weight_orig,
                                 conv_str + "u0": conv_pnt.weight_u})
        model_layer_idx += 1

    # Attention block
    layer_str = "Generator/attention/"
    layer_pnt = model.generator.layers[config.attention_layer_position]
    tf_to_pt_map.update({layer_str + "gamma/ema_0.9999": layer_pnt.gamma})
    for pt_name, tf_name in zip(['snconv1x1_g', 'snconv1x1_o_conv', 'snconv1x1_phi', 'snconv1x1_theta'],
                                ['g/', 'o_conv/', 'phi/', 'theta/']):
        sub_module_str = layer_str + tf_name
        sub_module_pnt = getattr(layer_pnt, pt_name)
        tf_to_pt_map.update({sub_module_str + "w/ema_0.9999": sub_module_pnt.weight_orig,
                             sub_module_str + "u0": sub_module_pnt.weight_u})

    # final batch norm and conv to rgb
    layer_str = "Generator/BatchNorm/"
    layer_pnt = model.generator.bn
    tf_to_pt_map.update({layer_str + "offset/ema_0.9999": layer_pnt.bias,
                         layer_str + "scale/ema_0.9999": layer_pnt.weight})
    layer_str = "Generator/conv_to_rgb/"
    layer_pnt = model.generator.conv_to_rgb
    tf_to_pt_map.update({layer_str + "b/ema_0.9999": layer_pnt.bias,
                         layer_str + "w/ema_0.9999": layer_pnt.weight_orig,
                         layer_str + "u0": layer_pnt.weight_u})
    return tf_to_pt_map


def load_tf_weights_in_biggan(model, config, tf_model_path, batch_norm_stats_path=None):
    """ Load tf checkpoints and standing statistics in a pytorch model
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        raise ImportError("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
    # Load weights from TF model
    checkpoint_path = tf_model_path + "/variables/variables"
    init_vars = tf.train.list_variables(checkpoint_path)
    from pprint import pprint
    pprint(init_vars)

    # Extract batch norm statistics from model if needed
    if batch_norm_stats_path:
        stats = torch.load(batch_norm_stats_path)
    else:
        logger.info("Extracting batch norm stats")
        stats = extract_batch_norm_stats(tf_model_path)

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_to_pytorch_map(model, config)

    tf_weights = {}
    for name in tf_to_pt_map.keys():
        array = tf.train.load_variable(checkpoint_path, name)
        tf_weights[name] = array
        # logger.info("Loading TF weight {} with shape {}".format(name, array.shape))

    # Load parameters
    with torch.no_grad():
        pt_params_pnt = set()
        for name, pointer in tf_to_pt_map.items():
            array = tf_weights[name]
            if pointer.dim() == 1:
                if pointer.dim() < array.ndim:
                    array = np.squeeze(array)
            elif pointer.dim() == 2:  # Weights
                array = np.transpose(array)
            elif pointer.dim() == 4:  # Convolutions
                array = np.transpose(array, (3, 2, 0, 1))
            else:
                raise "Wrong dimensions to adjust: " + str((pointer.shape, array.shape))
            if pointer.shape != array.shape:
                raise ValueError("Wrong dimensions: " + str((pointer.shape, array.shape)))
            logger.info("Initialize PyTorch weight {} with shape {}".format(name, pointer.shape))
            pointer.data = torch.from_numpy(array) if isinstance(array, np.ndarray) else torch.tensor(array)
            tf_weights.pop(name, None)
            pt_params_pnt.add(pointer.data_ptr())

        # Prepare SpectralNorm buffers by running one step of Spectral Norm (no need to train the model):
        for module in model.modules():
            for n, buffer in module.named_buffers():
                if n == 'weight_v':
                    weight_mat = module.weight_orig
                    weight_mat = weight_mat.reshape(weight_mat.size(0), -1)
                    u = module.weight_u

                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=config.eps)
                    buffer.data = v
                    pt_params_pnt.add(buffer.data_ptr())

                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=config.eps)
                    module.weight_u.data = u
                    pt_params_pnt.add(module.weight_u.data_ptr())

        # Load batch norm statistics
        index = 0
        for layer in model.generator.layers:
            if not hasattr(layer, 'bn_0'):
                continue
            for i in range(4):  #  Batchnorms
                bn_pointer = getattr(layer, 'bn_%d' % i)
                pointer = bn_pointer.running_means
                if pointer.shape != stats[index].shape:
                    raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                pointer.data = torch.from_numpy(stats[index])
                pt_params_pnt.add(pointer.data_ptr())

                pointer = bn_pointer.running_vars
                if pointer.shape != stats[index+1].shape:
                    raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
                pointer.data = torch.from_numpy(stats[index+1])
                pt_params_pnt.add(pointer.data_ptr())

                index += 2

        bn_pointer = model.generator.bn
        pointer = bn_pointer.running_means
        if pointer.shape != stats[index].shape:
            raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
        pointer.data = torch.from_numpy(stats[index])
        pt_params_pnt.add(pointer.data_ptr())

        pointer = bn_pointer.running_vars
        if pointer.shape != stats[index+1].shape:
            raise "Wrong dimensions: " + str((pointer.shape, stats[index].shape))
        pointer.data = torch.from_numpy(stats[index+1])
        pt_params_pnt.add(pointer.data_ptr())

    remaining_params = list(n for n, t in chain(model.named_parameters(), model.named_buffers()) \
                            if t.data_ptr() not in pt_params_pnt)

    logger.info("TF Weights not copied to PyTorch model: {} -".format(', '.join(tf_weights.keys())))
    logger.info("Remanining parameters/buffers from PyTorch model: {} -".format(', '.join(remaining_params)))

    return model


BigGAN128 = BigGANConfig(output_dim=128, z_dim=128, class_embed_dim=128, channel_width=128, num_classes=1000,
                         layers=[(False, 16, 16),
                                 (True, 16, 16),
                                 (False, 16, 16),
                                 (True, 16, 8),
                                 (False, 8, 8),
                                 (True, 8, 4),
                                 (False, 4, 4),
                                 (True, 4, 2),
                                 (False, 2, 2),
                                 (True, 2, 1)],
                         attention_layer_position=8, eps=1e-4, n_stats=51)

BigGAN256 = BigGANConfig(output_dim=256, z_dim=128, class_embed_dim=128, channel_width=128, num_classes=1000,
                         layers=[(False, 16, 16),
                                 (True, 16, 16),
                                 (False, 16, 16),
                                 (True, 16, 8),
                                 (False, 8, 8),
                                 (True, 8, 8),
                                 (False, 8, 8),
                                 (True, 8, 4),
                                 (False, 4, 4),
                                 (True, 4, 2),
                                 (False, 2, 2),
                                 (True, 2, 1)],
                         attention_layer_position=8, eps=1e-4, n_stats=51)

BigGAN512 = BigGANConfig(output_dim=512, z_dim=128, class_embed_dim=128, channel_width=128, num_classes=1000,
                         layers=[(False, 16, 16),
                                 (True, 16, 16),
                                 (False, 16, 16),
                                 (True, 16, 8),
                                 (False, 8, 8),
                                 (True, 8, 8),
                                 (False, 8, 8),
                                 (True, 8, 4),
                                 (False, 4, 4),
                                 (True, 4, 2),
                                 (False, 2, 2),
                                 (True, 2, 1),
                                 (False, 1, 1),
                                 (True, 1, 1)],
                         attention_layer_position=8, eps=1e-4, n_stats=51)


def main():
    parser = argparse.ArgumentParser(description="Convert a BigGAN TF Hub model in a PyTorch model")
    parser.add_argument("--model_type", type=str, default="", required=True,
                        help="BigGAN model type (128, 256, 512)")
    parser.add_argument("--tf_model_path", type=str, default="", required=True,
                        help="Path of the downloaded TF Hub model")
    parser.add_argument("--pt_save_path", type=str, default="",
                        help="Folder to save the PyTorch model (default: Folder of the TF Hub model)")
    parser.add_argument("--batch_norm_stats_path", type=str, default="",
                        help="Path of previously extracted batch norm statistics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.pt_save_path:
        args.pt_save_path = args.tf_model_path

    if args.model_type == "128":
        config = BigGAN128
    elif args.model_type == "256":
        config = BigGAN256
    elif args.model_type == "512":
        config = BigGAN512
    else:
        raise ValueError("model_type should be one of 128, 256 or 512")

    model = BigGAN(config)
    model = load_tf_weights_in_biggan(model, config, args.tf_model_path, args.batch_norm_stats_path)

    model_save_path = os.path.join(args.pt_save_path, WEIGHTS_NAME)
    config_save_path = os.path.join(args.pt_save_path, CONFIG_NAME)

    logger.info("Save model dump to {}".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    logger.info("Save configuration file to {}".format(config_save_path))
    with open(config_save_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())

if __name__ == "__main__":
    main()
