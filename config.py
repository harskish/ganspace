# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import sys
import argparse
import json
from copy import deepcopy

class Config:
    def __init__(self, **kwargs):
        self.from_args([]) # set all defaults
        self.default_args = deepcopy(self.__dict__)
        self.from_dict(kwargs) # override

    def __str__(self):
        custom = {}
        default = {}

        # Find non-default arguments
        for k, v in self.__dict__.items():
            if k == 'default_args':
                continue

            in_default = k in self.default_args
            same_value = self.default_args.get(k) == v

            if in_default and same_value:
                default[k] = v
            else:
                custom[k] = v

        config = {
            'custom': custom,
            'default': default
        }

        return json.dumps(config, indent=4)

    def __repr__(self):
        return self.__str__()

    def from_dict(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
        return self

    def from_args(self, args=sys.argv[1:]):
        parser = argparse.ArgumentParser(description='GAN component analysis config')
        parser.add_argument('--model', dest='model', type=str, default='StyleGAN', help='The network to analyze') # StyleGAN, DCGAN, ProGAN, BigGAN-XYZ
        parser.add_argument('--layer', dest='layer', type=str, default='g_mapping', help='The layer to analyze')
        parser.add_argument('--class', dest='output_class', type=str, default=None, help='Output class to generate (BigGAN: Imagenet, ProGAN: LSUN)')
        parser.add_argument('--est', dest='estimator', type=str, default='ipca', help='The algorithm to use [pca, fbpca, cupca, spca, ica]')
        parser.add_argument('--sparsity', type=float, default=1.0, help='Sparsity parameter of SPCA')
        parser.add_argument('--video', dest='make_video', action='store_true', help='Generate output videos (MP4s)')
        parser.add_argument('--batch', dest='batch_mode', action='store_true', help="Don't open windows, instead save results to file")
        parser.add_argument('-b', dest='batch_size', type=int, default=None, help='Minibatch size, leave empty for automatic detection')
        parser.add_argument('-c', dest='components', type=int, default=80, help='Number of components to keep')
        parser.add_argument('-n', type=int, default=300_000, help='Number of examples to use in decomposition')
        parser.add_argument('--use_w', action='store_true', help='Use W latent space (StyleGAN(2))')
        parser.add_argument('--sigma', type=float, default=2.0, help='Number of stdevs to walk in visualize.py')
        parser.add_argument('--inputs', type=str, default=None, help='Path to directory with named components')
        parser.add_argument('--seed', type=int, default=None, help='Seed used in decomposition')
        parser.add_argument('--plot_directions', dest='np_directions', type=int, default=14, help='Number of components/directions to plot')
        parser.add_argument('--plot_images', dest='np_images', type=int, default=5, help='Number of images per component/direction to plot')
        parser.add_argument('--video_directions', dest='nv_images', type=int, default=5, help='Number of components/directions to create a video of')
        parser.add_argument('--video_images', dest='nv_images', type=int, default=150, help='Number of frames within a video of one direction/component')
        parser.add_argument('--scatter', dest='show_scatter', action='store_true', help='Plot a 2D scatter-plot of the activation space of two principal components')
        parser.add_argument('--scatter_samples', dest='scatter_samples', type=int, default=1000, help='Number of samples in the 2D scatter plot of the first two principal components')
        parser.add_argument('--scatter_images', dest='scatter_images', action='store_true', help='Plot encoded images instead of points within the scatter plot')
        parser.add_argument('--scatter_x', dest='scatter_x_axis_pc', type=int, default=1, help='Number of PC for x-axis in the scatter plot')
        parser.add_argument('--scatter_y', dest='scatter_y_axis_pc', type=int, default=2, help='Number of PC for y-axis in the scatter plot')


        args = parser.parse_args(args)
        assert args.np_images % 2 != 0, 'The number of plotted images per component (--plot_images) have to be odd.'

        if(args.model == "StyleGAN2-ada" and args.layer == "g_mapping"):
            print("No layer \'g_mapping\' in StyleGAN2-ada. Assuming you meant \'mapping\'")
            args.layer = "mapping"

        return self.from_dict(args.__dict__)
