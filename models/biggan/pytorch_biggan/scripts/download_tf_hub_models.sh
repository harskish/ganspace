# Copyright (c) 2019-present, Thomas Wolf, Huggingface Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -x

models="128 256 512"

mkdir -p models/model_128
mkdir -p models/model_256
mkdir -p models/model_512

# Download TF Hub models.
for model in $models
do
    curl -L "https://tfhub.dev/deepmind/biggan-deep-$model/1?tf-hub-format=compressed" | tar -zxvC models/model_$model
done
