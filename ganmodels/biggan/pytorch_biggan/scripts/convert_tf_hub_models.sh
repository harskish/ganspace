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

# Convert TF Hub models.
for model in $models
do
    pytorch_pretrained_biggan --model_type $model --tf_model_path models/model_$model --pt_save_path models/model_$model
done
