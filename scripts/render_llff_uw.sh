#!/bin/bash
# This file was modified by Deborah Levy
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#export CUDA_VISIBLE_DEVICES=0

SCENE=Panama
EXPERIMENT=uw
EXPERIMENT_NAME=exp1
DATA_DIR=data/
CHECKPOINT_DIR=ckpt/"$EXPERIMENT"/"$SCENE"_"$EXPERIMENT_NAME"

python -m render \
  --gin_configs=${CHECKPOINT_DIR}/config.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 240" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 5" \
  --logtostderr
