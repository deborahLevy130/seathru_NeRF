#!/bin/bash

#CKPT_DIR="ckpt/uw/ablation_v1"
CKPT_DIR="ckpt/uw/cvpr2023"
#export CUDA_VISIBLE_DEVICES=1

EXPERIMENT=uw
DATA_DIR=data/"$EXPERIMENT"



for EXP_DIR in $CKPT_DIR/*;
do
  EXP_NAME=${EXP_DIR##*/}
  echo $EXP_NAME
  arrSCENE=(${EXP_NAME//_/ })
  SCENE="${arrSCENE[0]}"
  echo $SCENE

  python -m render \
  --gin_configs=${EXP_DIR}/config.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${EXP_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 240" \
  --gin_bindings="Config.render_dir = '${EXP_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 5" \
  --logtostderr


done
