#!/bin/bash

#CKPT_DIR="ckpt/uw/ablation_v1"
CKPT_DIR="ckpt/results/uw/cvpr2023/"
#export CUDA_VISIBLE_DEVICES=0

EXPERIMENT=uw
DATA_DIR=data/"$EXPERIMENT"



for EXP_DIR in $CKPT_DIR/*;
do
  EXP_NAME=${EXP_DIR##*/}
  echo $EXP_NAME
  arrSCENE=(${EXP_NAME//_/ })
  SCENE="${arrSCENE[0]}"
  echo $SCENE
  for mode in  train test
  do
    PREDS_DIR="${EXP_DIR}/${mode}_preds"
    if test -d "$PREDS_DIR"; then
      echo "----------------------------------"
      echo "  Skipping $PREDS_DIR...";
      echo "----------------------------------"
    else
      echo "----------------------------------"
      echo "  Processing $PREDS_DIR...";
      echo "----------------------------------"
      if [[ $mode == "train" ]]; then
        python -m eval \
          --gin_configs=${EXP_DIR}/config.gin \
          --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
          --gin_bindings="Config.checkpoint_dir = '${EXP_DIR}'" \
          --gin_bindings="Config.eval_on_train = True" \
          --gin_bindings="Config.eval_only_once = True" \
          --logtostderr
      else
        python -m eval \
          --gin_configs=${EXP_DIR}/config.gin \
          --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
          --gin_bindings="Config.checkpoint_dir = '${EXP_DIR}'" \
          --gin_bindings="Config.eval_only_once = True" \
          --logtostderr
      fi
    fi
  done
done




