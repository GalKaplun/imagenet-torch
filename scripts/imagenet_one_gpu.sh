#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}

#WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7 python ./run.py -m \
HYDRA_FULL_ERROR=1 WANDB_MODE=disabled python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[imagenet-one-gpu]' \
  datamodule.dataset=imagenet \
  model.arch=resnet50 \
  model.weight_decay=0.00005 \
  model.learning_rate=0.1 \
  $OTHER_PARAMS
