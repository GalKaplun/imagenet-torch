#!/bin/bash

MY_RUN=$RANDOM$RANDOM
OTHER_PARAMS=${@:1}

WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7 python ./run.py -m \
  run_id=$MY_RUN \
  logger.tags='[clean_cifar5m]' \
  datamodule.batch_size=256 \
  datamodule.dataset=cifar5m \
  model.weight_decay=0.00005 \
  model.learning_rate=0.1 \
  $OTHER_PARAMS
