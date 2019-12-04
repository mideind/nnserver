#!/bin/sh
# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

# No training steps, generate only
TRAIN_STEPS="0"

t2t-trainer \
  --generate_data \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR
