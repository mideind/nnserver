#!/bin/sh
# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

mkdir -p $EXPORT_DIR

t2t-exporter \
    --t2t_usr_dir=$USR_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --output_dir=$TRAIN_DIR \
    --hparams=$HPARAMS \
    --export_dir=$EXPORT_DIR \
    --decode_hparams=$EXPORT_DECODE_HPARAMS

