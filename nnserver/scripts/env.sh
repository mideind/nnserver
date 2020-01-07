#!/bin/sh
# Assumes running inside docker container
ABS_PATH=""

PROBLEM=parsing_icelandic16k_v5
MODEL=transformer
HPARAMS_SET=transformer_base_single_gpu

DATA_DIR=$ABS_PATH/t2t_data
TRAIN_DIR=$ABS_PATH/t2t_train/$PROBLEM.test
USR_DIR=$ABS_PATH/t2t_usr
TMP_DIR=$ABS_PATH/t2t_tmp

PROBLEM_DATA_DIR=$ABS_PATH/data/en-is/problems/$PROBLEM/

BEAM_SIZE=4
ALPHA=0.7
EXTRA_LENGTH=64
DECODE_HPARAMS="alpha=$ALPHA,beam_size=$BEAM_SIZE,extra_length=$EXTRA_LENGTH"
EXPORT_DECODE_HPARAMS="alpha=$ALPHA,beam_size=$BEAM_SIZE,extra_length=$EXTRA_LENGTH"


TRAIN_STEPS="1_000"
TRAIN_HPARAMS='batch_size=1600,eval_drop_long_sequences=True,max_length=400,shared_embedding_and_softmax_weights=False'
EVAL_STEPS=2000
EVAL_FREQ=25000

EXPORT_DIR=/models/$PROBLEM
