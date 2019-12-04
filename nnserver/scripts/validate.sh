#!/bin/sh
# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

TRNSL_SRC_FILE=$ABS_PATH/data/en-is/laser.tatoeba.isl-eng.eng
REFERENCE_FILE=$ABS_PATH/data/en-is/laser.tatoeba.isl-eng.isl

mkdir -p $PROBLEM_DATA_DIR
LOG_FILE=$PROBLEM_DATA_DIR/log.tatoeba.txt

TRNSL_RESULTS_FILE=$PROBLEM_DATA_DIR/laser.tatoeba.alpha-$ALPHA.beam_size-$BEAM_SIZE.extra_length-$EXTRA_LENGTH.isl

echo "-------------------------------------------" >> $LOG_FILE
date >> $LOG_FILE
echo "$PROBLEM" >> $LOG_FILE
echo "$DECODE_HPARAMS" >> $LOG_FILE
echo "$MESSAGE" >> $LOG_FILE

START_TIME=$(date +%s)
t2t-decoder \
    --t2t_usr_dir=$USR_DIR \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --tmp_dir=$TMP_DIR \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --output_dir=$TRAIN_DIR \
    --decode_from_file=$TRNSL_SRC_FILE \
    --decode_to_file=$TRNSL_RESULTS_FILE \
    --decode_hparams=$DECODE_HPARAMS
END_TIME=$(date +%s)
DECODE_ELAPS=$((END_TIME - START_TIME))

echo "Decode time: ${DECODE_ELAPS} seconds" >> $LOG_FILE
t2t-bleu --translation=$TRNSL_RESULTS_FILE --reference=$REFERENCE_FILE >> $LOG_FILE
echo "" >> $LOG_FILE
