#!/bin/sh
# Assumes running inside docker container
SCRIPT_NAME=$0
SCRIPT_DIR=$(dirname "$0")
. $SCRIPT_DIR/env.sh

$SCRIPT_DIR/generate_data.sh
$SCRIPT_DIR/train.sh
$SCRIPT_DIR/validate.sh
