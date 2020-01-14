##!/usr/bin/env bash
NN_DIR="/Users/vesteinnsnaebjarnarson/Work/mideind/nntrainer/nnserver/nnserver"
T2T_VERSION=1.14.1
IMAGE="ntrainer"
docker run --name "parsing-v4" \
       --rm \
       --interactive \
       --tty \
       --shm-size=1g \
       --ulimit memlock=-1 \
       --volume $NN_DIR/data:/data \
       --volume $NN_DIR/t2t_datagen:/t2t_tmp \
       --volume $NN_DIR/scripts:/scripts \
       --volume $NN_DIR/t2t_train:/t2t_train \
       --volume $NN_DIR/t2t_data:/t2t_data \
       --volume $NN_DIR/t2t_usr:/t2t_usr \
       --publish 8888:8888/tcp \
       --publish 9999:9999/tcp \
       --env HOST_PERMS="$(id -u):$(id -g)" \
       $IMAGE /scripts/train.sh
