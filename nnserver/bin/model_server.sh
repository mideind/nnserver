REST_PORT=8080
GRPC_PORT=8081
ABS_HOME="/home/haukur"
sudo docker pull tensorflow/serving
sudo docker run \
     --rm --gpus device=0 \
     --name "model_server" \
     --interactive \
     --tty \
     --shm-size=1g \
     --ulimit memlock=-1 \
     --volume $ABS_HOME/models:/models \
     --publish $REST_PORT:$REST_PORT/tcp \
     --publish $GRPC_PORT:$GRPC_PORT/tcp \
     --env HOST_PERMS="$(id -u):$(id -g)" \
     --env LANG="C.UTF-8" \
     --env LANGUAGE="C.UTF-8" \
     --env LC_ALL="C.UTF-8" \
     --env TERM="xterm-256color" \
     tensorflow/serving --model_config_file=/models/models.conf --port=$GRPC_PORT --rest_api_port=$REST_PORT
