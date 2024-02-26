docker run --gpus all  --shm-size=64g --net=host -dit --rm --name megatron -v /data:/data -v /root/.ssh:/root/.ssh nvcr.io/nvidia/pytorch:23.12-py3

docker exec -it megatron bash

docker stop megatron