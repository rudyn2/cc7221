#!/bin/bash

docker run -d --rm -v ~/cachefs/cbir/data:/home/cc7221/data -v ~/cachefs/cbir/weights/home/cc7221/weights \
       -e WANDB_API_KEY=5990125fc96222f63d970131fd574f587ed56e1e --gpus all -e NVIDIA_VISIBLE_DEVICES=1 \
       -it trainer bash

