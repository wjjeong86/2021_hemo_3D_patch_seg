#! /bin/bash

# https://pytorch.org/docs/stable/elastic/run.html 1.9 버전...
# single-node multi-worker

# 1.8 에서는 distributed.lunch 사용
# https://pytorch.org/docs/stable/distributed.html#launch-utility
# Launch utility
#
# Single-Node multi-process distributed training

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    ddp_example.py
    
    