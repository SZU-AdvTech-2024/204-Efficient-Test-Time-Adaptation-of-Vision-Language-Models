#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets I/A/V/R/S \
                                                --data-root /public/datasets/ \
                                                --backbone ViT-B/16

##!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
#                                                --wandb-log \
#                                                --datasets V \
#                                                --data-root /public/datasets/ \
#                                                --backbone ViT-B/16