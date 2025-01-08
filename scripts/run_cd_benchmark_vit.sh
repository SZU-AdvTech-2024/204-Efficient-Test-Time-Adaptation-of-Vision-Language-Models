#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --wandb-log \
                                                --datasets caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/stanford_cars/ucf101/sun397/food101 \
                                                --data-root /public/datasets/ \
                                                --backbone ViT-B/16

##!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
#                                                --wandb-log \
#                                                --datasets dtd \
#                                                --data-root /public/datasets/ \
#                                                --backbone ViT-B/16