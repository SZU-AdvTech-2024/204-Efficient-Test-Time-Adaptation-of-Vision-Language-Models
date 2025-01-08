import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn.functional as F
import operator
import torch.nn as nn
import matplotlib.pyplot as plt

import clip
from utils import *


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True,
                        help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true',
                        help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True,
                        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/',
                        help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def cache_key_value(image_features, cache, alpha, beta, clip_weights,neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []

        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (
            F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        return cache_keys, cache_values, all_classes


def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, clip_weights):
    affinity = image_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super(PositiveCacheResidue, self).__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cache_size]).half().cuda(), requires_grad=True)

    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys


def neg_compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super(TextResidue, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)

    def forward(self, x):
        new_clip_weights = x.clone() + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights

    def reset(self):
        self.residual = nn.Parameter(torch.zeros([self.feat_dim, self.cate_num]).half().cuda(), requires_grad=True)


def run_test_tda(pos_cfg, neg_cfg, lr_cfg, loader, clip_model, clip_weights):
    with torch.cuda.amp.autocast():
        pos_cache, neg_cache, accuracies = {}, {}, []

        print("----------------new TDA-----------------------")
        pos_enabled = pos_cfg['enabled']
        neg_enabled = neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}

        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in
                          ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        clip_weights_global = clip_weights.clone()

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local)
            new_clip_weights = text_residue(clip_weights_local)

            image_features_x, clip_logits, first_entropy, prob_map, pred = get_clip_logits(images, clip_model,
                                                                                           new_clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(first_entropy, clip_weights)

            if pos_enabled:
                entropy = get_entropy(first_entropy, clip_weights)
                update_cache(pos_cache, pred, [image_features_x, first_entropy], pos_params['shot_capacity'])  # new
                pos_cache_keys, pos_cache_values, all_classes = cache_key_value(image_features_x, pos_cache,
                                                                                pos_params['alpha'], pos_params['beta'],
                                                                                clip_weights)
                pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < \
                    neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features_x, first_entropy, prob_map], neg_params['shot_capacity'],
                             True)

            steps = 1
            for j in range(steps):
                new_clip_weights = text_residue(clip_weights_local)
                final_logits = clip_logits.clone()

                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(image_features_x, new_pos_cache_keys, pos_cache_values,
                                                         pos_params['alpha'], pos_params['beta'], clip_weights)
                    loss = avg_entropy(final_logits)

                lr_text = lr_cfg['text']
                optimizer = torch.optim.AdamW([
                    {'params': text_residue.parameters(), 'lr': lr_text, 'eps': 1e-3, 'weight_decay': 1e-1}
                ])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, new_clip_weights)
                final_logits = clip_logits.clone()

                #
                if pos_enabled and pos_cache:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits += compute_cache_logits(image_features, new_pos_cache_keys, pos_cache_values,
                                                         pos_params['alpha'], pos_params['beta'], clip_weights)

                # new neg sample
                if neg_enabled and neg_cache:
                    final_logits -= neg_compute_cache_logits(image_features, neg_cache, neg_params['alpha'],
                                                             neg_params['beta'], clip_weights, (
                                                                 neg_params['mask_threshold']['lower'],
                                                                 neg_params['mask_threshold']['upper']))

                acc = cls_acc(final_logits, target.cuda())
                accuracies.append(acc)
                wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

            if i % 1000 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
        return sum(accuracies) / len(accuracies)


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(cfg['positive'], cfg['negative'], cfg['learning_rate'], test_loader, clip_model,
                           clip_weights)

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()


if __name__ == "__main__":
    main()



