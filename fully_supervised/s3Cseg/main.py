import os
import sys
import json

import argparse
import configs.models
import configs.base

import time
import datetime
import math

import torch
import torch.nn as nn
from torch.nn.functional import one_hot

from models.build import build_model
from utils.logger import MetricLogger

import data.augmentations as aug
from torchvision.transforms import Compose

from data.dataset import SonarDataset
from torch.utils.data.dataloader import DataLoader as DL

from utils import utils
import wandb


arch_choices = ['ecnet','dcnet','rtseg']
arch_configs = {k: configs.models.__getattribute__(k) for k in arch_choices}


def get_args_parser():
    parser = argparse.ArgumentParser('s3Cseg Training', add_help=False)

    parser.add_argument('--wandb_entity', type=str, required=True,
                        help='WandB entity.')
    parser.add_argument('--wandb_project', type=str, required=True,
                        help='WandB project name.')
    parser.add_argument('--wandb_api_key', type=str, required=True,
                        help='WandB api key.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data.')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Path to save logs, checkpoints and models.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to configuration file.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from.')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='Path to pretrained weights.')
    parser.add_argument('--arch', type=str, default='unet', choices=arch_choices,
                        help='Type of architecture.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of distinct images loaded per GPU.')

    return parser


def main(args, config):
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    wandb_logger = wandb.init(
        entity=args.wandb_entity, project=args.wandb_project,
        dir=args.out_dir, config=config, resume=True,
    )

    # ================ preparing data ================
    
    data_transforms = Compose([
        aug.RandomRotate(),
        aug.RandomHorizontalFlip(),
        aug.RandomVerticalFlip(),
        aug.ColorJitter(),
        aug.GaussianBlur(),
        aug.Sharpen(),
        aug.ToTensor(),
        aug.Normalize(config.DATA.MEAN, config.DATA.STD)
    ])

    datasets = {x: SonarDataset(os.path.join(args.data_dir,x), data_transforms)
                    for x in ('train','val')}
    
    data_loaders = {x: DL(datasets[x], batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=True, pin_memory=True, drop_last=True)
                        for x in ('train','val')}

    # ================ building student/teacher networks ================

    model = build_model(config)
    model.cuda()

    # load pretrained weights, if any
    if args.load_weights:
        utils.load_pretrained_weights(model, args.load_weights, 'train')
    
    wandb_logger.watch(model)
    
    print(f"Model built: a {args.arch} network.")

    # ================ initializing optimizer ================

    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)

    # ================ initializing loss ================

    if config.DATA.CLASS_FREQ is not None:
        weights = 1/(len(config.DATA.CLASS_FREQ)*torch.FloatTensor(config.DATA.CLASS_FREQ))
        # weights /= sum(weights) # no normalization since reduction='mean' for the loss
        # see: https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
        weights = weights.cuda(non_blocking=True)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ================ initializing schedulers ================

    ITER_PER_EPOCH = len(data_loaders['train'])

    if config.TRAIN.LR_SCHEDULER == 'step':
        lr_schedule = utils.stepLR_scheduler(
            config.TRAIN.LR, config.TRAIN.EPOCHS, ITER_PER_EPOCH,
            config.TRAIN.LR_DECAY, config.TRAIN.LR_STEP
        )
    elif config.TRAIN.LR_SCHEDULER == 'poly':
        lr_schedule = utils.polyLR_scheduler(
            config.TRAIN.LR, config.TRAIN.EPOCHS, ITER_PER_EPOCH,
            config.TRAIN.WARMUP_EPOCHS
        )
    else:
        lr_schedule = utils.constLR_scheduler(
            config.TRAIN.LR, config.TRAIN.EPOCHS, ITER_PER_EPOCH
        )

    print(f"Loss, optimizer and scheduler ready.")

    # ================ training and logging ================

    to_restore = {'epoch': 0}
    if args.load_checkpoint:
        utils.resume_from_checkpoint(args.load_checkpoint, run_variables=to_restore,
                                    model=model, optimizer=optimizer, criterion=criterion)
    start_epoch = to_restore['epoch']

    start_time = time.time()
    print(f"Starting training of s3Cseg ! from epoch {start_epoch}")

    best_miou = 0.
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):

        # train and validate
        train_stats, val_stats = train_and_validate(model, data_loaders, criterion, optimizer,
                                        lr_schedule, epoch, ITER_PER_EPOCH, config, args)

        # save checkpoint
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch + 1,
            'config': config,
            'args': args
        }
        torch.save(save_dict, os.path.join(args.out_dir, 'checkpoint.pth'))

        # log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch
        }
        with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        wandb_logger.log({
            'loss': {
                'train': train_stats['loss'],
                'val': val_stats['loss']
            },
            'mIOU': {
                'train': train_stats['mean_iou'],
                'val': val_stats['mean_iou']
            },
            'lr': train_stats['lr']
        })
        
        # save best model
        if val_stats['mean_iou'] > best_miou:
            best_miou = val_stats['mean_iou']
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_best.pth'))
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_%d.pth'%(epoch+1)))

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_fin.pth'))

    wandb_logger.finish()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_and_validate(model, data_loaders, criterion, optimizer,
                        lr_schedule, epoch, iter_per_epoch, config, args):
    
    metric_loggers = {}
    headers = {}
    
    metric_loggers['train'] = MetricLogger(delimiter="  ")
    headers['train'] = 'Epoch: [{}/{}]'.format(epoch, config.TRAIN.EPOCHS)

    metric_loggers['val'] = MetricLogger(delimiter="  ")
    headers['val'] = 'Validation:'

    for phase in ('train', 'val'):
        
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        for it, (images, labels) in enumerate(metric_loggers[phase].log_every(data_loaders[phase],
                                                1, headers[phase])):

            # move images to gpu
            images = images.cuda(non_blocking=True)     # batch x channel x width x height
            labels = labels.cuda(non_blocking=True)     # batch x    _    x width x height

            # zero parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(phase=='train'):
                # forward pass
                output = model(images)                  # batch x classes x width x height
                preds = torch.argmax(output, dim=1)     # batch x    _    x width x height
                
                # compute loss
                loss = criterion(output, labels)

                # one hot
                labels = one_hot(labels, output.shape[1]).permute(0,3,1,2)  # B x C x H x W
                preds = one_hot(preds, output.shape[1]).permute(0,3,1,2)    # B x C x H x W

                # compute mean iou
                inter = torch.sum(torch.logical_and(preds,labels), dim=(2,3))       # B x C
                union = torch.sum(torch.logical_or(preds,labels), dim=(2,3))        # B x C
                miou = torch.mean((inter+1e-6)/(union+1e-6), dim=(0,1))

                # writing logs on a NaN to debug
                if not math.isfinite(loss.item()):
                    print('Loss is {}, stopping training'.format(loss.item()), force=True)
                    save_dict = {
                        'student': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'criterion': criterion.state_dict(),
                        'epoch': epoch + 1,
                        'config': config,
                        'args': args
                    }
                    torch.save(save_dict, os.path.join(args.out_dir, 'ckpt_NaN.pth'))
                    sys.exit(1)

                # optimize only if train
                if phase == 'train':
                    
                    # update weight decay and learning rate according to their schedule
                    it = iter_per_epoch * epoch + it        # global training iteration
                    for i, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = lr_schedule[it]
                        if i==0:            # only the first param group is regularized
                            param_group['weight_decay'] = config.TRAIN.WEIGHT_DECAY

                    # backprop
                    loss.backward()

                    # update model params
                    grad_norm = None
                    if config.TRAIN.CLIP_GRAD is not None:
                        grad_norm = utils.clip_gradients(model, config.TRAIN.CLIP_GRAD)
                    optimizer.step()
                    
                    # logging
                    metric_loggers[phase].update(loss=loss)
                    metric_loggers[phase].update(mean_iou=miou)
                    metric_loggers[phase].update(grad_norm=grad_norm)
                    metric_loggers[phase].update(lr=optimizer.param_groups[0]["lr"])
                   
                if phase == 'val':
                    # logging
                    metric_loggers[phase].update(loss=loss)
                    metric_loggers[phase].update(mean_iou=miou)

        print(f'Averaged {phase} stats:', metric_loggers[phase])

    return ({k: meter.global_avg for k, meter in metric_loggers['train'].meters.items()},
            {k: meter.global_avg for k, meter in metric_loggers['val'].meters.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser('s3Cseg Training', parents=[get_args_parser()])
    args = parser.parse_args()

    config = configs.base._C.clone()                    # Base Configurations
    config.merge_from_list(arch_configs[args.arch]())   # Architecture defaults
    if args.config_file:
        config.merge_from_file(args.config_file)        # User Customizations
    config.freeze()

    os.makedirs(args.out_dir, exist_ok=True)

    main(args, config)
