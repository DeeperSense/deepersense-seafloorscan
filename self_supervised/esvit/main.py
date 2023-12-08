# Modified by Hayat Rajani (hayat.rajani@udg.edu)
#
# Modified by Chunyuan Li  (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

from models.build import build_model
from utils.loss import EsViTLoss

from data.augmentations import DataAugmentation
from data.dataset import SonarDataset

from torch.utils.data.dataloader import DataLoader as DL
from torch.utils.data import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.logger import MetricLogger
from sklearn.metrics.pairwise import cosine_similarity

from utils import utils
import wandb


arch_choices = ['cswin_mini', 'lsda_mini', 'sima_mini','sima_tiny','sima_micro','sima_nano']
arch_funcs = {k: configs.models.__getattribute__(k) for k in arch_choices}


def get_args_parser():
    parser = argparse.ArgumentParser('EsViT Distributed Training', add_help=False)

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
    parser.add_argument('--arch', type=str, default='sima_mini', choices=arch_choices,
                        help='Name of architecture to train')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of distinct images loaded per GPU.')
    parser.add_argument('--use_fp16', type=bool, default=False,
                        help='''Whether or not to use half precision for training.
                        Improves training time and memory requirements,
                        but can provoke instability and slight decay of performance.
                        We recommend disabling mixed precision if the loss is unstable,
                        if reducing the patch size or if training with bigger ViTs.''')
    parser.add_argument('--distr_url', type=str, default='env://',
                        help='''url used to set up distributed training;
                        see https://pytorch.org/docs/stable/distributed.html''')

    return parser


def main(args, config):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    
    if utils.is_main_process():
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
        wandb_logger = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            dir=args.out_dir, config=config, resume=True,
        )

    # ================ preparing data ================

    data_transforms = DataAugmentation(
        config.AUG.MULTI_CROP.GLOBAL_CROPS_SIZE,
        config.AUG.MULTI_CROP.GLOBAL_CROPS_SCALE,
        config.AUG.MULTI_CROP.LOCAL_CROPS_SIZE,
        config.AUG.MULTI_CROP.LOCAL_CROPS_SCALE,
        config.AUG.MULTI_CROP.LOCAL_CROPS_NUMBER,
        config.DATA.MEAN, config.DATA.STD
    )

    dataset = SonarDataset(args.data_dir, data_transforms)
    sampler = DS(dataset, shuffle=True)
    data_loader = DL(dataset, sampler=sampler,
                     batch_size=args.batch_size, num_workers=args.num_workers,
                     shuffle=False, pin_memory=True, drop_last=True)

    # ================ building student/teacher networks ================

    student, teacher = build_model(config)
    student.cuda(); teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # we need DDP wrapper to have synchro batch norms working
        teacher = DDP(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = DDP(student, device_ids=[args.gpu])

    # load pretrained weights, if any
    if args.load_weights:
        utils.load_pretrained_weights(student, args.load_weights, 'train')
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    # watch gradients only for main process
    if utils.is_main_process():
        wandb_logger.watch(student)
    
    print(f'Student and Teacher are built: they are both {args.arch} network.')

    # ================ initializing optimizer ================

    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ================ initializing loss ================

    criterion = EsViTLoss(
        config.MODEL.HEAD.OUT_DIM,
        config.AUG.MULTI_CROP.LOCAL_CROPS_NUMBER + 2,  # total crops = n local + 2 global
        config.TRAIN.WARMUP_TEACHER_TEMP, config.TRAIN.TEACHER_TEMP,
        config.TRAIN.WARMUP_TEACHER_TEMP_EPOCHS, config.TRAIN.EPOCHS,
        debug=config.TRAIN.DEBUG
    ).cuda()

    # ================ initializing schedulers ================

    ITER_PER_EPOCH = len(data_loader)

    lr_schedule = utils.cosine_scheduler(
        config.TRAIN.LR * args.batch_size * utils.get_world_size() / 256.,
        config.TRAIN.MIN_LR, config.TRAIN.EPOCHS, ITER_PER_EPOCH,
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS
    )
    wt_schedule = utils.cosine_scheduler(
        config.TRAIN.WEIGHT_DECAY, config.TRAIN.WEIGHT_DECAY_END,
        config.TRAIN.EPOCHS, ITER_PER_EPOCH
    )
    momentum_schedule = utils.cosine_scheduler(
        config.TRAIN.MOMENTUM_TEACHER, 1, config.TRAIN.EPOCHS, ITER_PER_EPOCH
    )

    print(f'Loss, optimizer and schedulers ready.')

    # ================ training and logging ================

    to_restore = {'epoch': 0}
    if args.load_checkpoint:
        utils.resume_from_checkpoint(args.load_checkpoint, run_variables=to_restore,
                                      student=student, teacher=teacher,
                                      optimizer=optimizer, criterion=criterion,
                                      fp16_scaler=fp16_scaler)
    start_epoch = to_restore['epoch']

    start_time = time.time()
    print(f'Starting training of EsViT ! from epoch {start_epoch}')

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        # necessary to make shuffling work properly across multiple epochs
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(student, teacher, teacher_without_ddp,
                                      data_loader, criterion, optimizer,
                                      lr_schedule, wt_schedule, momentum_schedule,
                                      epoch, ITER_PER_EPOCH, fp16_scaler, config)

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch + 1,
            'config': config,
            'args': args
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.out_dir, 'checkpoint.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils.is_main_process():
            with open(os.path.join(args.out_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
            if config.TRAIN.DEBUG:
                wandb_logger.log({
                    'grad_norm': train_stats['grad_norm'],
                    'cos_sim_s': train_stats['cos_sim_s'],
                    'cos_sim_t': train_stats['cos_sim_t'],
                    'entropy': train_stats['H'],
                    'kl-div': train_stats['KL'],
                }, commit=False)
            wandb_logger.log({
                'wt': train_stats['wt'], 'lr': train_stats['lr'],
                'loss': train_stats['loss']
            }, commit=True)
        
        if (epoch+1) in (100,200,250,300,400):
            utils.save_on_master(student.module.backbone.state_dict(),
                                os.path.join(args.out_dir, f'student_{epoch+1}.pth'))
            utils.save_on_master(teacher.backbone.state_dict(),
                                os.path.join(args.out_dir, f'teacher_{epoch+1}.pth'))
            wandb_logger.alert(
                title=f'Milestone epoch {epoch+1}',
                text=f'Milestone models saved in {args.out_dir} \
                        \nRun evaluation script to gauge performance.',
                level=wandb.AlertLevel.INFO
            )

    utils.save_on_master(student.module.backbone.state_dict(),
                        os.path.join(args.out_dir, 'student.pth'))
    utils.save_on_master(teacher.backbone.state_dict(),
                        os.path.join(args.out_dir, 'teacher.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, data_loader,
                    criterion, optimizer, lr_schedule, wt_schedule, momentum_schedule,
                    epoch, iter_per_epoch, fp16_scaler, config):
    
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Epoch: [{}/{}]'.format(epoch, config.TRAIN.EPOCHS)

    for it, images in enumerate(metric_logger.log_every(data_loader, 1, header)):
        
        # update weight decay and learning rate according to their schedule
        it = iter_per_epoch * epoch + it        # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = wt_schedule[it]

        # move images to gpu; list -> batch x channel x width x height
        images = [im.cuda(non_blocking=True) for im in images]

        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = criterion(student_output, teacher_output, epoch)
        
        if config.TRAIN.DEBUG:
            sim = cosine_similarity(student_output[0].detach().cpu())
            sim = ((sim.sum(1)-1)/(sim.shape[1]-1)).mean()
            student_sim = sim.item()
            
            sim = cosine_similarity(teacher_output[0].detach().cpu())
            sim = ((sim.sum(1)-1)/(sim.shape[1]-1)).mean()
            teacher_sim = sim.item()

            loss, H, KL = loss

        # writing logs on a NaN to debug
        if not math.isfinite(loss.item()):
            print('Loss is {}, stopping training'.format(loss.item()), force=True)
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'criterion': criterion.state_dict(),
                'epoch': epoch + 1,
                'config': config,
                'args': args
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.out_dir, 'ckpt_NaN.pth'))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        grad_norm = None
        if fp16_scaler is None:
            loss.backward()
            torch.cuda.synchronize()
            if config.TRAIN.CLIP_GRAD is not None:
                grad_norm = utils.clip_gradients(student, config.TRAIN.CLIP_GRAD)
            utils.cancel_gradients_last_layer(epoch, student, config.MODEL.HEAD.FREEZE_LAST_LAYER)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            torch.cuda.synchronize()
            if config.TRAIN.CLIP_GRAD is not None:
                fp16_scaler.unscale_(optimizer)
                grad_norm = utils.clip_gradients(student, config.TRAIN.CLIP_GRAD)
            utils.cancel_gradients_last_layer(epoch, student, config.MODEL.HEAD.FREEZE_LAST_LAYER)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(wt=optimizer.param_groups[0]['weight_decay'])
            
        if config.TRAIN.DEBUG:
            metric_logger.update(grad_norm=grad_norm)
                
            metric_logger.update(cos_sim_s=student_sim)
            metric_logger.update(cos_sim_t=teacher_sim)

            metric_logger.update(KL=KL)
            metric_logger.update(H=H)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EsViT Distributed Training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    config = configs.base._C.clone()                    # Base Configurations
    config.merge_from_list(arch_funcs[args.arch]())     # Architecture defaults
    if args.config_file:
        config.merge_from_file(args.config_file)        # User Customizations
    config.freeze()

    os.makedirs(args.out_dir, exist_ok=True)
    
    main(args, config)
