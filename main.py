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
from data import dataset

import torch.multiprocessing as M
import torch.distributed as D

from torch.utils.data.dataloader import DataLoader as DL
from torch.utils.data import DistributedSampler as DS
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import average_precision_score
from utils.cam_utils import update_pseudo_labels, calculate_pseudo_miou

from utils.loss import Loss
from utils import utils

from utils.logger import PseudomaskLogger

import wandb


encoder_choices = ('sima_mini', 'sima_tiny', 'sima_micro', 'sima_nano')
encoder_configs = {k: configs.models.__getattribute__(k) for k in encoder_choices}

decoder_choices = ('symmetric', 'atrous')
decoder_configs = {k: configs.models.__getattribute__(k) for k in decoder_choices}


def get_args_parser():
    parser = argparse.ArgumentParser('w-s3Tseg Distributed Training', add_help=False)

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
    parser.add_argument('--encoder', type=str, default='sima_tiny', choices=encoder_choices,
                        help='Type of encoder architecture.')
    parser.add_argument('--decoder', type=str, default='atrous', choices=decoder_choices,
                        help='Type of decoder architecture.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers per GPU.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of distinct images loaded per GPU.')
    parser.add_argument('--distr_url', type=str, default='env://',
                        help='''url used to set up distributed training;
                        see https://pytorch.org/docs/stable/distributed.html''')

    return parser


def main(args, config):
    M.set_start_method('spawn')

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if utils.is_main_process():
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
        wandb_logger = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project,
            dir=args.out_dir, config=config, resume=True,
        )

    # ================ preparing data ========================================
    
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
    
    classification_datasets = {x: dataset.ClassificationDatasetMSF(
            data_dir=os.path.join(args.data_dir,x), scales=config.TRAIN.SCALES,
            transform=data_transforms
        ) 
        for x in ('train','val')
    }

    cls_samplers = {x: DS(classification_datasets[x], shuffle=True) for x in ('train', 'val')}

    cls_data_loaders = {x: DL(classification_datasets[x], sampler=cls_samplers[x],
            batch_size=args.batch_size, num_workers=0, persistent_workers=False,
            drop_last=True, pin_memory=False, collate_fn=utils.my_collate
        )
        for x in ('train','val')
    }

    segmentation_datasets = {x: dataset.PseudoSegmentationDataset(
            data_dir=os.path.join(args.data_dir,x),
            pseudo_mask_dir=os.path.join(args.out_dir,'pseudo_masks',x),
            num_classes=config.DATA.NUM_CLASSES, transform=data_transforms
        )
        for x in ('train','val')
    }

    seg_samplers = {x: DS(segmentation_datasets[x], shuffle=True) for x in ('train', 'val')}

    seg_data_loaders = {x: DL(segmentation_datasets[x], sampler=seg_samplers[x],
            batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True, 
            drop_last=True, pin_memory=True, collate_fn=utils.my_collate
        )
        for x in ('train','val')
    }

    # ================ building encoder/decoder networks ================
    
    model = build_model(config)
    model.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # load pretrained weights, if any
    if args.load_weights:
        utils.load_pretrained_weights(model, args.load_weights, 'train')
    
    # watch gradients only for main process
    if utils.is_main_process():
        wandb_logger.watch(model)
    
    # set pseudomask logger
    log_pseudomask = PseudomaskLogger(args.out_dir, args.num_workers*2)

    print(f"Model built: a {args.encoder} network.")

    # ================ initializing optimizer ================

    params_groups = utils.get_params_groups(model.module)
    optimizer = torch.optim.AdamW(params_groups)

    # ================ initializing loss ================

    weights = 1/torch.FloatTensor(config.DATA.CLASS_FREQ)
    weights /= sum(weights)
    weights = weights.cuda(non_blocking=True)
    criterion = Loss(weight=weights)

    # ================ initializing schedulers ================

    ITER_PER_EPOCH = len(cls_data_loaders['train'])

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
    crf_stop_epoch = config.TRAIN.CRF_STOP_EPOCH or config.TRAIN.EPOCHS

    start_time = time.time()
    print(f"Starting training of w-s3Tseg ! from epoch {start_epoch}")

    if start_epoch < config.TRAIN.CRF_START_EPOCH:
        isClassificationOnly = True 
        for param in model.module.decoder.parameters():
            param.requires_grad = False
    else:       
        isClassificationOnly = False

    best_miou = 0.
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):

        # necessary to make shuffling work properly across multiple epochs
        cls_data_loaders['train'].sampler.set_epoch(epoch)
        cls_data_loaders['val'].sampler.set_epoch(epoch)
        seg_data_loaders['train'].sampler.set_epoch(epoch)
        seg_data_loaders['val'].sampler.set_epoch(epoch)
    
        # pseudo-mask generation/logging
        if ((epoch%config.TRAIN.CRF_FREQUNCY==0 and epoch>config.TRAIN.CRF_START_EPOCH)
                or epoch==config.TRAIN.CRF_START_EPOCH) and epoch<=crf_stop_epoch:

            if isClassificationOnly:
                isClassificationOnly = False
                for param in model.module.decoder.parameters():
                    param.requires_grad = True

            D.barrier()

            if utils.is_main_process():

                update_pseudo_labels(
                    model.module,
                    config.DATA.IMAGE_SIZE,
                    args.batch_size,
                    args.num_workers*2,
                    cls_data_loaders,
                    segmentation_datasets,       
                )

                mask_miou = calculate_pseudo_miou(
                    segmentation_datasets['val'], config.DATA.NUM_CLASSES,
                    args.batch_size, args.num_workers
                )

                wandb_logger.log({
                    'mask_iou': mask_miou
                }, commit=False)
                
                # log pseudo masks
                log_pseudomask(epoch)

            D.barrier()
        
        # train and validate classification/segmentation branch
        train_stats, val_stats = train_and_validate(model, seg_data_loaders, 
            optimizer, lr_schedule, criterion, epoch, ITER_PER_EPOCH, isClassificationOnly,
            config, args)
        
        # save checkpoint
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'criterion': criterion.state_dict(),
            'epoch': epoch + 1,
            'config': config,
            'args': args
        }
        utils.save_on_master(save_dict, os.path.join(args.out_dir, 'checkpoint.pth'))

        # log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch
        }
        if utils.is_main_process():
            with open(os.path.join(args.out_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if isClassificationOnly:
                wandb_logger.log({
                    'cls_loss': {
                        'train': train_stats['cls_loss'],
                        'val': val_stats['cls_loss']
                    },
                    'lr': train_stats['lr']
                })
            else:
                wandb_logger.log({
                    'cls_loss': {
                        'train': train_stats['cls_loss'],
                        'val': val_stats['cls_loss']
                    },
                    'seg_loss': {
                        'train': train_stats['seg_loss'],
                        'val': val_stats['seg_loss']
                    },
                    'mIOU': {
                        'train': train_stats['mean_iou'],
                        'val': val_stats['mean_iou']
                    },
                    'lr': train_stats['lr']
                })

        # save best model
        if not isClassificationOnly:
            miou = val_stats['mean_iou']
            if miou > best_miou:
                best_miou = miou
                utils.save_on_master(model.module.state_dict(),
                                        os.path.join(args.out_dir, 'best_model.pth'))
        if (epoch+1) % 50 == 0:
            utils.save_on_master(model.module.state_dict(),
                                    os.path.join(args.out_dir, '%d_model.pth'%(epoch+1)))
    utils.save_on_master(model.module.state_dict(), os.path.join(args.out_dir, 'fin_model.pth'))

    if utils.is_main_process():
        wandb_logger.finish()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_and_validate(model, dataset_loader, optimizer, lr_schedule, criterion,
                        epoch, iter_per_epoch, isClassificationOnly, config, args):
    metric_loggers = {}
    headers = {}
    
    metric_loggers['train'] = MetricLogger(delimiter="  ")
    headers['train'] = 'Epoch: [{}/{}]'.format(epoch, config.TRAIN.EPOCHS)

    metric_loggers['val'] = MetricLogger(delimiter="  ")
    headers['val'] = 'Validation:'

    for phase in ('train', 'val'):

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for it, batch in enumerate(metric_loggers[phase].log_every(
                                        dataset_loader[phase], 1, headers[phase])):

            img = batch["img"].cuda(non_blocking=True)
            cls_label = batch["label"].cuda(non_blocking=True)
            seg_label = batch["pseudo_mask"].cuda(non_blocking=True).long()

            # zero parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            
                out = model(img, mode='train', isClassificationOnly=isClassificationOnly)
                cls_logits, seg_logits = out["cls"], out["seg"]
                
                cls_loss, seg_loss = criterion(cls_logits, seg_logits, cls_label, seg_label)
                loss = cls_loss + seg_loss

                if not isClassificationOnly:
                    seg_preds = torch.argmax(seg_logits.detach(), dim=1)
                    seg_preds = one_hot(seg_preds, seg_logits.shape[1]).permute(0,3,1,2)    # (B,K,H,W)
                    seg_label = one_hot(seg_label, seg_logits.shape[1]).permute(0,3,1,2)
                
                    inter = torch.sum(torch.logical_and(seg_preds, seg_label), dim=(2,3))   # B x K

                    union = torch.sum(torch.logical_or(seg_preds, seg_label), dim=(2,3))    # B x K
                    miou = torch.mean((inter+1e-6)/(union+1e-6), dim=(0,1))

                cls_label = cls_label.detach().cpu().numpy()
                cls_preds = torch.sigmoid(cls_logits.detach()).cpu().numpy()
                ap_score = average_precision_score(y_true=cls_label, y_score=cls_preds,
                                                        average='samples')

                # writing logs on a NaN to debug
                if not math.isfinite(loss.item()):
                    print('Loss is {}, stopping training'.format(loss.item()), force=True)
                    save_dict = {
                        'student': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'criterion': criterion.state_dict(),
                        'epoch': epoch + 1,
                        'config': config,
                        'args': args
                    }
                    utils.save_on_master(save_dict, os.path.join(args.out_dir, 'ckpt_NaN.pth'))
                    sys.exit(1)

                if phase == 'train':
                    # update weight decay and learning rate according to their schedule
                    it = iter_per_epoch * epoch + it    # global training iteration
                    for i, param_group in enumerate(optimizer.param_groups):
                        param_group['lr'] = 10*lr_schedule[it] if i<2 else lr_schedule[it]
                        if i%2==0:  # only the first param group in the pair is regularized
                            param_group['weight_decay'] = config.TRAIN.WEIGHT_DECAY

                    loss.backward()     # backward pass

                    grad_norm = None
                    if config.TRAIN.CLIP_GRAD is not None:
                        grad_norm = utils.clip_gradients(model, config.TRAIN.CLIP_GRAD)
                    
                    optimizer.step()    # update params

                    torch.cuda.synchronize()
                    metric_loggers[phase].update(total_loss=loss)
                    metric_loggers[phase].update(cls_loss=cls_loss)
                    if not isClassificationOnly:
                        metric_loggers[phase].update(seg_loss=seg_loss)
                        metric_loggers[phase].update(mean_iou=miou)
                    metric_loggers[phase].update(ap_score=ap_score)
                    metric_loggers[phase].update(grad_norm=grad_norm)
                    metric_loggers[phase].update(lr=optimizer.param_groups[0]["lr"])
                
                if phase == 'val':
                    torch.cuda.synchronize()
                    metric_loggers[phase].update(total_loss=loss)
                    metric_loggers[phase].update(cls_loss=cls_loss)
                    if not isClassificationOnly:
                        metric_loggers[phase].update(seg_loss=seg_loss)
                        metric_loggers[phase].update(mean_iou=miou)
                    metric_loggers[phase].update(ap_score=ap_score)
        
        # gather the stats from all processes
        metric_loggers[phase].synchronize_between_processes()
        print(f'Averaged {phase} stats:', metric_loggers[phase])

    return ({k: meter.global_avg for k, meter in metric_loggers['train'].meters.items()},
            {k: meter.global_avg for k, meter in metric_loggers['val'].meters.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser('w-s3Tseg Distributed Training', parents=[get_args_parser()])
    args = parser.parse_args()

    config = configs.base._C.clone()                            # Base Configurations
    config.merge_from_list(encoder_configs[args.encoder]())     # Architecture defaults
    config.merge_from_list(decoder_configs[args.decoder]())
    if args.config_file:
        config.merge_from_file(args.config_file)                # User Customizations

    config.freeze()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"config.yaml"), "w") as f:
        f.write(config.dump())

    main(args, config)