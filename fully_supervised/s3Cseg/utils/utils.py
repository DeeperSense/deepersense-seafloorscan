# Modified by Hayat Rajani (hayatrajani@gmail.com)
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

"""
Misc functions.
Mostly copy-paste from torchvision references or other public repos like DETR and timm:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
https://github.com/rwightman/pytorch-image-models
"""


import os
import numpy
import torch


def fix_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return numpy.mean(norms)


def polyLR_scheduler(base_value, epochs, niter_per_ep, warmup_epochs=0):
    max_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    iters = numpy.arange(max_iters - warmup_iters)
    
    warmup_schedule = numpy.array([])
    if warmup_epochs > 0:
        warmup_schedule = numpy.linspace(0, base_value, warmup_iters)
    schedule = base_value * (1.0 - iters / max_iters) ** 0.9
    schedule = numpy.concatenate((warmup_schedule, schedule))
    
    assert len(schedule) == max_iters
    return schedule


def stepLR_scheduler(base_value, epochs, niter_per_ep, gamma=0.1, step=10):
    max_iters = epochs * niter_per_ep
    epochs = numpy.arange(epochs//step)
    schedule = numpy.repeat(base_value * gamma ** epochs, step*niter_per_ep)
    #schedule = numpy.pad(schedule, (0,max_iters-len(schedule)), 'edge')
    schedule = numpy.concatenate((
        schedule, numpy.repeat(schedule[-1], max_iters-len(schedule))
    ))
    assert len(schedule) == max_iters
    return schedule


def constLR_scheduler(base_value, epochs, niter_per_ep):
    max_iters = epochs * niter_per_ep
    schedule = numpy.repeat(base_value, max_iters)
    assert len(schedule) == max_iters
    return schedule


def resume_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # reload variables important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def load_pretrained_weights(model, pretrained_weights, mode='train'):
    assert mode.lower() in ('train','eval'), \
        "Invalid mode arguement. Must be 'train' or 'eval'"
    if not os.path.isfile(pretrained_weights):
        return
    print('Found pretrained weights at {}'.format(pretrained_weights))
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if mode.lower() == 'train':
        msg = model.encoder.load_state_dict(state_dict, strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights loaded with msg: {}'.format(msg))
