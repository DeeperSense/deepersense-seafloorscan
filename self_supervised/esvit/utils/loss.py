# Modified by Hayat Rajani (hayat.rajani@udg.edu)
#
# Modified by Chunyuan Li (chunyl@microsoft.com)
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
Adapted from EsViT
https://github.com/microsoft/esvit/blob/main/main_esvit.py
"""


import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as D


class EsViTLoss(nn.Module):
    """View-level and region-level cross-entropy between softmax outputs of the
    teacher and student networks as described in the EsViT paper.
    Further computes teacher entropy and KL-divergence between student and teacher,
    with and without teacher centering/sharpening, to check for collapse.
    Args:
        out_dim: Projection head output dimensionality
        ncrops: Total number of crops (global+local)
        warmup_teacher_temp: Initial value for the teacher temperature: 0.04 works
            well in most cases. Try decreasing if training loss does not decrease.
        teacher_temp: Final value (after linear warmup) of the teacher temperature.
        warmup_teacher_temp_epochs: Number of warmup epochs for teacher temperature.
        nepochs: Total number of epochs for training.
        debug: If true, reports target entropy and KL-divergence to check for collapse.
        student_temp: Value of student temperature.
        center_momentum: EMA parameter for teacher centering.
    """

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, debug=False,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_grid", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        self.teacher_temp_schedule = numpy.concatenate((
            numpy.linspace(warmup_teacher_temp,
                           teacher_temp, warmup_teacher_temp_epochs),
            numpy.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        # calculate teacher entropy and KL divergence to check for collapse
        self.debug = debug
        self.H = lambda p: torch.sum(-F.softmax(p, dim=-1)*F.log_softmax(p, dim=-1), dim=-1)
        self.KL = lambda p, q: torch.sum(F.softmax(p, dim=-1)*(F.log_softmax(p, dim=-1)-F.log_softmax(q, dim=-1)), dim=-1)

    def forward(self, student_output, teacher_output, epoch):
        s_cls_out, s_region_out, s_fea, s_npatch = student_output
        t_cls_out, t_region_out, t_fea, t_npatch = teacher_output

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        t_cls = F.softmax((t_cls_out - self.center) / temp, dim=-1)
        t_cls = t_cls.detach().chunk(2)

        t_region = F.softmax((t_region_out - self.center_grid) / temp, dim=-1)
        t_region = t_region.detach().chunk(2)
        t_fea = t_fea.chunk(2)

        N = t_npatch[0]  # num of patches in the first view
        B = t_region[0].shape[0] // N  # batch size,

        # student sharpening
        s_cls = s_cls_out / self.student_temp
        s_cls = s_cls.chunk(self.ncrops)

        s_region = s_region_out / self.student_temp
        s_split_size = [s_npatch[0]] * 2 + [s_npatch[-1]] * (self.ncrops - 2)

        s_split_size_bs = [i * B for i in s_split_size]

        s_region = torch.split(s_region, s_split_size_bs, dim=0)
        s_fea = torch.split(s_fea, s_split_size_bs, dim=0)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(t_cls):
            for v in range(len(s_cls)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # view-level prediction loss
                loss = 0.5 * torch.sum(-q * F.log_softmax(s_cls[v], dim=-1), dim=-1) 
                
                # region-level prediction loss
                # B x T_s x K, B x T_s x P
                s_region_cur = s_region[v].view(B, s_split_size[v], -1)
                s_fea_cur = s_fea[v].view(B, s_split_size[v], -1)
                # B x T_t x K, B x T_t x P
                t_region_cur = t_region[iq].view(B, N, -1)
                t_fea_cur = t_fea[iq].view(B, N, -1)

                # similarity matrix between two sets of region features
                # B x T_s x T_t
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1),
                                                 F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1))
                # collect the argmax index in teacher for a given student feature
                # B x T_s
                region_sim_ind = region_sim_matrix.max(dim=2)[1]

                # B x T_s x K (index matrix: B, T_s, 1)
                t_indexed_region = torch.gather(t_region_cur, 1,
                                                region_sim_ind.unsqueeze(2).expand(
                                                    -1, -1, t_region_cur.size(2)))
                # B x T_s x K --> B
                loss_grid = torch.sum(-t_indexed_region * F.log_softmax(s_region_cur, dim=-1),
                                      dim=[-1]).mean(-1)

                loss += 0.5 * loss_grid
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        if self.debug:
            total_loss = (total_loss,
                *self.entropy(t_cls_out.detach().chunk(2), s_cls, epoch))   
        self.update_center(t_cls_out, t_region_out)

        return total_loss

    @torch.no_grad()
    def entropy(self, P, Q, epoch):
        H, KL = 0., 0.
        n_loss_terms = 0
        
        temp = self.teacher_temp_schedule[epoch]

        for i in range(len(P)):
            for j in range(len(Q)):
                if i==j:
                    continue
                
                H += self.H((P[i]-self.center)/temp).mean()
                KL += self.KL((P[i]-self.center)/temp, Q[j]).mean()
                
                n_loss_terms += 1
        
        H /= n_loss_terms
        KL /= n_loss_terms

        return H, KL

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_grid_output):
        # view-level center update
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        D.all_reduce(batch_center)
        batch_center /= (len(teacher_output) * D.get_world_size())
        
        # region-level center update
        batch_grid_center = torch.sum(teacher_grid_output, dim=0, keepdim=True)
        D.all_reduce(batch_grid_center)
        batch_grid_center /= (len(teacher_grid_output) * D.get_world_size())
        
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)
