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

"""
Adapted from EsViT
https://github.com/microsoft/esvit/blob/main/utils.py
"""


import torch
import torch.nn as nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs.
    Hence we do several forward passes = number of different resolutions used.
    We then concatenate all the output features and run the head forward on
    these concatenated features.
    """

    def __init__(self, backbone, view_head, region_head):
        super().__init__()

        # disable layers dedicated to label classification / dense prediction
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.view_head = view_head
        self.region_head = region_head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _cls, _tok = self.backbone(torch.cat(x[start_idx: end_idx]))
            B, N, C = _tok.shape
            if start_idx == 0:
                cls_out = _cls
                tok_out = _tok.reshape(B * N, C)
                num_tok = [N]
            else:
                cls_out = torch.cat((cls_out, _cls))
                tok_out = torch.cat((tok_out, _tok.reshape(B * N, C)))
                num_tok.append(N)
            start_idx = end_idx

        # Run the head forward on the concatenated features
        return self.view_head(cls_out), self.region_head(tok_out), tok_out, num_tok
