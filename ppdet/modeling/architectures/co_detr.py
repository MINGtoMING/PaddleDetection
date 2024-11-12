# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ['CoDETR']


@register
class CoDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 neck=None,
                 head='CoDETRHead',
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(CoDETR, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        # neck
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None
        # head
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            "neck": neck,
            "head": head,
        }

    def _forward(self):
        # backbone
        body_feats = self.backbone(self.inputs)
        # neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)
        # head
        if self.training:
            pad_mask = self.inputs.get('pad_mask', None)
            losses = self.head(body_feats, pad_mask, self.inputs)
            losses.update({
                'loss': paddle.add_n(
                    [v for k, v in losses.items() if 'log' not in k])
            })
            return losses
        else:
            preds = self.head(body_feats, None, self.inputs)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'],
                    self.inputs['scale_factor'],
                    self.inputs['image'].shape[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
