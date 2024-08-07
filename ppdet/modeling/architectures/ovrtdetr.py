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
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['OVRTDETR']


@register
class OVRTDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['exclude_post_process']

    def __init__(self,
                 image_backbone='ResNet',
                 text_backbone='CLIPFromPretrained',
                 transformer='OVRTDETRTransformer',
                 detr_head='DINOHead',
                 neck='OVHybridEncoder',
                 post_process='OVRTDETRPostProcess',
                 exclude_post_process=False):
        super(OVRTDETR, self).__init__()
        self.image_backbone = image_backbone
        self.text_backbone = text_backbone
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # image backbone
        image_backbone = create(cfg['image_backbone'])
        # text backbone
        text_backbone = create(cfg['text_backbone'])
        # neck
        kwargs = {'input_shape': image_backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg['neck'] else None
        # transformer
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': image_backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'image_backbone': image_backbone,
            'text_backbone': text_backbone,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck,
        }

    def _forward(self):
        # Backbone
        image_feats = self.image_backbone(self.inputs)
        text_feats = self.text_backbone(self.inputs)

        # Neck
        image_feats = self.neck(image_feats, text_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        text_token_mask = self.inputs.get('text_token_mask', None)
        out_transformer = self.transformer(image_feats, text_feats, pad_mask,
                                           text_token_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, image_feats,
                                         self.inputs)
            detr_losses.update({
                'loss':
                paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, image_feats)
            if self.exclude_post_process:
                bbox, logit, _ = preds
                output = {'bbox': bbox, 'logit': logit}
                return output
            else:
                bbox, bbox_num, _ = self.post_process(
                    preds, self.inputs['im_shape'],
                    self.inputs['scale_factor'],
                    self.inputs['image'][2:].shape)
                output = {'bbox': bbox, 'bbox_num': bbox_num}
                return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
