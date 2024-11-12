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

import copy

import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, create

__all__ = ['CoDINOHead']


@register
class CoDINOHead(nn.Layer):

    def __init__(self,
                 transformer='CoDINOTransformer',
                 query_head='DINOHead',
                 rpn_head='RPNHead',
                 roi_head='RoIHead',
                 bbox_head="CoATSSHead",
                 rpn_loss_weight=12.,
                 roi_loss_weight=12.):
        super().__init__()
        self.transformer = transformer
        self.query_head = query_head
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.bbox_head = bbox_head
        self.rpn_loss_weight = rpn_loss_weight
        self.roi_loss_weight = roi_loss_weight

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # transformer
        transformer = create(cfg['transformer'], **kwargs)
        query_head = create(cfg['query_head'])
        kwargs['input_shape'] = transformer.aux_out_shape
        rpn_head = create(cfg['rpn_head'], **kwargs)
        roi_head = create(cfg['roi_head'], **kwargs)
        # bbox_head = create(cfg['bbox_head'])

        return {
            'transformer': transformer,
            'query_head': query_head,
            'rpn_head': rpn_head,
            'roi_head': roi_head,
            # 'bbox_head': bbox_head,
        }

    def forward(self, feats, pad_mask=None, gt_meta=None):
        transformer_outs = self.transformer(feats, pad_mask, gt_meta)
        if self.training:
            losses = {}

            detr_loss = self.query_head(transformer_outs[:-1], None, gt_meta)
            losses.update(detr_loss)

            rcnn_gt_meta = self.prepare_rcnn_gt_meta(gt_meta)
            rois, rois_num, rpn_loss = self.rpn_head(transformer_outs[-1],
                                                     rcnn_gt_meta)
            losses.update({
                k: v * self.rpn_loss_weight
                for k, v in rpn_loss.items()
            })
            roi_loss, _ = self.roi_head(transformer_outs[-1], rois, rois_num,
                                        rcnn_gt_meta)
            losses.update({
                k: v * self.roi_loss_weight
                for k, v in roi_loss.items()
            })

            return losses
        else:
            detr_preds = self.query_head(transformer_outs[:-1], None)

            rois, rois_num, _ = self.rpn_head(transformer_outs[-1], gt_meta)

            return detr_preds



    @staticmethod
    def prepare_rcnn_gt_meta(gt_meta):
        batch_size, _, height, width = gt_meta['image'].shape
        rcnn_gt_meta = {k: v for k, v in gt_meta.items() if k != 'gt_bbox'}
        rcnn_gt_meta['gt_bbox'] = []
        for i in range(batch_size):
            if gt_meta['gt_bbox'][i].shape[0] > 0:
                rcnn_gt_meta['gt_bbox'].append(gt_meta['gt_bbox'][i].clone())
                # rescale
                rcnn_gt_meta['gt_bbox'][i][:, 0::2] *= width
                rcnn_gt_meta['gt_bbox'][i][:, 1::2] *= height
                # xywh2xyxy
                rcnn_gt_meta['gt_bbox'][
                    i][:, :2] -= rcnn_gt_meta['gt_bbox'][i][:, 2:] * 0.5
                rcnn_gt_meta['gt_bbox'][i][:, 2:] += rcnn_gt_meta['gt_bbox'][
                    i][:, :2]
            else:
                # shape: [0, 4]
                rcnn_gt_meta['gt_bbox'].append(gt_meta['gt_bbox'][i])
        return rcnn_gt_meta
