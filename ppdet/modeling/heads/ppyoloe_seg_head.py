# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ppdet.modeling.backbones.cspresnet import ConvBNLayer, RepVggBlock
from ppdet.modeling.layers import MultiClassNMS
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ..assigners.utils import generate_anchors_for_grid_cell
from ..bbox_utils import batch_distance2bbox, bbox_area
from ..initializer import bias_init_with_prob, constant_, normal_
from ..losses import GIoULoss

__all__ = ['PPYOLOESegHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish', attn_conv='convbn'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        if attn_conv == 'convbn':
            self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)
        elif attn_conv == 'repvgg':
            self.conv = RepVggBlock(feat_channels, feat_channels, act=act)
        else:
            self.conv = None
        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        if self.conv:
            return self.conv(feat * weight)
        else:
            return feat * weight


@register
class PPYOLOESegHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms',
        'exclude_post_process', 'use_shared_conv', 'for_distill'
    ]
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 reg_range=None,
                 num_protos=32,
                 dim_protonet=256,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='SegATSSAssigner',
                 assigner='SegTaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'mask': 2.5,
                 },
                 trt=False,
                 attn_conv='convbn',
                 exclude_nms=False,
                 exclude_post_process=False,
                 use_shared_conv=True,
                 for_distill=False):
        super(PPYOLOESegHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.num_protos = num_protos
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        if isinstance(self.nms, MultiClassNMS) and trt:
            self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        self.use_shared_conv = use_shared_conv
        self.for_distill = for_distill
        self.is_teacher = False

        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_coeff = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_reg.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
            self.stem_coeff.append(ESEAttn(in_c, act=act, attn_conv=attn_conv))
        # pred head
        self.pred_cls = nn.LayerList()
        self.pred_reg = nn.LayerList()
        self.pred_coeff = nn.LayerList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2D(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2D(
                    in_c, 4 * self.reg_channels, 3, padding=1))
            self.pred_coeff.append(
                nn.Conv2D(
                    in_c, self.num_protos, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_channels, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True

        # protonet
        self.protonet = nn.Sequential(
            ConvBNLayer(
                self.in_channels[-1], dim_protonet, 3, padding=1, act=act),
            nn.Conv2DTranspose(
                dim_protonet, dim_protonet, 2, 2),
            ConvBNLayer(
                dim_protonet, dim_protonet, 3, padding=1, act=act),
            ConvBNLayer(
                dim_protonet, self.num_protos, 1, act=act))

        self._init_weights()

        if self.for_distill:
            self.distill_pairs = {}

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        bias_coeff = bias_init_with_prob(0.01)
        for cls_, reg_, coeff_ in zip(
                self.pred_cls, self.pred_reg, self.pred_coeff):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)
            constant_(coeff_.weight)
            constant_(coeff_.bias, bias_coeff)

        proj = paddle.linspace(self.reg_range[0], self.reg_range[1] - 1,
                               self.reg_channels).reshape(
            [1, self.reg_channels, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train(self, feats, targets, aux_pred=None):
        prototypes = self.protonet(feats[-1])

        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](
                self.stem_reg[i](feat, avg_feat))
            mask_coeff = self.pred_coeff[i](
                self.stem_coeff[i](feat, avg_feat) + feat)
            # cls, reg and coeff
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
            mask_coeff_list.append(mask_coeff.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=1)

        if targets.get('is_teacher', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, \
                pred_dfls, mask_coeff_list, prototypes

        if targets.get('get_data', False):
            pred_deltas, pred_dfls = self._bbox_decode_fake(reg_distri_list)
            return cls_score_list, pred_deltas * stride_tensor, \
                pred_dfls, mask_coeff_list, prototypes

        return self.get_loss([
            cls_score_list, reg_distri_list, mask_coeff_list, prototypes,
            anchors, anchor_points, num_anchors_list, stride_tensor
        ], targets, aux_pred)

    def _generate_anchors(self, feats=None, dtype='float32'):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = paddle.arange(end=w) + self.grid_cell_offset
            shift_y = paddle.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = paddle.meshgrid(shift_y, shift_x)
            anchor_point = paddle.cast(
                paddle.stack(
                    [shift_x, shift_y], axis=-1), dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(paddle.full([h * w, 1], stride, dtype=dtype))
        anchor_points = paddle.concat(anchor_points)
        stride_tensor = paddle.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_channels, l]).transpose([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(
                    reg_dist, axis=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, axis=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        if self.use_shared_conv:
            reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        else:
            reg_dist_list = paddle.concat(reg_dist_list, axis=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None, aux_pred=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets, aux_pred)
        else:
            if targets is not None:
                # only for semi-det
                self.is_teacher = targets.get('is_teacher', False)
                if self.is_teacher:
                    return self.forward_train(feats, targets, aux_pred=None)
                else:
                    return self.forward_eval(feats)

            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        loss = F.binary_cross_entropy(
            score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy(
            pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox_decode_fake(self, pred_dist):
        _, l, _ = get_static_shape(pred_dist)
        pred_dist_dfl = F.softmax(
            pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist_dfl.transpose([0, 3, 1, 2
                                                            ])).squeeze(1)
        return pred_dist, pred_dist_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(self.reg_range[0],
                                                self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = paddle.cast(target.floor(), 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left - lower_bound,
            reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right - lower_bound,
            reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)

        if self.for_distill:
            # only used for LD main_kd distill
            self.distill_pairs['mask_positive_select'] = mask_positive

        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.astype('int32').unsqueeze(-1).tile(
                [1, 1, 4]).astype('bool')
            pred_bboxes_pos = paddle.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = paddle.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = paddle.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).astype('int32').tile(
                [1, 1, self.reg_channels * 4]).astype('bool')
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_channels])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos,
                                     self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
            if self.for_distill:
                self.distill_pairs['pred_bboxes_pos'] = pred_bboxes_pos
                self.distill_pairs['pred_dist_pos'] = pred_dist_pos
                self.distill_pairs['bbox_weight'] = bbox_weight
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    @staticmethod
    def _crop_mask(masks, boxes):
        _, mask_h, mask_w = masks.shape
        x1, y1, x2, y2 = paddle.chunk(boxes.unsqueeze(-1), 4, 1)
        rows = paddle.arange(mask_w, dtype=masks.dtype).unsqueeze(0).unsqueeze(0)
        cols = paddle.arange(mask_h, dtype=masks.dtype).unsqueeze(0).unsqueeze(-1)
        return masks * ((rows >= x1) * (rows < x2) * (cols >= y1) * (cols < y2))

    def _mask_loss(self, pred_coeffs, prototypes, assigned_labels, assigned_bboxes,
                   assigned_gt_index, gt_masks, assigned_scores_sum, stride_tensor):
        loss_mask = paddle.zeros([1])
        assigned_bboxes *= stride_tensor
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            batch_size, num_max_boxes = gt_masks.shape[:2]
            batch_ind = paddle.arange(end=batch_size).unsqueeze(-1)
            assigned_gt_index -= batch_ind * num_max_boxes

            proto_h, proto_w = prototypes.shape[-2:]
            gt_mask_h, gt_mask_w = gt_masks.shape[-2:]
            scale_y, scale_x = proto_h / gt_mask_h, proto_w / gt_mask_w
            mask_scale_factor = [scale_y, scale_x]
            bbox_scale_factor = paddle.to_tensor(
                [scale_x, scale_y, scale_x, scale_y]).unsqueeze(0)

            for i in range(batch_size):
                if mask_positive[i].sum() > 0:
                    coeff_mask = mask_positive[i].unsqueeze(-1).tile([1, self.num_protos])
                    pos_pred_coeff = paddle.masked_select(
                        pred_coeffs[i], coeff_mask).reshape([-1, self.num_protos])
                    pos_pred_mask = paddle.einsum(
                        'np,phw->nhw', pos_pred_coeff, prototypes[i])
                    pos_assigned_gt_index = paddle.masked_select(
                        assigned_gt_index[i], mask_positive[i])
                    pos_assigned_mask = paddle.gather(
                        gt_masks[i], pos_assigned_gt_index, axis=0)
                    pos_assigned_mask = F.interpolate(
                        pos_assigned_mask.unsqueeze(0),
                        scale_factor=mask_scale_factor,
                        mode='nearest',
                        align_corners=False).squeeze(0)
                    bbox_mask = mask_positive[i].unsqueeze(-1).tile([1, 4])
                    pos_assigned_bbox = paddle.masked_select(
                        assigned_bboxes[i], bbox_mask).reshape([-1, 4])
                    pos_assigned_bbox *= bbox_scale_factor
                    single_loss_mask = F.sigmoid_focal_loss(
                        pos_pred_mask, pos_assigned_mask, reduction='none')
                    single_loss_mask = self._crop_mask(single_loss_mask, pos_assigned_bbox)
                    area = bbox_area(pos_assigned_bbox)
                    single_loss_mask = (single_loss_mask.sum(axis=[1, 2]) / area).sum()
                    loss_mask += single_loss_mask
            loss_mask = loss_mask / num_pos

        return loss_mask

    def get_loss(self, head_outs, gt_meta, aux_pred=None):
        pred_scores, pred_distri, pred_coeffs, prototypes, \
            anchors, anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        if aux_pred is not None:
            pred_scores_aux = aux_pred[0]
            pred_bboxes_aux = self._bbox_decode(anchor_points_s, aux_pred[1])

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_masks = gt_meta['gt_segm']
        gt_masks = gt_masks.astype(paddle.float32)
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor,
                    return_index=True)
            alpha_l = 0.25
        else:
            if self.sm_use:
                # only used in smalldet of PPYOLOE-SOD model
                assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
                    self.assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        stride_tensor,
                        gt_labels,
                        gt_bboxes,
                        pad_gt_mask,
                        bg_index=self.num_classes,
                        return_index=True)
            else:
                if aux_pred is None:
                    if not hasattr(self, "assigned_labels"):
                        assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
                            self.assigner(
                                pred_scores.detach(),
                                pred_bboxes.detach() * stride_tensor,
                                anchor_points,
                                num_anchors_list,
                                gt_labels,
                                gt_bboxes,
                                pad_gt_mask,
                                bg_index=self.num_classes,
                                return_index=True)
                        if self.for_distill:
                            self.assigned_labels = assigned_labels
                            self.assigned_bboxes = assigned_bboxes
                            self.assigned_scores = assigned_scores
                            self.assigned_gt_index = assigned_gt_index

                    else:
                        # only used in distill
                        assigned_labels = self.assigned_labels
                        assigned_bboxes = self.assigned_bboxes
                        assigned_scores = self.assigned_scores
                        assigned_gt_index = self.assigned_gt_index

                else:
                    assigned_labels, assigned_bboxes, assigned_scores, assigned_gt_index = \
                        self.assigner(
                            pred_scores_aux.detach(),
                            pred_bboxes_aux.detach() * stride_tensor,
                            anchor_points,
                            num_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            pad_gt_mask,
                            bg_index=self.num_classes,
                            return_index=True)
            alpha_l = -1

        # rescale bbox
        assigned_bboxes /= stride_tensor

        assign_out_dict = self.get_loss_from_assign(
            pred_scores, pred_distri, pred_bboxes, pred_coeffs,
            prototypes, anchor_points_s, assigned_labels, assigned_bboxes,
            assigned_scores, assigned_gt_index, gt_masks, stride_tensor, alpha_l)

        if aux_pred is not None:
            assign_out_dict_aux = self.get_loss_from_assign(
                aux_pred[0], aux_pred[1], pred_bboxes_aux, aux_pred[2],
                aux_pred[3], anchor_points_s, assigned_labels, assigned_bboxes,
                assigned_scores, assigned_gt_index, gt_masks, stride_tensor, alpha_l)
            loss = {}
            for key in assign_out_dict.keys():
                loss[key] = assign_out_dict[key] + assign_out_dict_aux[key]
        else:
            loss = assign_out_dict

        return loss

    def get_loss_from_assign(self, pred_scores, pred_distri, pred_bboxes, pred_coeffs,
                             prototypes, anchor_points_s, assigned_labels, assigned_bboxes,
                             assigned_scores, assigned_gt_index, gt_masks, stride_tensor, alpha_l):
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(assigned_scores_sum)
            assigned_scores_sum /= paddle.distributed.get_world_size()
        assigned_scores_sum = paddle.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        if self.for_distill:
            self.distill_pairs['pred_cls_scores'] = pred_scores
            self.distill_pairs['pos_num'] = assigned_scores_sum
            self.distill_pairs['assigned_scores'] = assigned_scores

            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            self.distill_pairs['target_labels'] = one_hot_label

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss_mask = self._mask_loss(pred_coeffs, prototypes, assigned_labels,
                                    assigned_bboxes, assigned_gt_index,
                                    gt_masks, assigned_scores_sum, stride_tensor)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['mask'] * loss_mask
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_mask': loss_mask,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])],
                axis=-1), None, None
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores, None
            else:
                bbox_pred, bbox_num, nms_keep_idx = self.nms(pred_bboxes,
                                                             pred_scores)
                return bbox_pred, bbox_num, nms_keep_idx
