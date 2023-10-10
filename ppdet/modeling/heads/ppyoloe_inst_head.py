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

from ..bbox_utils import batch_distance2bbox, bbox_area
from ..losses import GIoULoss
from ..initializer import bias_init_with_prob, constant_, normal_
from ..assigners.utils import generate_anchors_for_grid_cell
from ppdet.modeling.backbones.cspresnet import ConvBNLayer
from ppdet.modeling.ops import get_static_shape, get_act_fn
from ppdet.modeling.layers import MultiClassNMS

__all__ = ['PPYOLOEInstHead']


class ESEAttn(nn.Layer):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2D(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class ProtoNet(nn.Layer):
    def __init__(self,
                 in_channels,
                 proto_channels=[256, 256, 256, 32],
                 proto_kernel_sizes=[3, -2, 3, 1],
                 act='swish'):
        super(ProtoNet, self).__init__()
        protonets = []
        for out_channels, kernel_size in zip(proto_channels,
                                             proto_kernel_sizes):
            if kernel_size > 0:
                layer = ConvBNLayer(
                    in_channels,
                    out_channels,
                    filter_size=kernel_size,
                    padding=kernel_size // 2,
                    act=act)
            else:
                if out_channels is None:
                    layer = nn.Upsample(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False)
                else:
                    layer = nn.Conv2DTranspose(
                        in_channels,
                        out_channels,
                        kernel_size=-kernel_size,
                        stride=-kernel_size)
            protonets.append(layer)
            in_channels = out_channels if out_channels is not None \
                else in_channels
        self.protonet = nn.Sequential(*protonets)

    def forward(self, x):
        return self.protonet(x)


@register
class PPYOLOEInstHead(nn.Layer):
    __shared__ = [
        'num_classes', 'eval_size', 'trt', 'exclude_nms', 'exclude_post_process'
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
                 num_prototypes=32,
                 proto_channels=[256, 256, 256, 32],
                 proto_kernel_sizes=[3, -2, 3, 1],
                 with_seg_branch=True,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'mask': 2.5,
                 },
                 trt=False,
                 exclude_nms=False,
                 exclude_post_process=False):
        super(PPYOLOEInstHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.num_prototypes = num_prototypes
        self.with_seg_branch = with_seg_branch
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
        # stem
        self.stem_cls = nn.LayerList()
        self.stem_reg = nn.LayerList()
        self.stem_coeff = nn.LayerList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
            self.stem_coeff.append(ESEAttn(in_c, act=act))
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
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
            self.pred_coeff.append(
                nn.Conv2D(
                    in_c, self.num_prototypes, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2D(self.reg_max + 1, 1, 1, bias_attr=False)
        self.proj_conv.skip_quant = True
        # mask protonet
        assert fpn_strides[-1] == 8
        self.mask_head = ProtoNet(
            in_channels=in_channels[-1],
            proto_channels=proto_channels,
            proto_kernel_sizes=proto_kernel_sizes,
            act=act)
        self._init_weights()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        proj = paddle.linspace(0, self.reg_max, self.reg_max + 1).reshape(
            [1, self.reg_max + 1, 1, 1])
        self.proj_conv.weight.set_value(proj)
        self.proj_conv.weight.stop_gradient = True
        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        mask_feat = self.mask_head(feats[-1])

        cls_score_list, reg_distri_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](
                self.stem_reg[i](feat, avg_feat))
            mask_coeff = self.pred_coeff[i](
                self.stem_coeff[i](feat, avg_feat))
            # cls, reg and coeff
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))
            mask_coeff_list.append(mask_coeff.flatten(2).transpose([0, 2, 1]))
        cls_score_list = paddle.concat(cls_score_list, axis=1)
        reg_distri_list = paddle.concat(reg_distri_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=1)

        return self.get_loss([
            cls_score_list, reg_distri_list, mask_coeff_list, mask_feat,
            anchors, anchor_points, num_anchors_list, stride_tensor
        ], targets)

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

        mask_feat = self.mask_head(feats[-1])

        cls_score_list, reg_dist_list, mask_coeff_list = [], [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](
                self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape(
                [-1, 4, self.reg_max + 1, l]).transpose([0, 2, 3, 1])
            reg_dist = self.proj_conv(F.softmax(reg_dist, axis=1)).squeeze(1)
            mask_coeff = self.pred_coeff[i](
                self.stem_coeff[i](feat, avg_feat))

            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)
            mask_coeff_list.append(mask_coeff.reshape([-1, self.num_prototypes, l]))

        cls_score_list = paddle.concat(cls_score_list, axis=-1)
        reg_dist_list = paddle.concat(reg_dist_list, axis=1)
        mask_coeff_list = paddle.concat(mask_coeff_list, axis=-1)

        return cls_score_list, reg_dist_list, mask_coeff_list, mask_feat, \
            anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
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
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_max + 1]))
        pred_dist = self.proj_conv(pred_dist.transpose([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = paddle.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return paddle.concat([lt, rb], -1).clip(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
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

            dist_mask = mask_positive.unsqueeze(-1).tile(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = paddle.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = paddle.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = paddle.zeros([1])
            loss_iou = paddle.zeros([1])
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def _crop_mask_by_bbox(self, mask, bbox):
        mask_h, mask_w = mask.shape[-2:]
        x1, y1, x2, y2 = paddle.chunk(bbox.reshape([-1, 4, 1]), 4, 1)
        row = paddle.arange(mask_w, dtype=bbox.dtype).reshape([1, 1, mask_w])
        col = paddle.arange(mask_h, dtype=bbox.dtype).reshape([1, mask_h, 1])
        masks_left = row >= x1
        masks_right = row < x2
        masks_up = col >= y1
        masks_down = col < y2
        crop_mask = (masks_left * masks_right * masks_up * masks_down)
        return mask * crop_mask.astype(mask.dtype)

    def _mask_iou(self, inputs, targets):
        inputs = (inputs > 0.5).astype('float32')
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        inter = (inputs * targets).sum(1)
        uniou = inputs.sum(1) + targets.sum(1) - inter
        iou = inter / (uniou + 1e-5)
        return iou.mean()

    def _mask_loss(self, pred_coeffs, mask_feats, pos_assigned_masks,
                   pos_keep_indexes, assigned_bboxes):
        batch_size, _, mask_h, mask_w = mask_feats.shape
        mask_iou = paddle.zeros([1])
        loss_mask = paddle.zeros([1])
        for i in range(batch_size):
            pos_keep_index = pos_keep_indexes[i]
            if pos_keep_index is not None:
                num_inst = pos_keep_index.shape[0]
                mask_feat = mask_feats[i].reshape(
                    [1, self.num_prototypes, mask_h, mask_w])
                pos_pred_coeff = pred_coeffs[i][pos_keep_index].reshape(
                    [num_inst, self.num_prototypes, 1, 1])
                pos_pred_mask = F.conv2d(
                    mask_feat, weight=pos_pred_coeff).squeeze(0)
                pos_assigned_bbox = assigned_bboxes[i][pos_keep_index]
                pos_assigned_bbox = pos_assigned_bbox.reshape([num_inst, 4])
                pos_assigned_bbox /= 4.
                pos_assigned_mask = F.interpolate(
                    pos_assigned_masks[i].unsqueeze(0),
                    size=[mask_h, mask_w],
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                single_loss_mask = F.binary_cross_entropy_with_logits(
                    pos_pred_mask, pos_assigned_mask, reduction='none')
                single_loss_mask = self._crop_mask_by_bbox(
                    single_loss_mask, pos_assigned_bbox).flatten(1).sum(1)
                pos_assigned_bbox_area = bbox_area(pos_assigned_bbox)
                single_loss_mask /= pos_assigned_bbox_area
                loss_mask += single_loss_mask.mean()

                mask_iou += self._mask_iou(self._crop_mask_by_bbox(
                    F.sigmoid(pos_pred_mask), pos_assigned_bbox), pos_assigned_mask)

        loss_mask = loss_mask / batch_size
        mask_iou = mask_iou / batch_size
        return loss_mask, mask_iou

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, pred_coeffs, mask_feats, anchors, \
            anchor_points, num_anchors_list, stride_tensor = head_outs

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_bboxes = gt_meta['gt_bbox']
        gt_masks = gt_meta['gt_segm'].astype('float32')
        pad_gt_mask = gt_meta['pad_gt_mask']
        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores, \
                pos_assigned_masks, pos_keep_indexes = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_masks,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores, \
                pos_assigned_masks, pos_keep_indexes = \
                self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_masks,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
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

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss_mask, mask_iou = self._mask_loss(pred_coeffs, mask_feats, pos_assigned_masks,
                        pos_keep_indexes, assigned_bboxes * stride_tensor)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + loss_mask * 2.5
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
            'loss_mask': loss_mask,
            'mask_iou': mask_iou
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, pred_coeffs, mask_feats, \
            anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return paddle.concat(
                [pred_bboxes, pred_scores.transpose([0, 2, 1])], axis=-1), None
        else:
            # scale bbox to origin
            scale_y, scale_x = paddle.split(scale_factor, 2, axis=-1)
            scale_factor = paddle.concat(
                [scale_x, scale_y, scale_x, scale_y],
                axis=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, nms_keep_index = self.nms(pred_bboxes, pred_scores)
                mask_preds = []
                if nms_keep_index.numel() > 0:
                    total_pred_split = bbox_num.flatten().numpy().tolist()
                    pred_bboxes = paddle.split(
                        bbox_pred[:, 2:], total_pred_split, axis=0)
                    keep_indexes = paddle.split(nms_keep_index, total_pred_split, axis=0)
                    pred_coeffs = pred_coeffs.transpose([0, 2, 1])
                    batch_size, _, mask_h, mask_w = mask_feats.shape
                    for i, (pred_coeff, pred_bbox, keep_index, mask_feat) in enumerate(zip(
                            pred_coeffs, pred_bboxes, keep_indexes, mask_feats)):
                        num_inst = keep_index.shape[0]
                        pos_pred_coeff = pred_coeff[keep_index.flatten()].reshape(
                            [num_inst, self.num_prototypes])
                        mask_feat = mask_feat.reshape(
                            [1, self.num_prototypes, mask_h, mask_w])
                        pos_pred_coeff = pos_pred_coeff.reshape(
                            [num_inst, self.num_prototypes, 1, 1])
                        pos_pred_mask = F.conv2d(
                            mask_feat, weight=pos_pred_coeff)
                        pos_pred_mask = F.interpolate(
                            pos_pred_mask,
                            scale_factor=[4 / float(scale_y[i]), 4 / float(scale_x[i])],
                            mode='bilinear',
                            align_corners=False).squeeze(0)
                        pos_pred_mask = (F.sigmoid(pos_pred_mask) > 0.5).astype(paddle.int32)
                        mask_preds.append(pos_pred_mask.astype(paddle.int32))
                mask_pred = paddle.concat(mask_preds, axis=0)
                mask_pred = self._crop_mask_by_bbox(mask_pred, bbox_pred[:, 2:])
                return bbox_pred, bbox_num, mask_pred
