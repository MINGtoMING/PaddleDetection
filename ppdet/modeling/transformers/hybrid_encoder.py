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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder

from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ..backbones.resnet import ConvNormLayer
from ..initializer import linear_init_
from ..layers import MultiHeadAttention
from ..shape_spec import ShapeSpec

__all__ = ['HybridEncoder', 'MaskHybridEncoder', 'OVHybridEncoder']


class CSPRepLayer(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              ksize=1,
                              stride=1,
                              bias=bias,
                              act=act)
        self.conv2 = BaseConv(in_channels,
                              hidden_channels,
                              ksize=1,
                              stride=1,
                              bias=bias,
                              act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(hidden_channels,
                                  out_channels,
                                  ksize=1,
                                  stride=1,
                                  bias=bias,
                                  act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@register
class TransformerLayer(nn.Layer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channel,
                              hidden_dim,
                              kernel_size=1,
                              bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2,
                            hidden_dim,
                            round(3 * depth_mult),
                            act=act,
                            expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2,
                            hidden_dim,
                            round(3 * depth_mult),
                            act=act,
                            expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat([
            paddle.sin(out_w),
            paddle.cos(out_w),
            paddle.sin(out_h),
            paddle.cos(out_h)
        ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False, is_teacher=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])
                if self.training or self.eval_size is None or is_teacher:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 -
                                            idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh,
                                          scale_factor=2.,
                                          mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]


class MaskFeatFPN(nn.Layer):

    def __init__(self,
                 in_channels=[256, 256, 256],
                 fpn_strides=[32, 16, 8],
                 feat_channels=256,
                 dropout_ratio=0.0,
                 out_channels=256,
                 align_corners=False,
                 act='swish'):
        super(MaskFeatFPN, self).__init__()
        assert len(in_channels) == len(fpn_strides)
        reorder_index = np.argsort(fpn_strides, axis=0)
        in_channels = [in_channels[i] for i in reorder_index]
        fpn_strides = [fpn_strides[i] for i in reorder_index]
        assert min(fpn_strides) == fpn_strides[0]
        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2D(dropout_ratio)

        self.scale_heads = nn.LayerList()
        for i in range(len(fpn_strides)):
            head_length = max(
                1, int(np.log2(fpn_strides[i]) - np.log2(fpn_strides[0])))
            scale_head = []
            for k in range(head_length):
                in_c = in_channels[i] if k == 0 else feat_channels
                scale_head.append(
                    nn.Sequential(BaseConv(in_c, feat_channels, 3, 1,
                                           act=act)))
                if fpn_strides[i] != fpn_strides[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=align_corners))

            self.scale_heads.append(nn.Sequential(*scale_head))

        self.output_conv = BaseConv(feat_channels, out_channels, 3, 1, act=act)

    def forward(self, inputs):
        x = [inputs[i] for i in self.reorder_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.fpn_strides)):
            output = output + F.interpolate(self.scale_heads[i](x[i]),
                                            size=output.shape[2:],
                                            mode='bilinear',
                                            align_corners=self.align_corners)

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.output_conv(output)
        return output


@register
@serializable
class MaskHybridEncoder(HybridEncoder):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size', 'num_prototypes']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 num_prototypes=32,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 mask_feat_channels=[64, 64],
                 act='silu',
                 trt=False,
                 eval_size=None):
        assert len(in_channels) == len(feat_strides)
        x4_feat_dim = in_channels.pop(0)
        x4_feat_stride = feat_strides.pop(0)
        use_encoder_idx = [i - 1 for i in use_encoder_idx]
        assert x4_feat_stride == 4

        super(MaskHybridEncoder,
              self).__init__(in_channels=in_channels,
                             feat_strides=feat_strides,
                             hidden_dim=hidden_dim,
                             use_encoder_idx=use_encoder_idx,
                             num_encoder_layers=num_encoder_layers,
                             encoder_layer=encoder_layer,
                             pe_temperature=pe_temperature,
                             expansion=expansion,
                             depth_mult=depth_mult,
                             act=act,
                             trt=trt,
                             eval_size=eval_size)

        self.mask_feat_head = MaskFeatFPN([hidden_dim] * len(feat_strides),
                                          feat_strides,
                                          feat_channels=mask_feat_channels[0],
                                          out_channels=mask_feat_channels[1],
                                          act=act)
        self.enc_mask_lateral = BaseConv(x4_feat_dim,
                                         mask_feat_channels[1],
                                         3,
                                         1,
                                         act=act)
        self.enc_mask_output = nn.Sequential(
            BaseConv(mask_feat_channels[1],
                     mask_feat_channels[1],
                     3,
                     1,
                     act=act),
            nn.Conv2D(mask_feat_channels[1], num_prototypes, 1))

    def forward(self, feats, for_mot=False, is_teacher=False):
        x4_feat = feats.pop(0)

        enc_feats = super(MaskHybridEncoder,
                          self).forward(feats,
                                        for_mot=for_mot,
                                        is_teacher=is_teacher)

        mask_feat = self.mask_feat_head(enc_feats)
        mask_feat = F.interpolate(mask_feat,
                                  scale_factor=2,
                                  mode='bilinear',
                                  align_corners=False)
        mask_feat += self.enc_mask_lateral(x4_feat)
        mask_feat = self.enc_mask_output(mask_feat)

        return enc_feats, mask_feat


class MaxSigmoidAttnLayer(nn.Layer):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 visual_embed_dim=512,
                 text_embed_dim=512,
                 num_heads=4,
                 with_scale=False):
        super(MaxSigmoidAttnLayer, self).__init__()
        assert out_channels % num_heads == 0
        assert text_embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = visual_embed_dim // num_heads

        if visual_embed_dim != in_channels:
            self.visual_proj = ConvNormLayer(in_channels,
                                             visual_embed_dim,
                                             filter_size=1,
                                             stride=1,
                                             act=None,
                                             freeze_norm=False)
        else:
            self.visual_proj = nn.Identity()

        self.text_proj = nn.Linear(text_embed_dim, visual_embed_dim)
        self.bias = self.create_parameter(
            shape=[1, num_heads, 1, 1],
            default_initializer=nn.initializer.Constant(0.))
        self.add_parameter('bias', self.bias)

        if with_scale:
            self.scale = self.create_parameter(
                shape=[1, num_heads, 1, 1],
                default_initializer=nn.initializer.Constant(1.))
            self.add_parameter('scale', self.scale)
        else:
            self.scale = 1.

        self.out_proj = ConvNormLayer(in_channels,
                                      out_channels,
                                      filter_size=3,
                                      stride=1,
                                      act=None,
                                      freeze_norm=False)

    def forward(self, visual_embed, text_embed):
        batch_num, _, height, width = visual_embed.shape

        out = self.out_proj(visual_embed)
        out = out.reshape([batch_num, self.num_heads, -1, height, width])

        text_embed = self.text_proj(text_embed)
        text_embed = text_embed.reshape([batch_num, -1, self.num_heads, self.head_dim])
        visual_embed = self.visual_proj(visual_embed)
        visual_embed = visual_embed.flatten(2).reshape(
            [batch_num, self.num_heads, self.head_dim, -1])
        visual_embed = visual_embed.transpose([0, 1, 3, 2])
        text_embed = text_embed.transpose([0, 2, 3, 1])
        # visual_embed: [batch_num, num_heads, hw, head_channels]
        # text_embed: [batch_num, num_heads, head_channels, num_words]
        attn_weight = paddle.matmul(visual_embed, text_embed)
        attn_weight = attn_weight.reshape(
            [batch_num, self.num_heads, height, width, -1])
        attn_weight = attn_weight.max(-1)
        attn_weight = attn_weight * self.head_dim ** -0.5
        attn_weight = attn_weight + self.bias
        attn_weight = attn_weight.sigmoid() * self.scale

        out = out * attn_weight.unsqueeze(2)
        out = out.reshape([batch_num, -1, height, width])
        return out


class MaxSigmoidCSPRepLayer(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 visual_embed_dim=512,
                 text_embed_dim=512,
                 num_heads=4,
                 with_scale=False,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(MaxSigmoidCSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels,
                              hidden_channels,
                              ksize=1,
                              stride=1,
                              bias=bias,
                              act=act)
        self.conv2 = BaseConv(in_channels,
                              hidden_channels,
                              ksize=1,
                              stride=1,
                              bias=bias,
                              act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        self.attn = MaxSigmoidAttnLayer(hidden_channels,
                                        hidden_channels,
                                        visual_embed_dim=visual_embed_dim,
                                        text_embed_dim=text_embed_dim,
                                        num_heads=num_heads,
                                        with_scale=with_scale)

        if hidden_channels != out_channels:
            self.conv3 = BaseConv(hidden_channels,
                                  out_channels,
                                  ksize=1,
                                  stride=1,
                                  bias=bias,
                                  act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x, guide):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_1 = self.attn(x_1, guide)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@register
@serializable
class OVHybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'text_embed_dim', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 text_embed_dim=512,
                 msa_embed_dims=[128, 256, 512],
                 msa_num_heads=[4, 8, 16],
                 msa_with_scale=False,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(OVHybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(in_channel,
                              hidden_dim,
                              kernel_size=1,
                              bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                MaxSigmoidCSPRepLayer(hidden_dim * 2,
                                      hidden_dim,
                                      visual_embed_dim=msa_embed_dims[len(self.in_channels) - 1 - idx],
                                      text_embed_dim=text_embed_dim,
                                      num_heads=msa_num_heads[len(self.in_channels) - 1 - idx],
                                      with_scale=msa_with_scale,
                                      num_blocks=round(3 * depth_mult),
                                      act=act,
                                      expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                MaxSigmoidCSPRepLayer(hidden_dim * 2,
                                      hidden_dim,
                                      visual_embed_dim=msa_embed_dims[idx],
                                      text_embed_dim=text_embed_dim,
                                      num_heads=msa_num_heads[idx],
                                      with_scale=msa_with_scale,
                                      num_blocks=round(3 * depth_mult),
                                      act=act,
                                      expansion=expansion))

        self._reset_parameters()

    def forward(self, feats, guide):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(feat_heigh,
                                          scale_factor=2.,
                                          mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], axis=1), guide)
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1), guide)
            outs.append(out)

        return outs

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat([
            paddle.sin(out_w),
            paddle.cos(out_w),
            paddle.sin(out_h),
            paddle.cos(out_h)
        ], axis=1)[None, :, :]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]