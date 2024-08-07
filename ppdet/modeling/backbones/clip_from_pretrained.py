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

import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModelWithProjection
from ppdet.core.workspace import register, serializable

__all__ = ['CLIPFromPretrained']


@register
@serializable
class CLIPFromPretrained(nn.Layer):

    def __init__(self,
                 model_name='openai/clip-vit-base-patch32',
                 freeze_all=True):
        super(CLIPFromPretrained, self).__init__()
        clip_cfg = CLIPTextConfig.from_pretrained(model_name,
                                                  attention_dropout=0.)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name, config=clip_cfg)

        if freeze_all:
            for p in self.model.parameters():
                p.stop_gradient = True

    def forward(self, inputs):
        text_token = inputs['text_token']
        batch_num, word_num = text_token.shape[:2]
        text_token = text_token.flatten(0, 1)
        text_token_mask = inputs.get('text_token_mask', None)
        if text_token_mask is not None:
            text_token_mask = text_token_mask.flatten(0, 1)
        text_feats = self.model(input_ids=text_token,
                                attention_mask=text_token_mask)['text_embeds']
        text_feats = F.normalize(text_feats, p=2, axis=-1)
        text_feats = text_feats.reshape([batch_num, word_num, -1])
        return text_feats
