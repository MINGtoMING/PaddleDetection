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

import copy
import os
import pickle
import random

import lmdb
from tqdm import tqdm

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger

from .dataset import DetDataset

logger = setup_logger(__name__)

__all__ = ['OVDDataSet']


@register
@serializable
class OVDDataSet(DetDataset):
    """
    Load dataset with LMDB format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. 
            False as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 repeat=1):
        super(OVDDataSet, self).__init__(dataset_dir,
                                         image_dir,
                                         anno_path,
                                         data_fields,
                                         sample_num,
                                         repeat=repeat)
        self.load_crowd = load_crowd
        self.env = None
        self.keys = []

    def check_or_download_dataset(self):
        pass

    def parse_dataset(self):
        image_dir = os.path.join(self.dataset_dir, self.image_dir)
        assert os.path.exists(image_dir), FileNotFoundError(image_dir)
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        assert os.path.exists(anno_path), FileNotFoundError(anno_path)
        self.env = lmdb.open(anno_path,
                             readonly=True,
                             lock=False,
                             max_readers=1024)
        with self.env.begin(write=False) as txn:
            total_num = txn.stat()['entries']
            for i in tqdm(range(total_num)):
                key = f"{i}".encode()
                self.keys.append(key)
                rec = pickle.loads(txn.get(key))
                assert "im_file" in rec, "Missing 'im_file' in record"
                im_path = os.path.join(image_dir, rec["im_file"])
                assert os.path.exists(
                    im_path), f"Image file does not exist: {im_path}"
                assert "im_id" in rec and rec[
                    "im_id"] == i, "Missing 'im_id' in record"

                for tag in self.data_fields:
                    if tag != "image":
                        assert tag in rec, f"Missing '{tag}' in record"

    def __getitem__(self, idx):
        n = len(self.keys)
        if self.repeat > 1:
            idx %= n
        # data batch
        with self.env.begin(write=False) as txn:
            key = self.keys[idx]
            roidb = copy.deepcopy(txn.get(key))
            if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
                key = random.choice(self.keys)
                roidb = [roidb, copy.deepcopy(txn.get(key))]
            elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
                key = random.choice(self.keys)
                roidb = [roidb, copy.deepcopy(txn.get(key))]
            elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
                roidb = [roidb] + [
                    copy.deepcopy(txn.get(key))
                    for key in random.choices(self.keys, k=4)
                ]
            elif self.pre_img_epoch == 0 or self._epoch < self.pre_img_epoch:
                # Add previous image as input, only used in CenterTrack
                idx_pre_img = idx - 1
                if idx_pre_img < 0:
                    idx_pre_img = idx + 1
                key = self.keys[idx_pre_img]
                roidb = [roidb, copy.deepcopy(txn.get(key))]
            if isinstance(roidb, Sequence):
                for r in roidb:
                    r['curr_iter'] = self._curr_iter
                    r['curr_epoch'] = self._epoch
            else:
                roidb['curr_iter'] = self._curr_iter
                roidb['curr_epoch'] = self._epoch
        self._curr_iter += 1

        if self.transform_schedulers:
            assert isinstance(self.transform_schedulers, list)
            if isinstance(roidb, Sequence):
                for r in roidb:
                    r['transform_schedulers'] = self.transform_schedulers
            else:
                roidb['transform_schedulers'] = self.transform_schedulers

        return self.transform(roidb)

    def __len__(self):
        return len(self.keys) * self.repeat