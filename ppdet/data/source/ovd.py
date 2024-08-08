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

import os

import numpy as np

from ppdet.core.workspace import register, serializable
from ppdet.utils.logger import setup_logger
from .dataset import DetDataset

logger = setup_logger(__name__)

__all__ = ['OVDDataSet']


@register
@serializable
class OVDDataSet(DetDataset):
    """
    Load ovd dataset.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        load_crowd (bool): whether to load crowded ground-truth.
            False as default
    """

    def __init__(
        self,
        dataset_dir=None,
        image_dir=None,
        anno_path=None,
        load_crowd=False,
    ):
        super(OVDDataSet, self).__init__(dataset_dir=dataset_dir,
                                         image_dir=image_dir,
                                         anno_path=anno_path)
        assert isinstance(self.image_dir, dict)
        self.load_crowd = load_crowd

    def check_or_download_dataset(self):
        assert os.path.exists(self.dataset_dir)
        annotation_path = os.path.join(self.dataset_dir, self.anno_path)
        assert os.path.isfile(annotation_path)
        for _, sub_image_dir in self.image_dir.items():
            sub_image_path = os.path.join(self.dataset_dir, sub_image_dir)
            assert os.path.isdir(sub_image_path)

    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = {
            source: os.path.join(self.dataset_dir, sub_image_dir)
            for source, sub_image_dir in self.image_dir.items()
        }

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        records = []
        ct = 0

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            source = img_anno['source']

            im_path = os.path.join(image_dir[source], im_fname)

            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(im_w, im_h, img_id))
                continue

            texts = img_anno['texts']
            if source == 'obj365':
                texts = [
                    'person', 'sneakers', 'chair', 'hat', 'lamp', 'bottle',
                    'cabinet/shelf', 'cup', 'car', 'glasses', 'picture/frame',
                    'desk', 'handbag', 'street lights', 'book', 'plate',
                    'helmet', 'leather shoes', 'pillow', 'glove',
                    'potted plant', 'bracelet', 'flower', 'tv', 'storage box',
                    'vase', 'bench', 'wine glass', 'boots', 'bowl',
                    'dining table', 'umbrella', 'boat', 'flag', 'speaker',
                    'trash bin/can', 'stool', 'backpack', 'couch', 'belt',
                    'carpet', 'basket', 'towel/napkin', 'slippers',
                    'barrel/bucket', 'coffee table', 'suv', 'toy', 'tie',
                    'bed', 'traffic light', 'pen/pencil', 'microphone',
                    'sandals', 'canned', 'necklace', 'mirror', 'faucet',
                    'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch',
                    'sink', 'horse', 'fish', 'apple', 'camera', 'candle',
                    'teddy bear', 'cake', 'motorcycle', 'wild bird', 'laptop',
                    'knife', 'traffic sign', 'cell phone', 'paddle', 'truck',
                    'cow', 'power outlet', 'clock', 'drum', 'fork', 'bus',
                    'hanger', 'nightstand', 'pot/pan', 'sheep', 'guitar',
                    'traffic cone', 'tea pot', 'keyboard', 'tripod', 'hockey',
                    'fan', 'dog', 'spoon', 'blackboard/whiteboard', 'balloon',
                    'air conditioner', 'cymbal', 'mouse', 'telephone',
                    'pickup truck', 'orange', 'banana', 'airplane', 'luggage',
                    'skis', 'soccer', 'trolley', 'oven', 'remote',
                    'baseball glove', 'paper towel', 'refrigerator', 'train',
                    'tomato', 'machinery vehicle', 'tent',
                    'shampoo/shower gel', 'head phone', 'lantern', 'donut',
                    'cleaning products', 'sailboat', 'tangerine', 'pizza',
                    'kite', 'computer box', 'elephant', 'toiletries',
                    'gas stove', 'broccoli', 'toilet', 'stroller', 'shovel',
                    'baseball bat', 'microwave', 'skateboard', 'surfboard',
                    'surveillance camera', 'gun', 'life saver', 'cat', 'lemon',
                    'liquid soap', 'zebra', 'duck', 'sports car', 'giraffe',
                    'pumpkin', 'piano', 'stop sign', 'radiator', 'converter',
                    'tissue ', 'carrot', 'washing machine', 'vent', 'cookies',
                    'cutting/chopping board', 'tennis racket', 'candy',
                    'skating and skiing shoes', 'scissors', 'folder',
                    'baseball', 'strawberry', 'bow tie', 'pigeon', 'pepper',
                    'coffee machine', 'bathtub', 'snowboard', 'suitcase',
                    'grapes', 'ladder', 'pear', 'american football',
                    'basketball', 'potato', 'paint brush', 'printer',
                    'billiards', 'fire hydrant', 'goose', 'projector',
                    'sausage', 'fire extinguisher', 'extension cord',
                    'facial mask', 'tennis ball', 'chopsticks',
                    'electronic stove and gas stove', 'pie', 'frisbee',
                    'kettle', 'hamburger', 'golf club', 'cucumber', 'clutch',
                    'blender', 'tong', 'slide', 'hot dog', 'toothbrush',
                    'facial cleanser', 'mango', 'deer', 'egg', 'violin',
                    'marker', 'ship', 'chicken', 'onion', 'ice cream', 'tape',
                    'wheelchair', 'plum', 'bar soap', 'scale', 'watermelon',
                    'cabbage', 'router/modem', 'golf ball', 'pine apple',
                    'crane', 'fire truck', 'peach', 'cello', 'notepaper',
                    'tricycle', 'toaster', 'helicopter', 'green beans',
                    'brush', 'carriage', 'cigar', 'earphone', 'penguin',
                    'hurdle', 'swing', 'radio', 'CD', 'parking meter', 'swan',
                    'garlic', 'french fries', 'horn', 'avocado', 'saxophone',
                    'trumpet', 'sandwich', 'cue', 'kiwi fruit', 'bear',
                    'fishing rod', 'cherry', 'tablet', 'green vegetables',
                    'nuts', 'corn', 'key', 'screwdriver', 'globe', 'broom',
                    'pliers', 'volleyball', 'hammer', 'eggplant', 'trophy',
                    'dates', 'board eraser', 'rice', 'tape measure/ruler',
                    'dumbbell', 'hamimelon', 'stapler', 'camel', 'lettuce',
                    'goldfish', 'meat balls', 'medal', 'toothpaste',
                    'antelope', 'shrimp', 'rickshaw', 'trombone',
                    'pomegranate', 'coconut', 'jellyfish', 'mushroom',
                    'calculator', 'treadmill', 'butterfly', 'egg tart',
                    'cheese', 'pig', 'pomelo', 'race car', 'rice cooker',
                    'tuba', 'crosswalk sign', 'papaya', 'hair drier',
                    'green onion', 'chips', 'dolphin', 'sushi', 'urinal',
                    'donkey', 'electric drill', 'spring rolls',
                    'tortoise/turtle', 'parrot', 'flute', 'measuring cup',
                    'shark', 'steak', 'poker card', 'binoculars', 'llama',
                    'radish', 'noodles', 'yak', 'mop', 'crab', 'microscope',
                    'barbell', 'bread/bun', 'baozi', 'lion', 'red cabbage',
                    'polar bear', 'lighter', 'seal', 'mangosteen', 'comb',
                    'eraser', 'pitaya', 'scallop', 'pencil case', 'saw',
                    'table tennis paddle', 'okra', 'starfish', 'eagle',
                    'monkey', 'durian', 'game board', 'rabbit', 'french horn',
                    'ambulance', 'asparagus', 'hoverboard', 'pasta', 'target',
                    'hotair balloon', 'chainsaw', 'lobster', 'iron',
                    'flashlight'
                ]
                texts = [text.split('/') for text in texts]

            if len(texts) == 0:
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
                'texts': texts,
            }

            ins_anno_ids = coco.getAnnIds(
                imgIds=[img_id], iscrowd=None if self.load_crowd else False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                # check gt bbox
                if 'bbox' not in inst.keys():
                    continue
                else:
                    if not any(np.array(inst['bbox'])):
                        continue

                x1, y1, box_w, box_h = inst['bbox']
                x2 = x1 + box_w
                y2 = y1 + box_h
                eps = 1e-5
                if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                    inst['clean_bbox'] = [
                        round(float(x), 3) for x in [x1, y1, x2, y2]
                    ]
                    bboxes.append(inst)
                else:
                    logger.warning(
                        'Found an invalid bbox in annotations: im_id: {}, '
                        'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                            img_id, float(inst['area']), x1, y1, x2, y2))

            num_bbox = len(bboxes)
            if num_bbox <= 0:
                continue

            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)

            for i, box in enumerate(bboxes):
                gt_class[i][0] = box['text_id']
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i][0] = box['iscrowd']

            gt_rec = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
            }

            for k, v in gt_rec.items():
                coco_rec[k] = v

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))

            records.append(coco_rec)
            ct += 1

        assert ct > 0, 'not found any coco record in %s' % anno_path
        logger.info(
            'Load [{} samples valid, {} samples invalid] in file {}.'.format(
                ct,
                len(img_ids) - ct, anno_path))
        self.roidbs = records
