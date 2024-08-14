from timeit import timeit

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg_path = 'configs/ovrtdetr/ovrtdetr_r18vd_6x_coco.yml'
cfg = load_config(cfg_path)
model = create(cfg.architecture)

print(model)

model.eval()

input = {}
input['image'] = paddle.rand([4, 3, 640, 640])
input['scale_factor'] = paddle.ones([4, 2])
input['im_shape'] = paddle.ones([4, 2]) * 640
input['text_token'] = paddle.randint(0, 3000, [4, 80, 4])
input['text_token_mask'] = paddle.ones([4, 80, 4], dtype=bool)

out = model(input)

for _ in range(100):
    print(timeit(lambda: model(input), number=1))


print(out)