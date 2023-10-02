# PP-YOLOE-Seg


## 简介
PP-YOLOE-Seg是基于PP-YOLOE并结合YOLACT的实时实例分割模型。

### 训练

请执行以下指令训练

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe_seg/ppyoloe_plus_seg_crn_l_80e_coco.yml --eval --amp
```
### 评估

执行以下命令在单个GPU上评估COCO val2017数据集

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyoloe_seg/ppyoloe_plus_seg_crn_l_80e_coco.yml -o weights=${model_weights}
```

在coco test-dev2017上评估，请先从[COCO数据集下载](https://cocodataset.org/#download)下载COCO test-dev2017数据集，然后解压到COCO数据集文件夹并像`configs/ppyolo/ppyolo_test.yml`一样配置`EvalDataset`。

### 推理

使用以下命令在单张GPU上预测图片，使用`--infer_img`推理单张图片以及使用`--infer_dir`推理文件中的所有图片。


```bash
# 推理单张图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe_seg/ppyoloe_plus_seg_crn_l_80e_coco.yml -o weights=${model_weights} --infer_img=demo/000000014439_640x640.jpg

# 推理文件中的所有图片
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyoloe_seg/ppyoloe_plus_crn_l_80e_coco.yml -o weights=${model_weights} --infer_dir=demo
```
