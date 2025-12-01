# ConvNeXt-Tiny for Medical Surgical Instruments Segmentation

本文档介绍如何使用 ConvNeXt-Tiny 作为 DI-MaskDINO 的 backbone 进行医疗手术器械实例分割。

## 📋 配置说明

### 场景特点
- **器械数量**: 每张图像约 20-30 把手术器械
- **摆放方式**: 密集、凌乱摆放
- **遮挡情况**: 部分器械被遮挡,细长器械并排摆放
- **器械特征**: 大小不一、长宽比差异大
- **类别数量**: 总计 500 个器械类别

### 配置优化

配置文件 `dimaskdino_convnext_tiny_medical_instruments.yaml` 针对上述场景进行了以下优化:

#### 1. Backbone: ConvNeXt-Tiny
- **优势**: 
  - 比 ResNet 更强的特征提取能力
  - 大卷积核 (7x7) 适合捕捉细长器械的形状特征
  - 层级归一化提升训练稳定性
- **参数**:
  - Depths: [3, 3, 9, 3]
  - Dims: [96, 192, 384, 768]
  - Drop Path Rate: 0.2 (增强正则化)

#### 2. 多尺度特征
- **NUM_FEATURE_LEVELS**: 4 (使用 res2-res5 全部四个尺度)
- **作用**: 更好地处理不同大小的器械

#### 3. 查询数量增加
- **NUM_OBJECT_QUERIES**: 400 (原 300)
- **FIRST_SELECTED_QUERIES**: 800 (原 600)
- **NUM_GUIDING_TOKENS**: 100 (原 50)
- **原因**: 场景中有 20-30 个器械,需要更多查询来覆盖

#### 4. 解码器加深
- **DEC_LAYERS**: 9 (原 6)
- **TI_LAYERS**: 3 (原 2)
- **原因**: 复杂场景需要更强的特征交互能力

#### 5. 采样点增加
- **TRAIN_NUM_POINTS**: 16384 (原 12544)
- **原因**: 细长器械需要更多采样点来精确分割边界

#### 6. 去噪查询增加
- **DN_NUM**: 150 (原 100)
- **原因**: 500 个类别需要更多去噪训练

#### 7. 遮挡处理
- **OVERLAP_THRESHOLD**: 0.6 (原 0.8)
- **原因**: 降低阈值以更好处理遮挡情况

#### 8. 训练优化
- **Batch Size**: 8 (考虑显存限制)
- **Learning Rate**: 0.0001
- **Epochs**: 50
- **Mixed Precision**: 启用 (加速训练)

## 🚀 使用步骤

### 步骤 1: 准备预训练权重

下载 ConvNeXt-Tiny ImageNet-1K 预训练权重:

```bash
# 下载官方权重
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth

# 转换为 Detectron2 格式
python tools/convert_convnext_to_d2.py \
    --source convnext_tiny_1k_224_ema.pth \
    --output convnext_tiny_1k_224_d2.pkl
```

### 步骤 2: 准备数据集

#### 2.1 数据集格式

将你的医疗器械数据集组织为 COCO 格式:

```
medical_instruments/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val/
    ├── img_101.jpg
    ├── img_102.jpg
    └── ...
```

#### 2.2 注册数据集

编辑 `datasets/register_medical_instruments.py`:

```python
# 设置你的数据集路径
DATA_ROOT = "/path/to/your/medical_instruments"
NUM_CLASSES = 500
```

在 `train_net.py` 中添加数据集注册:

```python
# 在文件开头导入
import sys
sys.path.insert(0, 'datasets')
from register_medical_instruments import register_all_medical_instruments

# 在 main() 函数开始处注册
def main(args):
    # 注册医疗器械数据集
    register_all_medical_instruments(
        data_root="/path/to/your/medical_instruments",
        num_classes=500
    )
    
    # ... 原有代码 ...
```

### 步骤 3: 修改配置文件

编辑 `configs/dimaskdino_convnext_tiny_medical_instruments.yaml`:

```yaml
MODEL:
  WEIGHTS: "path/to/convnext_tiny_1k_224_d2.pkl"  # 设置预训练权重路径

DATASETS:
  TRAIN: ("medical_instruments_train",)
  TEST: ("medical_instruments_val",)

SOLVER:
  # 根据你的数据集大小调整
  # MAX_ITER = (图像数量 / batch_size) * epochs
  # 例如: 5000 张图像, batch_size=8, 50 epochs
  # MAX_ITER = (5000/8) * 50 = 31250
  MAX_ITER: 31250
  STEPS: (25000, 28125)  # 在第 40 和 45 epoch 降低学习率

OUTPUT_DIR: "./output/medical_instruments_convnext_tiny"
```

### 步骤 4: 开始训练

```bash
# 单 GPU 训练
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --num-gpus 1

# 多 GPU 训练 (例如 4 GPUs)
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --num-gpus 4
```

### 步骤 5: 评估

```bash
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --eval-only \
    MODEL.WEIGHTS output/medical_instruments_convnext_tiny/model_final.pth
```

## 📊 性能优化建议

### 1. 数据增强

如果遮挡严重,可以启用 Copy-Paste 增强:

```yaml
INPUT:
  DATASET_MAPPER_NAME: "coco_instance_copy_paste"
```

### 2. 调整超参数

根据训练情况可能需要调整:

- **学习率**: 如果loss下降慢,可以尝试 `BASE_LR: 0.0002`
- **Batch Size**: 如果显存充足,可以增加到 16
- **查询数量**: 如果器械数量更多,可以继续增加 `NUM_OBJECT_QUERIES`

### 3. 长宽比处理

对于细长器械,可以添加特殊的长宽比处理:

```yaml
INPUT:
  CROP:
    ENABLED: True
    TYPE: "absolute_range"
    SIZE: (512, 1024)  # 允许非正方形裁剪
```

### 4. 测试时增强 (TTA)

使用测试时增强提升性能:

```python
# 在评估时
from dimaskdino.test_time_augmentation import SemanticSegmentorWithTTA

predictor = SemanticSegmentorWithTTA(cfg, model)
```

## 🔍 监控训练

推荐监控以下指标:

1. **总Loss**: 应该平稳下降
2. **Class Loss**: 分类损失,关注是否收敛
3. **Mask Loss & Dice Loss**: mask 质量指标
4. **Box Loss & GIoU Loss**: 边界框质量
5. **mAP**: 平均精度 (主要指标)
6. **AR**: 平均召回率

## 🐛 常见问题

### Q1: 显存不足
**A**: 
- 减小 batch size: `IMS_PER_BATCH: 4`
- 减少查询数: `NUM_OBJECT_QUERIES: 300`
- 减少采样点: `TRAIN_NUM_POINTS: 12544`
- 禁用混合精度: `SOLVER.AMP.ENABLED: False`

### Q2: 细长器械分割不准确
**A**:
- 增加采样点: `TRAIN_NUM_POINTS: 20480`
- 增加 mask 权重: `MASK_WEIGHT: 7.0`, `DICE_WEIGHT: 7.0`
- 使用更高分辨率: `IMAGE_SIZE: 1280`

### Q3: 遮挡处理效果差
**A**:
- 启用 Copy-Paste 增强
- 降低重叠阈值: `OVERLAP_THRESHOLD: 0.5`
- 增加解码器层数: `DEC_LAYERS: 12`

### Q4: 训练速度慢
**A**:
- 启用混合精度训练 (已默认启用)
- 增加 GPU 数量进行分布式训练
- 减少解码器层数: `DEC_LAYERS: 6`

## 📈 预期性能

在类似场景下,预期可以达到:

- **mAP@0.5**: 75-85%
- **mAP@0.75**: 60-70%
- **mAP**: 65-75%

实际性能取决于:
- 数据集质量和标注精度
- 训练时长和超参数调整
- 器械的复杂度和遮挡程度

## 📚 相关资源

- [ConvNeXt 论文](https://arxiv.org/abs/2201.03545)
- [DI-MaskDINO 原始仓库](https://github.com/IDEA-Research/MaskDINO)
- [Detectron2 文档](https://detectron2.readthedocs.io/)

## 💡 进阶技巧

### 使用更大的 ConvNeXt 模型

如果需要更强的性能,可以使用 ConvNeXt-Small/Base:

```yaml
MODEL:
  CONVNEXT:
    DEPTHS: [3, 3, 27, 3]  # ConvNeXt-Small
    DIMS: [96, 192, 384, 768]
```

或

```yaml
MODEL:
  CONVNEXT:
    DEPTHS: [3, 3, 27, 3]  # ConvNeXt-Base
    DIMS: [128, 256, 512, 1024]
```

### 类别平衡

如果 500 个类别数据不平衡,考虑使用类别采样策略或焦点损失。

---

如有问题,请查看日志或提交 issue。祝训练顺利! 🎉
