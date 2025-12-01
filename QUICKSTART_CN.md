# 🚀 ConvNeXt-Tiny 医疗器械分割快速入门

## ✅ 已完成的集成

ConvNeXt-Tiny backbone 已成功集成到 DI-MaskDINO 项目中,并针对你的医疗手术器械场景进行了优化!

### 📦 新增文件清单

```
✓ dimaskdino/modeling/backbone/convnext.py        # ConvNeXt 实现
✓ configs/dimaskdino_convnext_tiny_medical_instruments.yaml  # 医疗器械配置
✓ tools/convert_convnext_to_d2.py                 # 权重转换工具
✓ tools/test_convnext_backbone.py                 # 集成测试
✓ datasets/register_medical_instruments.py        # 数据集注册
✓ configs/CONVNEXT_MEDICAL_README.md              # 详细文档
✓ CONVNEXT_INTEGRATION_SUMMARY.md                 # 集成总结
```

### 🎯 场景优化

配置已针对你的场景优化:
- ✅ **500个器械类别**
- ✅ **20-30个密集摆放的器械/图**
- ✅ **细长器械的精确分割**
- ✅ **遮挡和重叠处理**
- ✅ **多尺度特征 (res2-res5)**

---

## 📋 使用步骤

### 第一步: 下载预训练权重 ⬇️

```bash
# 下载 ConvNeXt-Tiny ImageNet-1K 预训练权重
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth

# 转换为 Detectron2 格式
python tools/convert_convnext_to_d2.py \
    --source convnext_tiny_1k_224_ema.pth \
    --output convnext_tiny_1k_224_d2.pkl
```

### 第二步: 准备数据集 📁

#### 数据集结构
```
medical_instruments/
├── annotations/
│   ├── instances_train.json  # COCO 格式训练集标注
│   └── instances_val.json    # COCO 格式验证集标注
├── train/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── val/
    ├── img_101.jpg
    └── ...
```

#### 注册数据集

方法1: 在 `train_net.py` 中添加:
```python
# 在文件开头
import sys
sys.path.insert(0, 'datasets')
from register_medical_instruments import register_all_medical_instruments

# 在 main() 函数开始处
def main(args):
    register_all_medical_instruments(
        data_root="E:/path/to/your/medical_instruments",  # 修改为你的路径
        num_classes=500
    )
    # ... 原有代码 ...
```

方法2: 创建单独的注册脚本:
```python
# register_my_dataset.py
from datasets.register_medical_instruments import register_all_medical_instruments

register_all_medical_instruments(
    data_root="E:/path/to/your/medical_instruments",
    num_classes=500
)
```

### 第三步: 修改配置文件 ⚙️

编辑 `configs/dimaskdino_convnext_tiny_medical_instruments.yaml`:

```yaml
MODEL:
  # 设置预训练权重路径
  WEIGHTS: "E:/path/to/convnext_tiny_1k_224_d2.pkl"

DATASETS:
  TRAIN: ("medical_instruments_train",)
  TEST: ("medical_instruments_val",)

SOLVER:
  # 根据数据集大小调整迭代次数
  # 公式: MAX_ITER = (图像数量 / batch_size) * epochs
  # 示例: 5000张图, batch_size=8, 50 epochs
  # MAX_ITER = (5000/8) * 50 = 31250
  MAX_ITER: 31250
  STEPS: (25000, 28125)  # 在第40和45 epoch降低学习率
  
  IMS_PER_BATCH: 8  # 如果显存不足,可以降到4

OUTPUT_DIR: "./output/medical_instruments_convnext_tiny"
```

### 第四步: 开始训练 🏃

```bash
# 单GPU训练
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --num-gpus 1

# 多GPU训练 (例如4个GPU)
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --num-gpus 4
```

### 第五步: 评估模型 📊

```bash
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --eval-only \
    MODEL.WEIGHTS output/medical_instruments_convnext_tiny/model_final.pth
```

---

## 💡 重要参数说明

### 训练迭代次数计算

根据你的数据集大小计算 `MAX_ITER`:

```python
MAX_ITER = (训练图像数量 / batch_size) * epochs

# 示例:
# 3000张图, batch_size=8, 50 epochs: MAX_ITER = (3000/8) * 50 = 18750
# 5000张图, batch_size=8, 50 epochs: MAX_ITER = (5000/8) * 50 = 31250
# 10000张图, batch_size=8, 50 epochs: MAX_ITER = (10000/8) * 50 = 62500
```

学习率衰减步骤 (建议在第40和45 epoch):
```python
STEPS = (MAX_ITER * 0.8, MAX_ITER * 0.9)
```

### 显存优化

如果显存不足 (OOM):
```yaml
SOLVER:
  IMS_PER_BATCH: 4  # 降低batch size

INPUT:
  IMAGE_SIZE: 896  # 降低输入分辨率

MODEL:
  MaskDINO:
    NUM_OBJECT_QUERIES: 300  # 减少查询数
    TRAIN_NUM_POINTS: 12544  # 减少采样点
```

---

## 🔍 监控训练

### 关键指标

训练时关注以下指标:
- `total_loss`: 总损失,应平稳下降
- `loss_ce`: 分类损失
- `loss_mask`: Mask损失
- `loss_dice`: Dice损失
- `loss_bbox`: 边界框损失

### TensorBoard可视化

```bash
tensorboard --logdir output/medical_instruments_convnext_tiny
```

---

## 📈 预期性能

### 模型规模
- **参数量**: ~28M (ConvNeXt-Tiny) + ~30M (decoder) ≈ 60M
- **显存需求**: 
  - 训练: ~10-12GB (batch_size=8, 1024×1024)
  - 推理: ~4-6GB

### 训练时间 (单RTX 3090)
- **速度**: ~0.5s/iter
- **50 epochs (31K iters)**: ~4-5小时

### 性能指标 (预期)
- **mAP@0.5**: 75-85%
- **mAP@0.75**: 60-70%
- **mAP**: 65-75%

*实际性能取决于数据集质量、标注精度和训练时长*

---

## 🐛 常见问题

### Q1: 显存不足 (CUDA out of memory)
**解决方案**:
```yaml
SOLVER:
  IMS_PER_BATCH: 4  # 或更小
INPUT:
  IMAGE_SIZE: 896   # 或更小
```

### Q2: 训练loss不下降
**检查**:
1. 学习率是否合适 (默认0.0001)
2. 数据集是否正确加载
3. 预训练权重是否正确加载

**尝试**:
```yaml
SOLVER:
  BASE_LR: 0.0002  # 增加学习率
  WARMUP_ITERS: 1000  # 增加warmup
```

### Q3: 细长器械分割不准确
**优化**:
```yaml
MODEL:
  MaskDINO:
    TRAIN_NUM_POINTS: 20480  # 增加采样点
    MASK_WEIGHT: 7.0
    DICE_WEIGHT: 7.0
INPUT:
  IMAGE_SIZE: 1280  # 提高分辨率
```

### Q4: 遮挡处理效果差
**优化**:
```yaml
INPUT:
  DATASET_MAPPER_NAME: "coco_instance_copy_paste"  # 启用copy-paste增强

MODEL:
  MaskDINO:
    DEC_LAYERS: 12  # 增加解码器层数
    OVERLAP_THRESHOLD: 0.5  # 降低重叠阈值
```

---

## 📚 更多信息

- **详细文档**: `configs/CONVNEXT_MEDICAL_README.md`
- **集成总结**: `CONVNEXT_INTEGRATION_SUMMARY.md`
- **ConvNeXt论文**: https://arxiv.org/abs/2201.03545

---

## ✨ 下一步建议

1. **数据增强**: 如果数据量少,启用 copy-paste 增强
2. **模型调优**: 根据验证集表现调整超参数
3. **类别平衡**: 如果类别不平衡,考虑使用类别采样
4. **测试时增强**: 使用 TTA 提升最终性能

---

## 🎉 开始训练!

现在你已经准备好了!按照上述步骤开始训练,祝你取得好成绩!

如有问题,请查看详细文档或提issue。Good luck! 🚀
