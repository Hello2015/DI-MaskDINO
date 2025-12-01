# 🔧 修复权重加载错误

## ❌ 问题描述

当尝试加载转换后的ConvNeXt权重时，遇到以下错误：

```
ValueError: Unsupported type found in checkpoint! model: <class 'dict'>
```

## 🔍 根本原因

原始的 `convert_convnext_to_d2.py` 脚本使用 `pickle.dump()` 保存权重，但保存方式不正确：
- 使用pickle序列化的字典中的张量不是真正的PyTorch张量
- Detectron2的 `DetectionCheckpointer` 期望的是使用 `torch.save()` 保存的格式

## ✅ 解决方案

有两种方法修复这个问题:

### 方法1: 重新转换权重 (推荐)

使用更新的转换脚本重新转换权重：

```bash
python tools/convert_convnext_to_d2.py \
    --source convnext_tiny_1k_224_ema.pth \
    --output convnext_tiny_1k_224_d2_new.pkl
```

### 方法2: 修复已有的错误权重文件

如果你已经有一个用旧脚本转换的错误权重文件，可以使用修复脚本：

```bash
python tools/fix_convnext_weights.py \
    --input convnext_tiny_1k_224_d2.pkl \
    --output convnext_tiny_1k_224_d2_fixed.pkl
```

修复脚本会：
1. ✓ 加载错误格式的checkpoint
2. ✓ 转换所有值为正确的PyTorch张量
3. ✓ 使用 `torch.save()` 保存为正确格式
4. ✓ 验证修复后的权重文件

## 🚀 正确的使用流程

### 步骤1: 转换权重

使用更新的 `convert_convnext_to_d2.py` (已修复):

```bash
# 下载ConvNeXt-Tiny预训练权重
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth

# 转换为Detectron2格式
python tools/convert_convnext_to_d2.py \
    --source convnext_tiny_1k_224_ema.pth \
    --output convnext_tiny_1k_224_d2.pkl
```

### 步骤2: 修改配置文件

编辑 `configs/dimaskdino_convnext_tiny_medical_instruments.yaml`:

```yaml
MODEL:
  # 设置转换后的权重路径
  WEIGHTS: "tools/convnext_tiny_1k_224_d2.pkl"  # 或你的完整路径
```

### 步骤3: 开始训练

```bash
python train_net.py \
    --config-file configs/dimaskdino_convnext_tiny_medical_instruments.yaml \
    --num-gpus 1
```

## 📝 已更新的文件

### 1. `tools/convert_convnext_to_d2.py`
**修改内容**: 
- 使用 `torch.save()` 替代 `pickle.dump()`
- 确保所有张量都是正确的PyTorch Tensor格式
- 添加详细的注释说明

### 2. `tools/fix_convnext_weights.py` (新增)
**功能**:
- 修复已有的错误格式权重文件
- 支持加载pickle和torch格式
- 自动验证修复结果

## ⚠️ 注意事项

1. **权重格式**: Detectron2 `DetectionCheckpointer` 要求使用 `torch.save()` 保存的格式
2. **不兼容性**: 用pickle保存的权重与Detectron2不兼容
3. **验证方式**: 使用 `fix_convnext_weights.py` 可以验证权重是否正确

## 🧪 测试转换

在运行训练前，可以测试权重是否能正确加载：

```python
import torch

# 加载转换后的权重
checkpoint = torch.load("tools/convnext_tiny_1k_224_d2.pkl", map_location="cpu")

# 检查结构
assert "model" in checkpoint, "Missing 'model' key"
print(f"✓ Contains {len(checkpoint['model'])} weights")

# 检查张量类型
for key, value in list(checkpoint['model'].items())[:3]:
    assert isinstance(value, torch.Tensor), f"Value at {key} is not a tensor!"
    print(f"✓ {key}: {value.shape}")

print("✓ All checks passed!")
```

## 💡 常见问题

### Q: 转换后权重文件很小 (< 10MB)？
**A**: 这可能是因为用pickle保存的字典包含了错误的数据类型。使用修复脚本或重新转换。

### Q: 如何检查权重文件格式是否正确？
**A**: 
```bash
python -c "
import torch
ckpt = torch.load('your_weights.pkl', map_location='cpu')
print('Keys:', list(ckpt.keys()))
if 'model' in ckpt:
    model = ckpt['model']
    sample_key = list(model.keys())[0]
    sample_val = model[sample_key]
    print(f'Sample value type: {type(sample_val).__name__}')
    print(f'Sample value shape: {sample_val.shape if hasattr(sample_val, \"shape\") else \"N/A\"}')
"
```

### Q: 修复后仍然无法加载？
**A**: 
1. 检查原始的 `.pth` 文件是否完整
2. 尝试用 `torch.load()` 直接加载原始文件验证
3. 查看权重文件大小是否合理 (~120MB for ConvNeXt-Tiny)

---

## 📚 相关文件

- `tools/convert_convnext_to_d2.py` - 更新的权重转换脚本
- `tools/fix_convnext_weights.py` - 权重修复脚本
- `QUICKSTART_CN.md` - 中文快速入门指南

---

**更新时间**: 2025-12-01
