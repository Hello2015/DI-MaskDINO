# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copy-Paste augmentation with Gaussian blur and Poisson blending

import random
import numpy as np
import cv2
import torch


def poisson_blend(source, target, mask, offset_x, offset_y):
    """
    使用泊松融合将源图像融合到目标图像
    Args:
        source: 源图像 [H, W, C]
        target: 目标图像 [H, W, C]
        mask: 融合区域mask [H, W]
        offset_x, offset_y: 融合位置偏移
    Returns:
        blended: 融合后的图像
    """
    h, w = target.shape[:2]
    src_h, src_w = source.shape[:2]
    
    # 计算融合区域
    x1, y1 = max(0, offset_x), max(0, offset_y)
    x2, y2 = min(w, offset_x + src_w), min(h, offset_y + src_h)
    
    # 调整源图像和mask大小
    src_crop_x1 = max(0, -offset_x)
    src_crop_y1 = max(0, -offset_y)
    src_crop_x2 = src_crop_x1 + (x2 - x1)
    src_crop_y2 = src_crop_y1 + (y2 - y1)
    
    if src_crop_x2 <= src_crop_x1 or src_crop_y2 <= src_crop_y1:
        return target
    
    source_crop = source[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
    mask_crop = mask[src_crop_y1:src_crop_y2, src_crop_x1:src_crop_x2]
    
    if source_crop.size == 0:
        return target
    
    # 使用OpenCV的泊松融合
    try:
        # 确保mask是uint8类型
        mask_uint8 = (mask_crop * 255).astype(np.uint8)
        
        # 创建目标区域副本
        target_region = target[y1:y2, x1:x2].copy()
        
        # 计算融合中心点
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        
        # 泊松融合
        blended_region = cv2.seamlessClone(
            source_crop, target_region, mask_uint8, center, cv2.NORMAL_CLONE
        )
        
        # 将融合后的区域放回目标图像
        result = target.copy()
        result[y1:y2, x1:x2] = blended_region
        return result
        
    except Exception as e:
        # 如果泊松融合失败，使用简单的alpha混合
        print(f"Poisson blending failed: {e}, using alpha blending instead")
        alpha = mask_crop[..., None]
        blended_region = source_crop * alpha + target_region * (1 - alpha)
        result = target.copy()
        result[y1:y2, x1:x2] = blended_region.astype(np.uint8)
        return result


class CopyPasteAugmentation:
    """Copy-Paste数据增强类"""
    
    def __init__(self, prob=0.5, max_objects=3, use_poisson=True):
        """
        Args:
            prob: 应用增强的概率
            max_objects: 最大粘贴对象数量
            use_poisson: 是否使用泊松融合
        """
        self.prob = prob
        self.max_objects = max_objects
        self.use_poisson = use_poisson
    
    def __call__(self, dataset_dicts):
        """
        对一批数据进行Copy-Paste增强
        Args:
            dataset_dicts: 数据集字典列表
        Returns:
            增强后的数据集字典列表
        """
        if random.random() > self.prob or len(dataset_dicts) < 2:
            return dataset_dicts
        
        # 随机选择源图像和目标图像
        target_idx = random.randint(0, len(dataset_dicts) - 1)
        source_idx = random.randint(0, len(dataset_dicts) - 1)
        
        while source_idx == target_idx:
            source_idx = random.randint(0, len(dataset_dicts) - 1)
        
        target_dict = dataset_dicts[target_idx]
        source_dict = dataset_dicts[source_idx]
        
        # 执行Copy-Paste增强
        augmented_dict = self._copy_paste_single(target_dict, source_dict)
        
        if augmented_dict is not None:
            dataset_dicts[target_idx] = augmented_dict
        
        return dataset_dicts
    
    def _copy_paste_single(self, target_dict, source_dict):
        """对单个目标图像执行Copy-Paste增强"""
        # 读取图像
        target_image = target_dict["image"]
        source_image = source_dict["image"]
        
        # 确保图像是numpy数组格式
        if isinstance(target_image, torch.Tensor):
            target_image = target_image.permute(1, 2, 0).numpy()
        if isinstance(source_image, torch.Tensor):
            source_image = source_image.permute(1, 2, 0).numpy()
        
        # 检查是否有实例标注
        if "instances" not in source_dict or "instances" not in target_dict:
            return None
        
        source_instances = source_dict["instances"]
        target_instances = target_dict["instances"]
        
        # 获取源图像的masks
        if not hasattr(source_instances, 'gt_masks') or source_instances.gt_masks.numel() == 0:
            return None
        
        source_masks = source_instances.gt_masks
        num_objects = min(source_masks.shape[0], self.max_objects)
        
        if num_objects == 0:
            return None
        
        # 随机选择要粘贴的对象
        selected_indices = random.sample(range(source_masks.shape[0]), num_objects)
        
        # 创建增强后的图像和实例
        augmented_image = target_image.copy()
        augmented_instances = self._copy_instances(target_instances)
        
        for idx in selected_indices:
            # 获取对象的mask
            mask = source_masks[idx].numpy()
            
            # 随机选择粘贴位置
            h, w = target_image.shape[:2]
            src_h, src_w = mask.shape
            
            offset_x = random.randint(0, w - src_w) if w > src_w else 0
            offset_y = random.randint(0, h - src_h) if h > src_h else 0
            
            # 提取源图像中的对象区域
            object_region = source_image * mask[..., None]
            
            # 使用泊松融合或简单粘贴
            if self.use_poisson and cv2.__version__ >= '3.0':
                augmented_image = poisson_blend(
                    object_region, augmented_image, mask, offset_x, offset_y
                )
            else:
                # 简单粘贴
                x1, y1 = max(0, offset_x), max(0, offset_y)
                x2, y2 = min(w, offset_x + src_w), min(h, offset_y + src_h)
                
                src_x1 = max(0, -offset_x)
                src_y1 = max(0, -offset_y)
                src_x2 = src_x1 + (x2 - x1)
                src_y2 = src_y1 + (y2 - y1)
                
                if src_x2 > src_x1 and src_y2 > src_y1:
                    object_crop = object_region[src_y1:src_y2, src_x1:src_x2]
                    mask_crop = mask[src_y1:src_y2, src_x1:src_x2]
                    
                    alpha = mask_crop[..., None]
                    augmented_image[y1:y2, x1:x2] = (
                        object_crop * alpha + augmented_image[y1:y2, x1:x2] * (1 - alpha)
                    )
            
            # 添加新的实例标注
            self._add_instance(augmented_instances, source_instances, idx, 
                              offset_x, offset_y, mask)
        
        # 更新数据字典
        augmented_dict = target_dict.copy()
        augmented_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(augmented_image.transpose(2, 0, 1))
        )
        augmented_dict["instances"] = augmented_instances
        
        return augmented_dict
    
    def _copy_instances(self, instances):
        """复制实例对象"""
        new_instances = Instances(instances.image_size)
        
        for field in instances._fields:
            if hasattr(instances, field):
                value = getattr(instances, field)
                if isinstance(value, torch.Tensor):
                    new_instances.set(field, value.clone())
                else:
                    new_instances.set(field, copy.deepcopy(value))
        
        return new_instances
    
    def _add_instance(self, target_instances, source_instances, idx, offset_x, offset_y, mask):
        """向目标实例添加新的实例"""
        # 获取源实例的属性
        fields_to_copy = ['gt_classes', 'gt_boxes', 'gt_masks']
        
        for field in fields_to_copy:
            if not hasattr(source_instances, field):
                continue
            
            source_value = getattr(source_instances, field)
            
            if field == 'gt_boxes':
                # 更新边界框位置
                boxes = source_value.tensor[idx:idx+1].clone()
                boxes[:, 0] += offset_x  # x1
                boxes[:, 1] += offset_y  # y1
                boxes[:, 2] += offset_x  # x2
                boxes[:, 3] += offset_y  # y2
                
                # 根据mask重新计算精确的bbox
                from dimaskdino.utils.box_ops import masks_to_boxes
                h, w = mask.shape
                y_coords, x_coords = np.where(mask > 0.5)
                if len(y_coords) > 0 and len(x_coords) > 0:
                    x_min, y_min = x_coords.min(), y_coords.min()
                    x_max, y_max = x_coords.max(), y_coords.max()
                    boxes[:, 0] = x_min + offset_x
                    boxes[:, 1] = y_min + offset_y
                    boxes[:, 2] = x_max + offset_x
                    boxes[:, 3] = y_max + offset_y
                
                if hasattr(target_instances, 'gt_boxes'):
                    new_boxes = torch.cat([target_instances.gt_boxes.tensor, boxes], dim=0)
                    target_instances.gt_boxes.tensor = new_boxes
                else:
                    from detectron2.structures import Boxes
                    target_instances.gt_boxes = Boxes(boxes)
            
            elif field == 'gt_masks':
                # 更新mask位置
                new_mask = torch.zeros_like(mask)
                h, w = mask.shape
                y_coords, x_coords = np.where(mask > 0.5)
                
                if len(y_coords) > 0 and len(x_coords) > 0:
                    for y, x in zip(y_coords, x_coords):
                        new_y = y + offset_y
                        new_x = x + offset_x
                        if 0 <= new_y < h and 0 <= new_x < w:
                            new_mask[new_y, new_x] = 1
                
                new_mask_tensor = torch.from_numpy(new_mask).unsqueeze(0)
                
                if hasattr(target_instances, 'gt_masks'):
                    target_instances.gt_masks = torch.cat([
                        target_instances.gt_masks, new_mask_tensor
                    ], dim=0)
                else:
                    target_instances.gt_masks = new_mask_tensor
            
            else:
                # 直接复制其他属性
                source_val = source_value[idx:idx+1].clone()
                
                if hasattr(target_instances, field):
                    target_val = getattr(target_instances, field)
                    new_val = torch.cat([target_val, source_val], dim=0)
                    setattr(target_instances, field, new_val)
                else:
                    setattr(target_instances, field, source_val)