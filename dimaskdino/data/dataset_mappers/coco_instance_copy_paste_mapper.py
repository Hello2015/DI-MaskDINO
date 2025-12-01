# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from COCOInstanceNewBaselineDatasetMapper with Copy-Paste augmentation

import copy
import logging
import random

import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, PolygonMasks

from pycocotools import mask as coco_mask

# 导入自定义数据增强
from ..augmentations.copy_paste_augmentation import CopyPasteAugmentation

__all__ = ["COCOInstanceCopyPasteDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation


class COCOInstanceCopyPasteDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer with Copy-Paste augmentation.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation
    with additional Copy-Paste data augmentation.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        mask_format="polygon",
        copy_paste_prob=0.5,
        use_poisson_blend=True,
        max_copy_objects=3,
        batch_size=2,
    ):
        """
        Args:
            is_train: for training or inference
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
            mask_format: format of masks, either "polygon" or "bitmask"
            copy_paste_prob: Copy-Paste增强的应用概率
            use_poisson_blend: 是否使用泊松融合
            max_copy_objects: 最大复制对象数量
            batch_size: 批处理大小，用于Copy-Paste增强
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceCopyPasteDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.mask_format = mask_format
        self.batch_size = batch_size
        
        # 初始化Copy-Paste增强
        self.copy_paste_aug = None
        if is_train and copy_paste_prob > 0:
            self.copy_paste_aug = CopyPasteAugmentation(
                prob=copy_paste_prob,
                max_objects=max_copy_objects,
                use_poisson=use_poisson_blend
            )
            logging.getLogger(__name__).info(
                f"[COCOInstanceCopyPasteDatasetMapper] Copy-Paste augmentation enabled with prob={copy_paste_prob}"
            )
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
            "mask_format": cfg.INPUT.MASK_FORMAT,
        }
        
        # 添加Copy-Paste增强配置
        if hasattr(cfg.INPUT, 'COPY_PASTE_PROB'):
            ret["copy_paste_prob"] = cfg.INPUT.COPY_PASTE_PROB
        else:
            ret["copy_paste_prob"] = 0.5  # 默认概率
            
        if hasattr(cfg.INPUT, 'USE_POISSON_BLEND'):
            ret["use_poisson_blend"] = cfg.INPUT.USE_POISSON_BLEND
        else:
            ret["use_poisson_blend"] = True  # 默认使用泊松融合
            
        if hasattr(cfg.INPUT, 'MAX_COPY_OBJECTS'):
            ret["max_copy_objects"] = cfg.INPUT.MAX_COPY_OBJECTS
        else:
            ret["max_copy_objects"] = 3  # 默认最大对象数
            
        if hasattr(cfg.DATALOADER, 'BATCH_SIZE'):
            ret["batch_size"] = cfg.DATALOADER.BATCH_SIZE
        else:
            ret["batch_size"] = 2  # 默认批大小
            
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Process annotations based on mask format
            if self.mask_format == "bitmask":
                # For bitmask format, use utils.annotations_to_instances with mask_format argument
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                # if not instances.has('gt_masks'):  # this is to avoid empty annotation
                # instances.gt_masks = BitMasks(torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8))
                instances.gt_masks = instances.gt_masks.tensor
                # Need to filter empty instances first (due to augmentation)
                # instances = utils.filter_empty_instances(instances)
            else:
                # Default to polygon format
                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                instances = utils.annotations_to_instances(annos, image_shape)
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if not instances.has('gt_masks'):  # this is to avoid empty annotation
                    instances.gt_masks = PolygonMasks([])
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)
                # Generate masks from polygon
                h, w = instances.image_size
                if hasattr(instances, 'gt_masks'):
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict

    def map_batch(self, dataset_dicts):
        """
        批处理映射函数，用于应用Copy-Paste增强
        Args:
            dataset_dicts: 一批数据集字典
        Returns:
            增强后的数据集字典列表
        """
        # 首先对每个样本应用基本变换
        processed_dicts = [self(dataset_dict) for dataset_dict in dataset_dicts]
        
        # 如果启用了Copy-Paste增强，则应用增强
        if self.copy_paste_aug is not None and len(processed_dicts) >= 2:
            processed_dicts = self.copy_paste_aug(processed_dicts)
        
        return processed_dicts