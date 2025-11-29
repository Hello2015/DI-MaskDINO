# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Custom data loader with Copy-Paste augmentation support

import logging
from typing import List

import torch
from detectron2.data import (
    DatasetFromList,
    MapDataset,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.build import (
    _train_loader_from_config,
    build_batch_data_loader,
    trivial_batch_collator,
)
from detectron2.data.common import DatasetFromList
from detectron2.data.samplers import TrainingSampler
from torch.utils.data.sampler import BatchSampler

from .dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

logger = logging.getLogger(__name__)


def build_custom_train_loader(cfg, mapper=None):
    """
    Build a custom train loader with Copy-Paste augmentation support
    """
    if mapper is None:
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
    
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    
    dataset = DatasetFromList(dataset_dicts, copy=False)
    
    # Check if mapper supports batch processing
    if hasattr(mapper, 'map_batch'):
        # Use batch processing for Copy-Paste augmentation
        return _build_batch_aware_loader(cfg, dataset, mapper)
    else:
        # Fall back to standard loader
        return build_detection_train_loader(cfg, mapper=mapper)


def _build_batch_aware_loader(cfg, dataset, mapper):
    """
    Build a data loader that supports batch-level augmentation
    """
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using sampler {} for training".format(sampler_name))
    
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    
    # Create a batch sampler
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    
    # Create a dataset that applies batch-level transformations
    batch_dataset = _MapDatasetBatch(dataset, mapper, batch_size)
    
    data_loader = torch.utils.data.DataLoader(
        batch_dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=None,
    )
    
    return data_loader


class _MapDatasetBatch(torch.utils.data.Dataset):
    """
    A dataset that applies batch-level transformations
    """
    
    def __init__(self, dataset, mapper, batch_size):
        self.dataset = dataset
        self.mapper = mapper
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __getitem__(self, idx):
        """
        Get a batch of data and apply batch-level transformations
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))
        
        # Get individual samples
        batch_dicts = [self.dataset[i] for i in range(start_idx, end_idx)]
        
        # Apply batch-level transformations if available
        if hasattr(self.mapper, 'map_batch'):
            batch_dicts = self.mapper.map_batch(batch_dicts)
        else:
            # Fall back to individual mapping
            batch_dicts = [self.mapper(d) for d in batch_dicts]
        
        return batch_dicts


def build_custom_train_loader_from_config(cfg):
    """
    Build a custom train loader from config
    """
    return build_custom_train_loader(cfg)