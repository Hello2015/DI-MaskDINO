#!/usr/bin/env python
# ------------------------------------------------------------------------
# Register Medical Surgical Instruments Dataset
# ------------------------------------------------------------------------
# This script registers your medical instruments dataset in COCO format.
# 
# Dataset structure should be:
# medical_instruments/
#   ├── annotations/
#   │   ├── instances_train.json
#   │   └── instances_val.json
#   ├── train/
#   │   └── *.jpg (or *.png)
#   └── val/
#       └── *.jpg (or *.png)
# ------------------------------------------------------------------------

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_medical_instruments_dataset(
    dataset_name,
    json_file,
    image_root,
    num_classes=500
):
    """
    Register a medical instruments dataset in COCO format.
    
    Args:
        dataset_name (str): Name to register the dataset (e.g., "medical_instruments_train")
        json_file (str): Path to COCO format JSON annotation file
        image_root (str): Path to directory containing images
        num_classes (int): Number of instrument categories (default: 500)
    """
    # Register the dataset
    register_coco_instances(
        dataset_name,
        {},
        json_file,
        image_root
    )
    
    # Set metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=[f"instrument_{i}" for i in range(num_classes)],  # Placeholder names
        evaluator_type="coco",
    )
    
    print(f"Registered dataset: {dataset_name}")
    print(f"  JSON: {json_file}")
    print(f"  Images: {image_root}")
    print(f"  Classes: {num_classes}")


def register_all_medical_instruments(data_root, num_classes=500):
    """
    Register train and validation splits of medical instruments dataset.
    
    Args:
        data_root (str): Root directory of the dataset
        num_classes (int): Number of instrument categories
    """
    # Register training set
    register_medical_instruments_dataset(
        "medical_instruments_train",
        os.path.join(data_root, "annotations", "instances_train.json"),
        os.path.join(data_root, "train"),
        num_classes=num_classes
    )
    
    # Register validation set
    register_medical_instruments_dataset(
        "medical_instruments_val",
        os.path.join(data_root, "annotations", "instances_val.json"),
        os.path.join(data_root, "val"),
        num_classes=num_classes
    )


# Example usage:
if __name__ == "__main__":
    # Modify this path to your dataset location
    DATA_ROOT = "/path/to/your/medical_instruments"
    NUM_CLASSES = 500
    
    register_all_medical_instruments(DATA_ROOT, NUM_CLASSES)
    
    # Verify registration
    from detectron2.data import DatasetCatalog
    print("\nRegistered datasets:")
    print(DatasetCatalog.list())


# To use this in your training:
# 1. Place this file in dimaskdino/data/datasets/
# 2. Import it in dimaskdino/data/datasets/__init__.py:
#    from .register_medical_instruments import register_all_medical_instruments
# 3. Call it before training with your dataset path
