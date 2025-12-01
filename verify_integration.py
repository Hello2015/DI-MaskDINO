#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单验证 ConvNeXt 集成是否正确
不需要安装依赖,仅检查文件结构
"""

import os
import sys

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - 文件不存在!")
        return False

def check_file_content(filepath, search_strings, description):
    """检查文件是否包含特定内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for search_str in search_strings:
            if search_str in content:
                print(f"  ✓ 包含: {search_str}")
            else:
                print(f"  ✗ 缺失: {search_str}")
                return False
        return True
    except Exception as e:
        print(f"  ✗ 读取文件失败: {e}")
        return False

def main():
    print("=" * 70)
    print("ConvNeXt-Tiny 集成验证")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_passed = True
    
    print("\n1. 检查核心实现文件...")
    files_to_check = [
        (os.path.join(base_dir, "dimaskdino/modeling/backbone/convnext.py"), "ConvNeXt backbone 实现"),
        (os.path.join(base_dir, "dimaskdino/config.py"), "配置文件"),
        (os.path.join(base_dir, "dimaskdino/modeling/backbone/__init__.py"), "Backbone __init__"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            all_passed = False
    
    print("\n2. 检查配置文件...")
    config_files = [
        (os.path.join(base_dir, "configs/dimaskdino_convnext_tiny_medical_instruments.yaml"), "医疗器械配置"),
        (os.path.join(base_dir, "configs/CONVNEXT_MEDICAL_README.md"), "使用文档"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            all_passed = False
    
    print("\n3. 检查工具脚本...")
    tool_files = [
        (os.path.join(base_dir, "tools/convert_convnext_to_d2.py"), "权重转换脚本"),
        (os.path.join(base_dir, "tools/test_convnext_backbone.py"), "集成测试脚本"),
    ]
    
    for filepath, desc in tool_files:
        if not check_file_exists(filepath, desc):
            all_passed = False
    
    print("\n4. 检查数据集注册...")
    if not check_file_exists(
        os.path.join(base_dir, "datasets/register_medical_instruments.py"),
        "数据集注册脚本"
    ):
        all_passed = False
    
    print("\n5. 检查关键代码...")
    
    # 检查 convnext.py
    print("\n  检查 ConvNeXt backbone:")
    if not check_file_content(
        os.path.join(base_dir, "dimaskdino/modeling/backbone/convnext.py"),
        ["class D2ConvNeXt", "BACKBONE_REGISTRY.register()", "class ConvNeXt"],
        "ConvNeXt 实现"
    ):
        all_passed = False
    
    # 检查 config.py
    print("\n  检查配置项:")
    if not check_file_content(
        os.path.join(base_dir, "dimaskdino/config.py"),
        ["cfg.MODEL.CONVNEXT", "DEPTHS", "DIMS", "DROP_PATH_RATE"],
        "ConvNeXt 配置"
    ):
        all_passed = False
    
    # 检查 __init__.py
    print("\n  检查 backbone 导出:")
    if not check_file_content(
        os.path.join(base_dir, "dimaskdino/modeling/backbone/__init__.py"),
        ["from .convnext import D2ConvNeXt", "D2ConvNeXt"],
        "Backbone 注册"
    ):
        all_passed = False
    
    # 检查配置文件
    print("\n  检查医疗器械配置:")
    if not check_file_content(
        os.path.join(base_dir, "configs/dimaskdino_convnext_tiny_medical_instruments.yaml"),
        ["D2ConvNeXt", "NUM_CLASSES: 500", "NUM_OBJECT_QUERIES: 400"],
        "医疗器械配置"
    ):
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有验证通过!")
        print("ConvNeXt-Tiny 已成功集成到 DI-MaskDINO 项目中")
        print("\n下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 下载预训练权重并转换")
        print("3. 准备数据集")
        print("4. 开始训练")
        print("\n详细说明请查看: configs/CONVNEXT_MEDICAL_README.md")
    else:
        print("✗ 部分验证失败,请检查上述错误")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
