import json

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

sysCoco = COCO_CATEGORIES.copy()

# 使用labels中的标签替换sysCoco中的name

def register_opcoco():
    register_coco_instances("opcoco38_train", {},
                            "coco/opcoco38/annotations/test.json",
                            "coco/opcoco38/images/test")
    register_coco_instances("opcoco38_val", {},
                            "coco/opcoco38/annotations/val.json",
                            "coco/opcoco38/images/val")
    # 读取datasets/opcoco/annotations/test.json,获取其中的category信息
    labels = json.load(open("coco/opcoco38/annotations/test.json", "r"))["categories"]

    for i in range(len(labels)):
        sysCoco[i]["name"] = labels[i]['name']
        sysCoco[i]["id"] = i

    while len(sysCoco) > len(labels):
        sysCoco.pop()
    thing_classes = [k["name"] for k in sysCoco]
    catlog = MetadataCatalog.get('opcoco38_val')
    catlog.set(thing_classes=thing_classes)

    print(sysCoco[:len(labels)])
