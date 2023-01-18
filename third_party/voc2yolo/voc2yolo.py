import glob
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from os import getcwd

dirs = ['train', 'val']
classes = ['person', 'car']

class VOC2YOLO:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verify_exists(self.data_dir)
        self.verify_exists(self.output_dir)

    def __call__(self):
        img_dir = self.data_dir / "JPEGImages"
        ann_dir = self.data_dir / "Annotations"
        yolo_ann_dir = self.output_dir / "labels"
        self.verify_exists(img_dir)
        self.verify_exists(ann_dir)
        for img_path in img_dir.glob("*.jpg"):
            basename = img_path.stem
            ann_path = ann_dir / f"{basename}.xml"
            yolo_ann_path = yolo_ann_dir / f"{basename}.txt"
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            with open(yolo_ann_path, "w") as out_file:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                         float(xmlbox.find('ymax').text))
                    bb = self.convert_bbox(b, (w, h))
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    @staticmethod
    def convert_bbox(box, shape):
        dw = 1. / (shape[0])
        dh = 1. / (shape[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def verify_exists(file_path, strict=True):
        file_path = Path(file_path)
        if not file_path.exists():
            if strict:
                raise FileNotFoundError(f'[ERROR] The {file_path} is not exists!!!')
            else:
                print(f"[WARNING] The {file_path} is not exists.")
                return False
        return True
