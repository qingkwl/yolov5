# Modified from https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from PIL import ExifTags

# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def make_dirs(dir_name='new_dir/'):
    # Create folders
    dir_name = Path(dir_name)
    if dir_name.exists():
        shutil.rmtree(dir_name)  # delete dir
    for p in (dir_name, dir_name / 'labels', dir_name / 'images'):
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir_name


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x

class COCO2YOLO:
    def __init__(self, json_dir: str, save_dir: str, use_segments=False, class_id_map=None):
        self.json_dir = json_dir
        self.save_dir = make_dirs(save_dir)
        self.class_id_map = class_id_map
        self.use_segments = use_segments

    def __call__(self):
        json_dir = self.json_dir
        save_dir = self.save_dir
        class_id_map = self.class_id_map
        use_segments = self.use_segments
        for json_file in sorted(Path(json_dir).resolve().glob('instances_*.json')):
            folder_name = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
            folder_name.mkdir()
            with open(json_file) as file:
                data = json.load(file)

            # Create image dict
            images = {'%g' % x['id']: x for x in data['images']}
            # Create image-annotations dict
            imgToAnns = defaultdict(list)
            for ann in data['annotations']:
                imgToAnns[ann['image_id']].append(ann)

            # Write labels file
            for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
                img = images['%g' % img_id]
                h, w, file_name = img['height'], img['width'], img['file_name']

                bboxes = []
                segments = []
                for ann in anns:
                    if ann['iscrowd']:
                        continue
                    category_id = class_id_map[ann['category_id'] - 1] if class_id_map is not None \
                                  else ann['category_id'] - 1  # class
                    # The COCO box format is [top left x, top left y, width, height]
                    box = self.transfer_bbox(annotation=ann, width=w, height=h)
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    box = [category_id] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if use_segments:
                        if len(ann['segmentation']) > 1:
                            s = merge_multi_segment(ann['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [category_id] + s
                        if s not in segments:
                            segments.append(s)
                self.write_file(folder_name / file_name, bboxes, segments)

    @staticmethod
    def transfer_bbox(annotation, width, height):
        box = np.array(annotation['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= width  # normalize x
        box[[1, 3]] /= height  # normalize y
        return box

    def write_file(self, file_path: Path, bboxes, segments):
        # Write
        with open(file_path.with_suffix('.txt'), 'a') as file:
            for i in range(len(bboxes)):
                line = *(segments[i] if self.use_segments else bboxes[i]),  # cls, box or segments
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert labels of coco format to yolo format.")
    parser.add_argument('--json_dir', type=str, help="Directory which contains COCO annotations json files.")
    parser.add_argument('--save_dir', type=str, help="Directory to save labels of yolo format.")
    args = parser.parse_args()
    coco2yolo = COCO2YOLO(args.json_dir,  # directory with *.json
                          args.save_dir,
                          use_segments=False,
                          class_id_map=coco91_to_coco80_class())
    coco2yolo()
