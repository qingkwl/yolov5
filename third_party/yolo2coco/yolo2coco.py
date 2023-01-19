import os
import numpy as np
import json
import cv2
import sys
id_counter = 0 # To record the id
out = {'annotations': [], 
           'categories': [], ##### change the categories to match your dataset!
           'images': [],
           'info': {"contributor": "", "year": "", "version": "", "url": "", "description": "", "date_created": ""},
           'licenses': {"id": 0, "name": "", "url": ""}
           }

def categories_data(categories_list):
    if isinstance(categories_list, list):
        categories_count = 1
        for category in categories_list: # if txt.readlines is null, this for loop would not run
            cat = {'id': categories_count,
                'name': category, 
                "supercategory": ""           
            }
            out['categories'].append(cat)
            categories_count = categories_count + 1 
    else:
        print('categories info not list')


def annotations_data(ann_dir, image_id, height, width):
    # id, bbox, iscrowd, image_id, category_id
    global id_counter
    with open(ann_dir + '.txt','r') as txt:
        for line in txt.readlines(): # if txt.readlines is null, this for loop would not run
            data = line.strip()
            data = data.split() 
            # convert the center into the top-left point!
            data[1] = float(data[1])* width - 0.5 * float(data[3])* width #
            data[2] = float(data[2])* height - 0.5 * float(data[4])* height #
            data[3] = float(data[3])* width #
            data[4] = float(data[4])* height #
            bbox = [data[1],data[2], data[3], data[4]]
            ann = {'id': id_counter,
                'bbox': bbox,
                'area': data[3] * data[4],
                'iscrowd': 0,
                'image_id': str(image_id),
                'category_id': int(data[0]) + 1            
            }
            out['annotations'].append(ann)
            id_counter = id_counter + 1 

def images_data(img_file, img_id):
    img = cv2.imread(img_file)
    height, width = img.shape[0], img.shape[1]
    imgs = {'id': str(img_id),
            'height': height, ##### change the 600 to your raw image height
            'width': width, ##### change the 800 to your raw image width
            'file_name':str(img_id + '.jpg'),
            "coco_url": "", 
            "flickr_url": "", 
            "date_captured": 0, 
            "license": 0
    }
    out['images'].append(imgs)
    return height, width

def yolo2coco(img_txt_file, categories_list, json_file):
    json_dir = json_file.replace('instances_val2017.json', '')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir) 
    categories_data(categories_list)
    with open(img_txt_file, 'r') as txt:
        for line in txt.readlines():
            line = line.replace('\r','').replace('\n','')
            img_id = line.split('/')[-1].split('.')[0]
            height, width = images_data(line, img_id)
            ann_dir = line.split('.')[0].replace('images', 'labels')
            annotations_data(ann_dir, img_id, height, width)

    with open(json_file,'w') as outfile: 
        json.dump(out, outfile, separators=(',', ':'))