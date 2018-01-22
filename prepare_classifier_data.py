#!/usr/bin/env python
"""
Prepare labelled traffic light images for classifier training
"""

import sys
import os
import cv2
import random
import pickle
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

from read_label_file import get_all_labels

def ir(some_value):
    """Int-round function for short array indexing """
    return int(round(some_value))

def extract_bbox_images(input_yaml_list, label_dict):
    images = []
    labels = []
    for input_yaml in input_yaml_list:
        if not os.path.exists(input_yaml):
            raise IOError('Could not find yaml path', input_yaml)

        samples = get_all_labels(input_yaml)

        for image_dict in tqdm(samples, desc='reading ' + input_yaml, unit='samples'):
            image = cv2.imread(image_dict['path'])
            if image is None:
                raise IOError('Could not open image path', image_dict['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width = image.shape[1]
            height = image.shape[0]
            for box in image_dict['boxes']:
                # ignore occluded or label not in the label map
                if box['occluded']:
                    continue
                label = box['label'].lower()
                if label not in label_dict:
                    continue
                # extract bbox size
                xmin = ir(box['x_min'])
                ymin = ir(box['y_min'])
                xmax = ir(box['x_max'])
                ymax = ir(box['y_max'])
                if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0 or ymax - ymin < 0.01 * height or xmax - xmin < 0.01 * width:
                    continue
                roi = image[ymin:(ymax+1), xmin:(xmax+1)]
                (h, w) = roi.shape[:2]
                if h == 0 or w == 0:
                    continue
                #print("h = {0}, w = {0}".format(h, w))
                image = cv2.resize(roi, (32, 32), interpolation = cv2.INTER_CUBIC) 
                images.append(image)
                labels.append(label_dict[label])
    return (images, labels)

def extract_bbox_images_only(in_dir):
    files = glob(in_dir + "/*.png")
    images = []
    for image_fn in tqdm(files, desc='reading ' + in_dir, unit='samples'):
        image = cv2.imread(image_fn)
        if image is None:
            raise IOError('Could not open image file: ', image_fn)

        # get the label file path
        label_fn = os.path.splitext(image_fn)[0] + ".txt"
        if not os.path.exists(label_fn):
            print("Cannot find label file: ", label_fn)
            continue
        # read the label file, obtain box info
        with open(label_fn, 'r') as f:
            boxes_info = f.readlines()[1:]
        boxes_info = list(map(lambda x: x.split(), boxes_info))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]
        for box_info in boxes_info:
            xmin = int(box_info[0])
            xmax = int(box_info[2])
            ymin = int(box_info[1])
            ymax = int(box_info[3])
            if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0 or ymax - ymin < 0.01 * height or xmax - xmin < 0.01 * width:
                continue
            roi = image[ymin : (ymax+1), xmin : (xmax+1)]
            (h, w) = roi.shape[:2]
            if h == 0 or w == 0:
                 continue
            image = cv2.resize(roi, (32, 32), interpolation = cv2.INTER_CUBIC) 
            images.append(image)

    return images

def summary(labels, label_dict, name):
    label_count= {}
    for label in label_dict:
        label_count[label_dict[label]] = 0
    for label in labels:
        label_count[label] += 1
    print("============ {0} set summary, count = {1}".format(name, len(labels)))
    for label in label_count:
        print("{0} : {1}".format(label, label_count[label]))

if __name__ == '__main__':
    label_dict = {'green':0, 'red':1, 'yellow':2, 'off':3}

    #(images, labels) = extract_bbox_images(['data/train.yaml', 'data/test.yaml', 'data/additional_train.yaml'], label_dict)
    #(images, labels) = extract_bbox_images(['data/train.yaml', 'data/additional_train.yaml'], label_dict)
    #(images, labels) = extract_bbox_images(['data/additional_train.yaml'], label_dict)
    (images, labels) = extract_bbox_images(['data/train.yaml'], label_dict)
    summary(labels, label_dict, 'bosch')

    udacity_images_g = extract_bbox_images_only('data/udacity/green')
    udacity_labels_g = [label_dict['green']] * len(udacity_images_g)
    udacity_images_r = extract_bbox_images_only('data/udacity/red')
    udacity_labels_r = [label_dict['red']] * len(udacity_images_r)
    udacity_images_y = extract_bbox_images_only('data/udacity/yellow')
    udacity_labels_y = [label_dict['yellow']] * len(udacity_images_y)
    udacity_images = udacity_images_g + udacity_images_r + udacity_images_y
    udacity_labels = udacity_labels_g + udacity_labels_r + udacity_labels_y
    summary(udacity_labels, label_dict, 'udacity')

    images += udacity_images
    labels += udacity_labels

    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.15)
    print("============ train test split, train count = {0}, val count = {1}".format(len(X_train), len(X_val)))

    data_dir = 'data/classifier'
    if not os.path.exists(data_dir):
      os.makedirs(train_dir)

    summary(Y_train, label_dict, 'train')
    output = open('data/classifier/train.pkl', 'wb')
    pickle.dump({"images" : X_train, "labels" : Y_train}, output)
    output.close()
    
    summary(Y_val, label_dict, 'val')
    output = open('data/classifier/val.pkl', 'wb')
    pickle.dump({"images" : X_val, "labels" : Y_val}, output)
    output.close()
    
        
    
