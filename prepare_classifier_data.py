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
                if xmin <= 0 or xmax <= 0 or ymin <= 0 or ymax <= 0 or xmax-xmin<=8 or ymax-ymin<=13:
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

    (images, labels) = extract_bbox_images(['data/train.yaml', 'data/test.yaml', 'data/additional_train.yaml'], label_dict)
    #(images, labels) = extract_bbox_images(['data/additional_train.yaml'], label_dict)
    summary(labels, label_dict, 'all')

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
    
        
    
