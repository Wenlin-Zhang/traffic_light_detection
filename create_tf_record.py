
import hashlib
import random
import cv2
import os
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from tqdm import tqdm
from glob import glob
from read_label_file import get_all_labels

# Set the app flags up
flags = tf.app.flags
flags.DEFINE_string('data_dir', 'data', 'Root directory to traffic lights dataset.')
flags.DEFINE_string('output_dir', 'data', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'config/traffic_lights_label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS

def read_bosch_dataset(input_yaml):
    """Read the bosch dataset into a sample list.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data and removes small ones.

    Args:
        input_yaml: the input yaml file path (train.yaml, test.yaml or additional_train.yaml).
               You should download the Bosch Small Traffic Lights Dataset 
               (https://hci.iwr.uni-heidelberg.de/node/6132), and put the data in a folder
               which contains the following structure:
               data
               ├── train.yaml
               ├── test.yaml
               ├── additional_train.yaml
               ├── rgb
               │   ├── train
               │   ├── test
               │   ├── additional
               │   │   ├── 2015-10-05-10-52-01_bag
               │   │   │   ├── 24594.png
               │   │   │   ├── 24664.png
               │   │   │   └── 24734.png
               │   │   ├── 2015-10-05-10-55-33_bag
               │   │   │   ├── 56988.png
               │   │   │   ├── 57058.png
               ...
               Note that the image paths contained in the test.yaml files should be modified 
               to the relative path to the data folder.

    Returns:
      samples: a list of samples, each sample is a dict which have the following structure:
             sample = {'filename': filename, 'format': 'PNG', 'width': width, 'height': height,
                       'xmins': xmins, 'xmaxs': xmaxs, 'ymins': ymins, 'ymaxs': ymaxs}
             where width and height are integers and xmins, xmaxs, ymins, ymaxs are float list 
             constaining the bounding boxes' coordinates.

    """

    # Read the input yaml file
    records = get_all_labels(input_yaml)

    # Loop over the images
    samples = []
    for record in tqdm(records, desc='reading ' + input_yaml, unit='samples'):

        # Read the image
        filename = os.path.join(FLAGS.data_dir, record['path'])
        if not os.path.exists(filename):
            continue
        image = cv2.imread(filename)
        width = image.shape[1]
        height = image.shape[0]

        # Enumerate all boxes and calculate normalized size
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for box_info in record['boxes']:
            # ignore occluded or no label
            if box_info['occluded'] or box_info['label'] == 'off':
                continue
            # ignore too small bbox
            xmin = box_info['x_min']
            xmax = box_info['x_max']
            ymin = box_info['y_min']
            ymax = box_info['y_max']
            if ymin < 0 or ymax < 0 or xmin < 0 or xmax < 0 or ymax - ymin < 0.01 * height or xmax - xmin < 0.01 * width:
                continue
            # calculate the normalized bbox size
            xmins.append(float(xmin)/width)
            xmaxs.append(float(xmax)/width)
            ymins.append(float(ymin)/height)
            ymaxs.append(float(ymin)/height)
        if len(xmins) == 0:
            continue

        # construct the sample
        sample = {
            'filename': filename,
            'format': 'PNG',
            'width': width,
            'height': height,
            'xmins': xmins,
            'xmaxs': xmaxs,
            'ymins': ymins,
            'ymaxs': ymaxs,
        }
        samples.append(sample)

    return samples

def read_udacity_dataset(image_folder):
    image_files = glob(image_folder + '/*.png')
    samples = []
    for image_fn in tqdm(image_files, desc='udacity', unit='samples'):
        if not os.path.exists(image_fn):
            continue
        # get the label file path
        label_fn = os.path.splitext(image_fn)[0] + ".txt"
        if not os.path.exists(label_fn):
            continue

        # read the label file, obtain box info
        with open(label_fn, 'r') as f:
            boxes_info = f.readlines()[1:]
        boxes_info = list(map(lambda x: x.split(), boxes_info))

        # read the image file
        image = cv2.imread(image_fn)
        width = image.shape[1]
        height = image.shape[0]
 
        # decode the bound box info
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for box_info in boxes_info:
            xmins.append(float(box_info[0]) / width)
            xmaxs.append(float(box_info[2]) / width)
            ymins.append(float(box_info[1]) / height)
            ymaxs.append(float(box_info[3]) / height)
        if len(xmins) == 0:
            continue
        
        # construct the sample
        sample = {
            'filename': image_fn,
            'format': 'PNG',
            'width': width,
            'height': height,
            'xmins': xmins,
            'xmaxs': xmaxs,
            'ymins': ymins,
            'ymaxs': ymaxs,
        }
        samples.append(sample)

    return samples

# convert a sample to tf example
def sample_to_tf_example(sample):

    # Read the raw image file
    with open(sample['filename'], 'rb') as f:
        encoded = f.read()
    key = hashlib.sha256(encoded).hexdigest()

    bbox_count = len(sample['xmins'])

    # Create the example object
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(sample['height']),
        'image/width': dataset_util.int64_feature(sample['width']),
        'image/filename': dataset_util.bytes_feature(sample['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(sample['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded),
        'image/format': dataset_util.bytes_feature(sample['format'].encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(sample['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(sample['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(sample['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(sample['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature(['traffic-lights'.encode('utf8')] * bbox_count),
        'image/object/class/label': dataset_util.int64_list_feature([1] * bbox_count),
    }))
    return example

# write sample list to tf record file
def create_tf_record(output_filename, samples):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for sample in tqdm(samples, desc='creating ' + output_filename, unit='samples'):
        tf_example = sample_to_tf_example(sample)
        writer.write(tf_example.SerializeToString())
    writer.close()

#-------------------------------------------------------------------------------
def main(_):
    # read the bosch data
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    bosch_train = read_bosch_dataset(FLAGS.data_dir + "/train.yaml")
    print('number of bosch training samples: ', len(bosch_train))
    #bosch_train_additional = read_bosch_dataset(FLAGS.data_dir + "/additional_train.yaml")
    #print('number of bosch additional training samples: ', len(bosch_train_additional))
    #test_samples = read_bosch_dataset(FLAGS.data_dir + "/test.yaml")
    #print('number of bosch test samples: ', len(test_samples))
    # read the udacity data
    udacity_samples = []
    for subdir in ["red", "green", "yellow", "unknown"]:
        udacity_samples = udacity_samples + read_udacity_dataset(os.path.join(FLAGS.data_dir, "udacity", subdir))
    print('number of udacity samples: ', len(udacity_samples))

    # Split the whole training data into training/validation sets
    #samples = bosch_train + udacity_samples
    #num_samples = len(samples)
    #num_train = int(0.9 * num_samples)
    #train_samples = samples[:num_train]
    #val_samples = samples[num_train:]
    #print('split the whole training data to %d training samples and  %d validation samples.',
    #             len(train_samples), len(val_samples))
    # Create the record files
    #train_record_fname = os.path.join(FLAGS.output_dir, 'train.record')
    #val_record_fname = os.path.join(FLAGS.output_dir, 'val.record')
    #create_tf_record(train_record_fname, train_samples)
    #create_tf_record(val_record_fname, val_samples)

    # train with all bosch train and udacity samples
    train_set = bosch_train + udacity_samples
    random.shuffle(train_set)
    train_record_fname = os.path.join(FLAGS.output_dir, 'train.record')
    create_tf_record(train_record_fname, train_set)

    # Split the whole training data into training/validation sets
    # bosch_whole = bosch_train + bosch_train_additional
    # random.shuffle(bosch_whole)
    #num_samples = len(bosch_whole)
    #num_train = int(0.9 * num_samples)
    #train_samples = bosch_whole[:num_train]
    #val_samples = bosch_whole[num_train:]
    #print('split the whole training data to %d training samples and  %d validation samples.',
    #             len(train_samples), len(val_samples))
    # Create the record files
    #train_record_fname = os.path.join(FLAGS.output_dir, 'train.record')
    #val_record_fname = os.path.join(FLAGS.output_dir, 'val.record')
    #test_record_fname = os.path.join(FLAGS.output_dir, 'test.record')
    #create_tf_record(train_record_fname, train_samples)
    #create_tf_record(val_record_fname, val_samples)
    #create_tf_record(test_record_fname, test_samples)
#-------------------------------------------------------------------------------
if __name__ == '__main__':
  tf.app.run()
