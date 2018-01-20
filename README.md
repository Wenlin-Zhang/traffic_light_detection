# traffic_light_detection
Implementation of a traffic light detector for the udacity sdc nanodegreee using tensorflow object detection API 

# 1. download the Bosch dataset
You should first download the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132), and 
rearrange the folder structure as following:

```
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
...
```

Note that the image paths contained in the test.yaml files should be modified 
to the relative path to the data folder.

# 2. install the tensorflow object detection API
You can find the tensorflow object detection API [here](https://github.com/tensorflow/models/tree/master/research/object_detection),
and follow the installation instruction [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). 

# 3. create the tfrecord data file
```
python create_tf_record.py
```

# 4. train the detection model

## 4.1 download the pre-trained models
download faster_rcnn_resnet101_coco_2017_11_08 and ssd_mobilenet_v1_coco_2017_11_17 model files from [here](http://download.tensorflow.org/models/object_detection), put the model files in `models` folder as following:
```
models
├── faster_rcnn_resnet101_coco_2017_11_08
│   ├── frozen_inference_graph.pb
│   ├── ...
├── ssd_mobilenet_v1_coco_2017_11_17
│   ├── frozen_inference_graph.pb
│   ├── ...
├── faster_rcnn_resnet101_traffic_lights
├── ssd_mobilenet_v1_traffic_lights
```

## 4.2 train the ssd_mobilenet model
```
python object_detection/train.py --logtostderr \
  --pipeline_config_path=config/ssd_mobilenet_v1_traffic_lights.config \
  --train_dir=models/ssd_mobilenet_v1_traffic_lights
```

## 4.3 train the faster_rcnn_resnet101 model
```
python object_detection/train.py --logtostderr \
  --pipeline_config_path=config/faster_rcnn_resnet101_traffic_lights.config \
  --train_dir=models/faster_rcnn_resnet101_traffic_lights
```

## 4.4 export the detection model
```
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/ssd_mobilenet_v1_traffic_lights.config \
    --trained_checkpoint_prefix models/ssd_mobilenet_v1_traffic_lights/model.ckpt-xxx \
    --output_directory models/ssd_mobilenet_v1_traffic_lights/export
    
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/faster_rcnn_resnet101_traffic_lights.config \
    --trained_checkpoint_prefix models/faster_rcnn_resnet101_traffic_lights/model.ckpt-xxx \
    --output_directory models/faster_rcnn_resnet101_traffic_lights/export
```

# 5. evaluate the detection model
```
python object_detection/eval.py --logtostderr \
    --checkpoint_dir=models/faster_rcnn_resnet101_traffic_lights/export \
    --eval_dir=models/faster_rcnn_resnet101_traffic_lights/eval \
    --pipeline_config_path=config/faster_rcnn_resnet101_traffic_lights.config
    
python object_detection/eval.py --logtostderr \
    --checkpoint_dir=models/ssd_mobilenet_v1_traffic_lights/export \
    --eval_dir=models/ssd_mobilenet_v1_traffic_lights/eval \
    --pipeline_config_path=config/ssd_mobilenet_v1_traffic_lights.config
```
run `detection_test.ipynb`

# 6. prepare the classifier data
```
python prepare_classifier_data.py
```

# 7. train and evaluate the classifier
run `train_classifier.ipynb`

# 8. export the classifier model
```
 python freeze_graph.py --input_graph=models/classifier/graph.pb \
    --input_checkpoint=models/classifier/model.ckpt \
    --input_binary=true \
    --output_graph=models/classifier/frozen_graph.pb \
    --output_node_names=prediction
```

# 9. test the tl_classifier
```
 python test_tl_classifier.py input output
```
