# traffic_light_detection
Implementation of a traffic light detector for the udacity sdc nanodegreee using tensorflow object detection API 

# download the Bosch dataset
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

# install the tensorflow object detection API
You can find the tensorflow object detection API [here](https://github.com/tensorflow/models/tree/master/research/object_detection), 
follow the installation instruction [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). 
