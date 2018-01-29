#from styx_msgs.msg import TrafficLight
from TrafficLight import TrafficLight
import cv2
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.sess = tf.Session()
        self.load_detector()
        self.load_classifier()
        self.detect_threshold = 0.9

        # traffic light state map
        # 'green':0, 'red':1, 'yellow':2, 'off':3
        self.light_state_dict = {0: TrafficLight.GREEN,
                                 1: TrafficLight.RED,
                                 2: TrafficLight.YELLOW,
                                 3: TrafficLight.UNKNOWN}

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        detect_boxes, detect_scores = self.run_detector(image)
        if len(detect_boxes) == 0:
            return TrafficLight.UNKNOWN
 
        # get the max score index
        index = np.argmax(detect_scores)
        light_states = self.run_classifier(image, [detect_boxes[index]])

        return light_states[0]

    def load_detector(self):
        detector_graph_def = tf.GraphDef()
        #with open('models/ssd_mobilenet_v1_traffic_lights/frozen_inference_graph.pb', 'rb') as f:
        #with open('models/faster_rcnn_resnet101_traffic_lights/frozen_inference_graph.pb', 'rb') as f:
        #with open('models/faster_rcnn_inception_v2_traffic_lights/frozen_inference_graph.pb', 'rb') as f:
        with open('models/faster_rcnn_resnet50_traffic_lights/frozen_inference_graph.pb', 'rb') as f:
            serialized = f.read()
            detector_graph_def.ParseFromString(serialized)
        tf.import_graph_def(detector_graph_def, name='detector')
        self.detection_input = self.sess.graph.get_tensor_by_name('detector/image_tensor:0')
        self.detection_boxes = self.sess.graph.get_tensor_by_name('detector/detection_boxes:0')
        self.detection_scores = self.sess.graph.get_tensor_by_name('detector/detection_scores:0')

    def load_classifier(self):
        classifier_graph_def = tf.GraphDef()
        with open('models/classifier/frozen_graph.pb', 'rb') as f:
            serialized = f.read()
            classifier_graph_def.ParseFromString(serialized)
        tf.import_graph_def(classifier_graph_def, name='classifier')
        self.classifier_input = self.sess.graph.get_tensor_by_name('classifier/input_image:0')
        self.classifier_keep_prob = self.sess.graph.get_tensor_by_name('classifier/keep_prob:0')
        self.classifier_prediction = self.sess.graph.get_tensor_by_name('classifier/prediction:0')

    def run_detector(self, image):
        # run the detection net
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores) = self.sess.run(
            [self.detection_boxes, self.detection_scores],
            feed_dict={self.detection_input: image_expanded})
        if (len(boxes) == 0):
            return [], []
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        # get the real box which score is above the detection threshold
        detected = []
        for i in range(len(boxes)):
            if scores[i] >= self.detect_threshold:
                detected.append(i)
        if (len(detected) == 0):
            return [], []
        boxes = boxes[detected]
        scores = scores[detected]        

        # convert the normalized size to pixels
        h = image.shape[0]
        w = image.shape[1]
        for box in boxes:
            box[0] = int(h * box[0]) # ymin
            box[1] = int(w * box[1]) # xmin
            box[2] = int(h * box[2]) # ymax
            box[3] = int(w * box[3]) # xmax
        
        # check box size
        valid = []
        for i in range(len(boxes)):
            box = boxes[i]
            if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
                continue
            elif  box[3] - box[1] < 2 or box[2] - box[0] < 2:
                continue
            else:
                valid.append(i)
        return boxes[valid], scores[valid]
      
    def run_classifier(self, image, detect_boxes):
        num_lights = len(detect_boxes)
        if num_lights == 0:
            raise Exception('The number of detected boxes is zero.')
        elif num_lights > 9:
            num_lights = 9

        image_light_batch = np.zeros((num_lights, 32, 32, 3), dtype=np.int32)
        for i in range(num_lights):
            ymin, xmin, ymax, xmax =  detect_boxes[i].astype(np.int32)
            image_light = image[ymin:ymax, xmin:xmax]
            image_light_batch[i, :, :, :] = cv2.resize(image_light, (32, 32))

        light_states_index = self.sess.run(self.classifier_prediction,
                                           feed_dict={self.classifier_input: image_light_batch,
                                                      self.classifier_keep_prob: 1.0})

        light_states = map(lambda i: self.light_state_dict[i], light_states_index)

        return list(light_states)
