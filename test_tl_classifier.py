from tl_classifier import TLClassifier
#from styx_msgs.msg import TrafficLight
from TrafficLight import TrafficLight
import cv2
import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm

light_labels = {
    TrafficLight.UNKNOWN:'off', 
    TrafficLight.RED: 'red',
    TrafficLight.GREEN:'green', 
    TrafficLight.YELLOW:'yellow', 
    }

light_colors = {
    TrafficLight.UNKNOWN: (70,  70,  70),
    TrafficLight.GREEN: ( 0, 255,  0),
    TrafficLight.YELLOW: (0, 225, 255),
    TrafficLight.RED: ( 0,  0, 255)}

def draw_box(image, box, color, text):
    ymin, xmin, ymax, xmax = box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.rectangle(image, (xmin - 1, ymin), (xmax + 1, ymin - 20), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (xmin + 5, ymin - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def print_usage():
    print("Usage: test_classifier.py in_dir out_dir")

def main():
    # check usage
    if len(sys.argv) != 3:
        print_usage()
        return

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]

    # get test images
    files = glob(in_dir + "/*.png")
    
    # perform detection & classification on each image, write the result image to the out dir
    tl_classifier = TLClassifier()
    tl_classifier.detect_threshold = 0.1
    for in_file in tqdm(files):
        out_file = os.path.join(out_dir, os.path.basename(in_file))
        image = cv2.cvtColor(cv2.imread(in_file), cv2.COLOR_BGR2RGB)
        detect_boxes, detect_scores = tl_classifier.run_detector(image)
        #print("{0} lights detect".format(len(detect_boxes)))
        if len(detect_boxes) == 0:
            cv2.imwrite(out_file, image)
            continue
        light_states = tl_classifier.run_classifier(image, detect_boxes)
        for box, state in zip(detect_boxes, light_states):
            color = light_colors[state]
            label = light_labels[state]
            draw_box(image, box.astype(np.int32), color, label)
        cv2.imwrite(out_file, image)

if __name__ == '__main__':
    sys.exit(main())
