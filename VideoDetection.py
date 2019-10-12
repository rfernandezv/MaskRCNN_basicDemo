# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import sys
import visualize 
import config
import model as modellib
import cv2

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# define the test configuration
class TestConfig(config.Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 80

# define the model
rcnn = modellib.MaskRCNN(mode='inference', model_dir=DEFAULT_LOGS_DIR, config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)

colors = visualize.random_colors(len(class_names))

cap = cv2.VideoCapture(0)
while True:

     _, frame = cap.read()
     predictions = rcnn.detect([frame],
                                verbose=0)  # We are replicating the same image to fill up the batch_size
     p = predictions[0]

     output = visualize.display_instances_video(frame, p['rois'], p['masks'], p['class_ids'],
                                 class_names, p['scores'], colors=colors, real_time=True)
     cv2.imshow("Mask RCNN", output)
     k = cv2.waitKey(5)
     if k & 0xFF == ord('q'):
         break
cap.release()
cv2.destroyAllWindows()