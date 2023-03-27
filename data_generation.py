# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from cv2 import cv2_imshow
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

#link for roboflow dataset
roboflow_link = (curl -L "https://app.roboflow.com/ds/KhjLZS7DRn?key=04wQwy2VHP")  

#now the link from roboflow for the annotation files
roboflow_link > roboflow.zip; unzip roboflow.zip; rm roboflow.zip  #chance of a syntax problem

#making coco instances for the training, validation and testing of dataset
def coco_int():
    register_coco_instances("my_dataset_train", {}, "/content/train/_annotations.coco.json", "/content/train")
    register_coco_instances("my_dataset_val", {}, "/content/valid/_annotations.coco.json", "/content/valid")
    register_coco_instances("my_dataset_test", {}, "/content/test/_annotations.coco.json", "/content/test")
    return True
