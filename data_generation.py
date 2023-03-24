# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import main

# import some common libraries
import numpy as np
import cv2
import random
from cv2 import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

#now the link from roboflow for the annotation files
curl -L ,main.roboflow_link > roboflow.zip; unzip, roboflow.zip; rm ,roboflow.zip  #chance of a syntax problem

#making coco instances for the training, validation and testing of dataset
def coco_int(traind,vald,testd):
    register_coco_instances("my_dataset_train", {}, traind , "/content/train")
    register_coco_instances("my_dataset_val", {}, vald, "/content/valid")
    register_coco_instances("my_dataset_test", {}, testd, "/content/test")
    return Trues