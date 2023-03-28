from train import trainerr
import config
from data_generation import coco_int,robolink
from utility import visual, evaluation
import utility
import inference
import cv2
from google.colab.patches import cv2_imshow
from detectron2.engine import DefaultPredictor

if __name__ == "__main__":

    #link for roboflow dataset
    #roboflow_link = "https://app.roboflow.com/ds/KhjLZS7DRn?key=04wQwy2VHP"  
    robolink()

    coco_int()
    print("Visualization of training data:")

    visual()

    #now the output directory and training will be done

    trainer=trainerr()

    #test evaluation
    evaluation(trainer)
    #inference with detectron2
    inference.inference()

     #now the test result
    utility.testresult()

    #train visualization
    utility.trainvilualize()
