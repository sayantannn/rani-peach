from train import trainerr
import config
from data_generation import coco_int
from utility import visual
import utility
import inference

if __name__ == "__main__":

    #link for roboflow dataset
    roboflow_link = "https://app.roboflow.com/ds/KhjLZS7DRn?key=04wQwy2VHP"  

    #traindm = "/content/train/_annotations.coco.json"
    #valdm = "/content/valid/_annotations.coco.json"
    #testdm = "/content/test/_annotations.coco.json"

    coco_int(traindm, valdm, testdm)
    print("Visualization of training data:")

    utility.visual()

    #now the output directory and training will be done

    trainerr()

    #test evaluation
    utility.evaluation()
    #inference with detectron2
    inference.inference()

     #now the test result
    utility.testresult()

    #train visualization
    utility.trainvilualize()
