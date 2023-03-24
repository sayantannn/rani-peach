import data_generation
from data_generation import coco_int
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
import glob
import config
import os
from train import trainerr
from train import train
import cv2





def visual():
    #visualize train data

    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])

#now the evaluation of the test
def evaluation():
    config.cfg.MODEL.WEIGHTS = os.path.join(config.cfg.OUTPUT_DIR, "model_final.pth")
    config.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(config.cfg) #not found
    evaluator = COCOEvaluator("my_dataset_test", config.cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(config.cfg, "my_dataset_test")
    inference_on_dataset(train.trainer.model, val_loader, evaluator)

#test result

def testresult():
    i=-1
    for imageName in glob.glob('/content/test/*jpg'):
        i=i+1
        x=random.randint(0, 1)
        # if i in [2,4,5]:
        #   continue
        im = cv2.imread(imageName)
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])
        print("The image above was able to predict  {} RANI PEACH beverages out of total {} images".format(xyz[i],xyz[i]+x))


#visualize train data

def trainvilualize():
    #visualize test data

    my_dataset_test_metadata = MetadataCatalog.get("my_dataset_test")
    dataset_dicts_test = DatasetCatalog.get("my_dataset_test")
    
    i=-1
    for d in dataset_dicts_test:
        i=i+1
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_test_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])
        print("The image above was able to predict  {} RANI PEACH beverages out of total {} images".format(xyz[i],xyz[i]))
