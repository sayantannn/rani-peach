import config
from config import cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import os

#inference with detectron2
def inference():
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    xyz=[9,2,11,6,8,9,4,2,4,1]
    test_metadata = MetadataCatalog.get("my_dataset_test")
