from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
 
#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.
class CocoTrainer(DefaultTrainer):

  @classmethod          #its a decorator
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)
  
getconfig= "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
getcheckpoint="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(getconfig))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(getcheckpoint)  # Let training initialize from model zoo
# cfg.MODEL.WEIGHTS="/content/drive/MyDrive/model_final_68b088.pkl"
cfg.SOLVER.IMS_PER_BATCH = 5
cfg.SOLVER.BASE_LR = 0.02


cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.MAX_ITER = 101 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (100, 150)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 50
  


