import config
import os
from config import cfg

def trainerr():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = config.CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    trainer.train()
    return True
    

#here it starts the training
