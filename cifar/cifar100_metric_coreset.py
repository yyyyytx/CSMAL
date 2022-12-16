import os, sys

sys.path.append('/home/liu/ytx/CSMAL')

from data_utils import cifar
from cifar.CONFIG import config_cifar100
import torch
from active_utils.Log import get_logger
from active_trainers.MetricLearnerCoreset import MetricTrainer

import numpy as np

device = torch.device('cuda:0')
logger = get_logger('./cifar100_metric.log')

def seed_torch(seed=2018):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


method_str = '====================cifar100 Metric==============='
logger.info(method_str)
for i in range(3):

    seed_torch(i)
    active_trainer = MetricTrainer(config=config_cifar100, device=device, logger=logger)
    active_trainer.start_loop(None, None, None, None, 2000, 10, i, is_semi=False, pesudo_count=0, is_mixup=False)
