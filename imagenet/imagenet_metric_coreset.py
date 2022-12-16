import os, sys

sys.path.append('/home/liu/ytx/CSMAL')

from data_utils import cifar
from imagenet.CONFIG import config_imagenet
import torch
from active_utils.Log import get_logger
from active_trainers.MetricLearnerCoresetImagenet import MetricTrainer

import numpy as np

device = torch.device('cuda:1')
logger = get_logger('./imagenet_metric_coreset.log')

def seed_torch(seed=2018):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# method_str = '====================cifar10 Metric==============='
# logger.info(method_str)
# # for i in range(5):
# seed_torch(0)
# # active_trainer = MetricTrainer(config=CONFIG.config, device=device, logger=logger, positive_margin=0.3)
# # active_trainer.start_loop(None, train_dataset, test_dataset, unlabel_dataset, 1000, 10, i)
# active_trainer = MetricTrainer(config=config_cifar10, device=device, logger=logger,positive_margin=0.2)
# active_trainer.start_loop(None, None, None, None, 1000, 10, 0, is_semi=False, pesudo_count=0, is_mixup=False)

method_str = '====================imagenet Metric  3090==============='
logger.info(method_str)
for i in range(3):

    seed_torch(i)
    # active_trainer = MetricTrainer(config=CONFIG.config, device=device, logger=logger, positive_margin=0.3)
    # active_trainer.start_loop(None, train_dataset, test_dataset, unlabel_dataset, 1000, 10, i)
    active_trainer = MetricTrainer(config=config_imagenet, device=device, logger=logger,positive_margin=0.1)
    active_trainer.start_loop(None, None, None, None, 5, i, is_semi=False, pesudo_count=0, is_mixup=False)