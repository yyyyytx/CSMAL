import os, sys

sys.path.append('/home/liu/ytx/CSMAL')

from data_utils import imagenet64
from imagenet.CONFIG import config_imagenet
import torch
from active_utils.Log import get_logger
from active_trainers.VAALTrainerImagenet import VAALTrainer

import numpy as np


train_dataset = imagenet64.get_train_dataset()
unlabel_dataset = imagenet64.get_unlabel_dataset()
test_dataset = imagenet64.get_test_dataset()


def seed_torch(seed=2018):
    import os, random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


device = torch.device('cuda:0')
logger = get_logger('./imagenet_vaal.log')


for i in range(3):
    seed_torch(i)
    # model = ResNet18(len(train_dataset.classes)).to(device)
    active_trainer = VAALTrainer(config=config_imagenet, device=device,logger=logger)
    active_trainer.start_loop(None, train_dataset, test_dataset, 5, index_loop=i)


