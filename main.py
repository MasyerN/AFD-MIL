import os
import torch
from torch import nn
from Models import model_fn
from Trainers import trainers_fn
from utils.cfgs_loader import load_yaml
from utils.metrics import compute_score_2class, compute_score
from utils.dataset import MILdataset
import numpy as np
import random
import warnings
#nohup /home/omnisky/anaconda3/envs/NanTH/bin/python -u /home/omnisky/hdd_15T_sdc/NanTH/Baseline/main.py > /home/omnisky/sde/NanTH/result/confusion_matrix/peak-abmil-fj.log 2>&1 &
warnings.filterwarnings('ignore')
CRITERION = {'ce': nn.CrossEntropyLoss, 'nll': nn.NLLLoss}
 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfgs = load_yaml("./config/ABMIL.yaml")
os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['cuda']
 
if cfgs["seed"] == -1:
    cfgs["seed"] = np.random.randint(0, 23333)
print(cfgs)
setup_seed(cfgs["seed"])

def main():
    model = model_fn[cfgs['task']](cfgs)
    criterion = CRITERION[cfgs['loss']]()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfgs["lr"], weight_decay=cfgs['weight_decay'])
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4, nesterov=True)
    # load data
    train_dset = MILdataset(cfgs, 'train')
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cfgs["batch_size"], shuffle=False)
    # this loader used for pseudo-label generation, due to the low speed of generation via instance loader with big batch size
    bag_dset = MILdataset(cfgs, 'train')
    bag_loader = torch.utils.data.DataLoader(bag_dset, batch_size=1, shuffle=False)
    test_dset = MILdataset(cfgs, 'test')
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False)
 
    trainer = trainers_fn[cfgs["task"]](model, criterion, compute_score, optimizer, train_loader, test_loader, bag_loader, cfgs)
    trainer.train()


if __name__ == '__main__':
    main()
