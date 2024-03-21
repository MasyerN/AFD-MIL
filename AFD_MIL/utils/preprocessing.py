import torch
import numpy as np
from torchvision.models import resnet101


# /home/omnisky/hdd_15T_sdd/NanTH/IRL_DATA/all_slide/baseline_train_0728.pth
# /home/omnisky/hdd_15T_sdd/NanTH/IRL_DATA/all_slide/baseline_test_0728.pth
# .pth file is a dict including the key ["slides", "targets", "grid"]
# we should extract the visual feature first

def extract(pth_path, model):
    data_info = torch.load(pth_path)
    