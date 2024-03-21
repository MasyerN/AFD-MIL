import torch
from torch.utils import data
import pandas as pd
import os
from os import path as osp
import numpy as np
from torch.nn import functional as F


class MILdataset(data.Dataset):
    def __init__(self, cfgs, split='train'):
        self.cfgs = cfgs
        self.k = cfgs["k"]

        if split == 'train':
            self.bag_info = pd.read_csv(osp.join(self.cfgs["train_data"], 'training_30.csv'))
            self.bag_list = osp.join(self.cfgs["train_data"], "pt_files")
        else:
            self.bag_info = pd.read_csv(osp.join(self.cfgs["test_data"], 'testing.csv'))   # all bag
            self.bag_list = osp.join(self.cfgs["test_data"], "pt_files")
        '''
        if isinstance(data_num, int) == 1:
            sampled_df = self.bag_info.groupby('label').apply(lambda x: x.sample(data_num, replace=True)).reset_index(drop=True)
            shuffled_df = sampled_df.sample(frac=1).reset_index(drop=True)
            self.bag_info = shuffled_df
            #print(self.bag_info[1730])
        '''
        self._bag_or_instance = 'bag'
        self.instance_info = {}               # all instance
        self.selected_instance_info = {}      # selected instance for training
        self.selected_bag_info = {}           # selected bag for training
        self.data_info = None
        self.targets = []
        self.df2dict()
        self.make_instance_set()
        self.set_mode()
    
    def df2dict(self):
        bag_info = {}
        for index in range(len(self.bag_info)):
            slide_id = self.bag_info["slide_id"].iloc[index]
            target = self.bag_info["label"].iloc[index]
            try:
                feature = torch.load(osp.join(self.bag_list, f"{slide_id.split('/')[-1]}.pt"))
                bag_info[len(bag_info)] = {"slide_id": slide_id, "label": target, "num_instance": feature.size(0)}
                self.targets.append(target)
            except:
                a = 0
        self.bag_info = bag_info

    def make_instance_set(self):
        self.instance_info = {}
        count = 0
        print(len(self.bag_info))
        for index in range(len(self.bag_info)):
            slide_id = self.bag_info[index]["slide_id"]
            target = self.bag_info[index]["label"]
            num_instance = self.bag_info[index]["num_instance"]
            for instance_index in range(num_instance):
                self.instance_info[count] = {"slide_id": slide_id, "label": target, "instance_index": instance_index}
                count += 1
            #print(f'\r{(index+1)/len(self.bag_info)*100:.2f}%', end='', flush=True)
        print(f"\t{count} tiles is collected")
    
    def set_mode(self, mode='bag'):
        self._bag_or_instance = mode
        if mode == 'bag':
            self.data_info = self.bag_info
        elif mode == 'instance':
            self.data_info = self.instance_info
        elif mode == 'selected_bag':
            self.data_info = self.selected_bag_info
        elif mode == 'selected_instance':
            self.data_info = self.selected_instance_info
        else:
            raise ValueError("wrong mode")
    

    def top_k_select(self, pred, is_in_bag=False):
        count = 0
        _count = 0
        return_result = [[] for j in range(len(self.bag_info))]
        # shuffle data
        # instance
        index_selected_instance = np.arange(0, len(self.bag_info)*self.k)
        np.random.shuffle(index_selected_instance)
        # bag
        index_selected_bag = np.arange(0, len(self.bag_info))
        np.random.shuffle(index_selected_bag)
        k = self.k
        # selection
        for index in range(len(self.bag_info)):
            slide_id = self.bag_info[index]["slide_id"]
            target = self.bag_info[index]["label"]
            num_instance = self.bag_info[index]["num_instance"]
            _pred = pred[count: count+num_instance, 1]
            _index = np.argsort(_pred).tolist()  # min, ..., max
            
            #_atten = pred[count: count+num_instance, 2]
            #_score = np.argsort(_atten).tolist()            
            return_result[index].append(slide_id)
            return_result[index].append(target)
            return_result[index].append(_index[-k:])# + _score[-k:])# + _index[: k])
            if is_in_bag:
                # k = self.k if self.k < _index.shape[0] else _index.shape[0]
                self.selected_bag_info[index_selected_bag[_count]] = {"slide_id": slide_id, "label": target, "instance_index": _index[-k:]}# }
                _count += 1
            else:
                for ii in range(k):
                    self.selected_instance_info[index_selected_instance[_count]] = {"slide_id": slide_id, "label": target, "instance_index": _index[-ii]}
                    _count += 1
            count += num_instance
        return return_result

    
    def __getitem__(self, index):
        slide_id = self.data_info[index]["slide_id"]
        target = self.data_info[index]["label"]
        feature = torch.load(osp.join(self.bag_list, f"{slide_id.split('/')[-1]}.pt"))

        if self._bag_or_instance == 'bag':
            return feature, target, slide_id
        
        elif self._bag_or_instance == 'selected_bag':
            _index = self.data_info[index]["instance_index"]
            _feature = torch.zeros([self.k, self.cfgs['feature_dim']])
            for i in range(len(_index)):
                _feature[i] = feature[_index[i]]
            feature = _feature
            return feature, target, slide_id#, _index
        else:
            feature = feature[self.data_info[index]["instance_index"]]
            return feature, target, slide_id
    
    def __len__(self):
        return len(self.data_info)
