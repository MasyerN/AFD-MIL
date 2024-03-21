from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import numpy as np


class MixABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader, True)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            input = feature.cuda()
            targets = target.cuda()
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)      
            j_0 = torch.softmax(_, 1)
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            if epoch <= self.cfgs["asynchronous"]:
                loss = loss2
            else:
                loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        for i, (feature, target, slide_id) in enumerate(loader):
            input = feature.cuda()
            with torch.no_grad():
                output, atten_target, atten_score = self.model.encoder.inference(input)
            probs.extend(torch.cat((output, atten_score), 1).detach().cpu().numpy())
        return np.array(probs)

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output = self.model.inference(input)
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
        return probs, targets

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader, False)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, pred)
            print(epoch, 'Validation:', score)
            torch.cuda.empty_cache()
        
