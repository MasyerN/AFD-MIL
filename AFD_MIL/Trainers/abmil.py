from .base import BaseTrainer
import numpy as np
import torch
from torch.nn import functional as F
import time
import config
from config import parser as reweight_parser
stable_cfg = reweight_parser.parse_args()
#import reweighting.weight_learner2 as weight_learner
import reweighting
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix
def draw_confusion_matrix(true_labels, predicted_labels, name):
    fontsize = 14

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # 设置类别名称
    class_names = ['N', 'B', 'M', 'SCC', 'SK']

    # 绘制混淆矩阵图
    plt.figure(figsize=(4, 3))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=fontsize)
    plt.yticks(tick_marks, class_names, fontsize=fontsize)

    # 添加标签
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(conf_matrix[i, j]), fontsize=11, horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

    #plt.ylabel('Target')
    #plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()
    plt.savefig('/home/omnisky/sde/NanTH/result/confusion_matrix/' + name + '.png')


class Stable_ABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            pre_features = self.model.pre_features
            pre_weight1 = self.model.pre_weight1
            weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            self.model.pre_features.data.copy_(pre_features)
            self.model.pre_weight1.data.copy_(pre_weight1)
            output, _ = self.model(input, weight1)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True) 
            loss1 = self.criterion(output, targets)
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))
            
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            if epoch >= self.cfgs["asynchronous"]:
                loss = loss1 + loss2
            else:
                loss = loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model.encoder(input.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        preds = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                pre_features = self.model.pre_features.detach().clone()
                pre_weight1 = self.model.pre_weight1.detach().clone()
                weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                y = self.model.inference(input, weight1)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                prob = y.softmax(dim=-1)
                pred = prob.argmax(dim=-1)
                probs.append(prob[0][1].detach().cpu().numpy())
                targets.append(target.numpy())
                preds.append(pred.detach().cpu().numpy())
                del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return probs, preds, targets

    def train(self):
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            probs, preds, target = self.inference(self.test_loader, epoch)
            score = self.metric_ftns(target, preds, probs)
            info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(score)
            self._check_best(epoch, score)
            self.logger(info)
            torch.cuda.empty_cache()
        score = self.best_metric_info
        info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        self.logger(info)

class Origin_ABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        #pred = self._inference_for_selection(self.bag_loader)
        #self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        #self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        #self.model.encoder.eval()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)

            loss1 = self.criterion(output, targets)
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            loss = loss1# + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model.encoder(input.reshape([-1, 1024]))  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        pred = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                output, result = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy()[0])
                pred.append(result.detach().cpu().numpy())
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return np.array(probs), np.array(pred), np.array(targets)

    def train(self):
        save_log = []
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            #pred = self._inference_for_selection(self.test_loader)
            #self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            #self.test_loader.dataset.set_mode('selected_bag')

            probs, pred, target = self.inference(self.test_loader, epoch)
            save_log.append({'t': target, 'p': pred})
            draw_confusion_matrix(target, pred, 'ABMIL_L_30_' + str(epoch))
            score = self.metric_ftns(target, pred, probs, 'macro')
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(epoch, 'Validation:', score)
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        torch.save(save_log, r'/home/omnisky/sde/NanTH/result/confusion_matrix/abmil_l_30.pth')
        #score = self.best_metric_info
        #print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)
 

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
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #maxpooling_out, _ = self.model.encoder(input.detach().clone()[0])
            #selected_num = torch.argsort(torch.softmax(maxpooling_out, 1)[:, 1], descending=True)
            #loss2 = self.criterion(torch.index_select(maxpooling_out, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)
            selected_num = torch.argsort(torch.softmax(_, 1)[:, 1], descending=True)      
            j_0 = torch.softmax(_, 1)
            loss1 = self.criterion(output, targets)
            #fans_loss1 = 1.0 / torch.exp(torch.tensor(loss1.clone().detach().cpu().tolist()).cuda())
            loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num[: 2]), targets.repeat(2))# * fans_loss1
            #losss3 = self.criterion(torch.index_select(_, dim=0, index=selected_num[-1 :]), targets.repeat(1)) * fans_loss1
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            if epoch <= self.cfgs["asynchronous"]:
                loss = loss2
            else:
                loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader, if_train):
        self.model.eval()
        probs = []
        for i, (feature, target, slide_id) in enumerate(loader):
            input = feature.cuda()
            '''
            if if_train:
                self.model.train()
                output, atten_target, atten_score = self.model.encoder.inference(input)  # [B, num_classes]            
                loss = self.criterion(atten_target, target.cuda())
                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()
            else:
            '''
            with torch.no_grad():
                output = self.model.encoder.inference(input)
                #self.model.eval()
                #output, atten_target, atten_score = self.model.encoder.inference(input)
            #probs.extend(torch.cat((output, atten_score), 1).detach().cpu().numpy())
            probs.extend(output.detach().cpu().numpy())
        return np.array(probs)

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                output = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                probs.append(output.detach().cpu().numpy())
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
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
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(epoch, 'Validation:', score)
            #model_name = '/home/omnisky/sde/NanTH/result/mix_abmil/camelyon/' + str(epoch) + '_' + str(score['acc'])[: 4] + '.pth'
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        
        #score = self.best_metric_info
        #print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)
        
        
class IRL_ABMIL(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, train_loader, test_loader, bag_loader, cfgs):
        super().__init__(model, criterion, metric_ftns, optimizer, cfgs)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bag_loader = bag_loader

    def _train_epoch(self, epoch):
        self.bag_loader.dataset.set_mode('bag')
        pred = self._inference_for_selection(self.bag_loader)
        self.train_loader.dataset.top_k_select(pred, is_in_bag=True)
        self.train_loader.dataset.set_mode('selected_bag')
        loss = self._train_iter(epoch)
        
        # logger
        print(f'Training\tEpoch: [{epoch+1}/{self.cfgs["epochs"]}]\tLoss: {loss}')
    
    def _train_iter(self, epoch):
        self.model.train()
        self.model.encoder.train()
        running_loss = 0.
        # default batch size is 1, but fail to train. So I random select the patches in the bag level.
        # features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
        # targets = torch.zeros([self.cfgs["batch_size"]]).long()
        for i, (feature, target, slide_id) in enumerate(self.train_loader):
            # [1*k, N] -> [B*K, N]
            # if (i+1) % self.cfgs["batch_size"] == 0 or (i+1) == len(self.bag_loader):
            input = feature.cuda()
            targets = target.cuda()
            #maxpooling_out, _ = self.model.encoder(input.detach().clone()[0])
            #selected_num = torch.argsort(torch.softmax(maxpooling_out, 1)[:, 1], descending=True)
            #loss2 = self.criterion(torch.index_select(maxpooling_out, dim=0, index=selected_num[: 2]), targets.repeat(2))
            #pre_features = self.model.pre_features
            #pre_weight1 = self.model.pre_weight1
            #weight1, pre_features, pre_weight1 = reweighting.weight_learner(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
            #self.model.pre_features.data.copy_(pre_features)
            #self.model.pre_weight1.data.copy_(pre_weight1)
            output, _= self.model(input)
            selected_num0 = torch.argsort(torch.softmax(_, 1)[:, 0], descending = True)
            selected_num1 = torch.argsort(torch.softmax(_, 1)[:, 1], descending = True)
            selected_num2 = torch.argsort(torch.softmax(_, 1)[:, 2], descending = True)
            selected_num3 = torch.argsort(torch.softmax(_, 1)[:, 3], descending = True)
            selected_num4 = torch.argsort(torch.softmax(_, 1)[:, 4], descending = True)
            if targets == 0:
                loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num0[: 2]), targets.repeat(2))       
            elif targets == 1:
                loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num1[: 2]), targets.repeat(2))
            elif targets == 2:
                loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num2[: 2]), targets.repeat(2))

            elif targets == 3:
                loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num3[: 2]), targets.repeat(2))
            elif targets == 4:
                loss2 = self.criterion(torch.index_select(_, dim=0, index=selected_num4[: 2]), targets.repeat(2))

            else:
                print('error')
            loss1 = self.criterion(output, targets)
            
            #t = targets.unsqueeze(0)
            #t = targets.repeat(_.size()[1])
            #loss2 = self.criterion(_[0], targets.repeat(_.size()[1]))
            if epoch >= self.cfgs["asynchronous"]:
                loss = loss1 + 0.01 * loss2
            else:
                loss = loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()*input.size(0)
            #print(f"\rTraining\t{(i+1)/len(self.train_loader)*100:.2f}%\tloss: {loss.item():.5f}", end='', flush=True)
            features = torch.zeros([self.cfgs["batch_size"], self.cfgs["sample_size"], 1024])
            targets = torch.zeros([self.cfgs["batch_size"]]).long()
            # else:
            #     length = feature.size(1)
            #     _index = np.arange(0, length)
            #     np.random.shuffle(_index)
            #     if length > self.cfgs["sample_size"]:
            #         features[i % self.cfgs["batch_size"]] = feature[0, _index[:self.cfgs["sample_size"]]]
            #     else:
            #         features[i % self.cfgs["batch_size"], _index[:length]] = feature[0, _index[:length]]
            #     targets[i % self.cfgs["batch_size"]] = target
        print('')
        return running_loss/len(self.train_loader.dataset)

    def _inference_for_selection(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                output, _ = self.model.encoder(input[0])  # [B, num_classes]
                probs.extend(output.detach().cpu().numpy())
                #print(f'\rinference progress: {(i+1)/len(loader)*100:.1f}%', end='', flush=True)
            print("")
            probs = np.array(probs).reshape([-1, self.cfgs["num_classes"]])
        return probs 

    def inference(self, loader, epoch = 0):
        self.model.eval()
        probs = []
        pred = []
        targets = []
        with torch.no_grad():
            for i, (feature, target, slide_id) in enumerate(loader):
                input = feature.cuda()
                #pre_features = self.model.pre_features.detach().clone()
                #pre_weight1 = self.model.pre_weight1.detach().clone()
                #weight1 = reweighting.weight_learner2(input[0], pre_features, pre_weight1, stable_cfg, epoch, i)
                output, prob = self.model.inference(input)#torch.tensor([1.0 for x in input.size()[0]]).cuda())
                pred.append(output.detach().cpu().numpy())
                probs.append(prob.detach().cpu().numpy()[0])
                targets.append(target.numpy())
                #del weight1
                # print(f'inference progress: {i+1}/{len(loader)}')
        return np.array(pred), np.array(probs), np.array(targets)

    def train(self):
        save_log = []
        for epoch in range(self.cfgs["epochs"]): 
            self._train_epoch(epoch)
            # Validation
            self.test_loader.dataset.set_mode('bag')
            pred = self._inference_for_selection(self.test_loader)
            self.test_loader.dataset.top_k_select(pred, is_in_bag=True)
            self.test_loader.dataset.set_mode('selected_bag')

            pred, probs, target = self.inference(self.test_loader, epoch)
            save_log.append({'t': target, 'p': pred})
            draw_confusion_matrix(target, pred, 'PEAK-ABMIL_L_Epr_' + str(epoch))
            score = self.metric_ftns(target, pred, probs, 'macro')
            #info = f'Epoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n'
            #print(f'Validation\tEpoch: [{epoch + 1}/{self.cfgs["epochs"]}]\tf1: {score["f1"]}\tprecision: {score["precision"]}\trecall: {score["recall"]}\tACC: {score["acc"]}')
            print(epoch, 'Validation:', score)
            model_name = '/home/omnisky/sde/NanTH/result/mix_abmil/' + str(epoch) + '_' + str(score['acc'])[: 4] + '.pth'
            if score['acc'] > 0.99:
                torch.save(self.model, model_name)
            #self._check_best(epoch, score)
            #self.logger(info)
            torch.cuda.empty_cache()
        torch.save(save_log, r'/home/omnisky/sde/NanTH/result/confusion_matrix/peak-abmil_l_epr_.pth')
        #score = self.best_metric_info
        #print(score)
        #info = 20*'#' + f'\nf1: {score["f1"]}, precision: {score["precision"]}, recall: {score["recall"]}, acc: {score["acc"]}\n' + 20*'#'
        #self.logger(info)