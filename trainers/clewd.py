import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.optim as optim
import datasets
import models
import utils
import losses
import cvxpy as cp
from pathlib import Path
import os.path as osp
import torchvision.models as torch_models
import torch.nn as nn
import matplotlib
import ot
from thop import profile
from thop import clever_format
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0 ]
# print('111111111111111111111111111111111111111111111111111111111')


class RobustAdversarialTrainer(object):
    def __init__(self, config):
        self.config = config
        # self.device = 'cuda:0'

        # Create dataloader
        source_loader, target_loader, nclasses = datasets.form_visda_datasets(config=config, ignore_anomaly=True)




        ##### tgrs补实验做nwpu之前用的
        # source_loader, target_loader, nclasses, sourcey_loader = datasets.form_visda_datasets(config=config,  ignore_anomaly=True)





        # ###
        # source_loader, target_loader, nclasses, test_loader = datasets.form_visda_datasets(config=config, ignore_anomaly=True)
        # ###

        self.source_loader = source_loader
        self.target_loader = target_loader
        self.nclasses = nclasses

        ##### tgrs补实验做nwpu之前用的
        # self.sourcey_loader = sourcey_loader

        ##
        # self.test_loader=test_loader
        # ##

        # Create model
        self.netF, self.nemb = models.form_models(config)
        # print(self.netF)
        self.netC = models.Classifier(self.nemb, self.nclasses, nlayers=1)
        utils.weights_init(self.netC)
        # print(self.netC)
        # self.netD = models.Classifier(self.nemb, 1, nlayers=3, use_spectral=True)
        # utils.weights_init(self.netD)
        # print(self.netD)

        # self.netF = self.netF.to(self.device)
        self.netF = self.netF.cuda()

        # self.netC = self.netC.to(self.device)
        self.netC = self.netC.cuda()

        # self.netD = self.netD.to(self.device)
        # self.netD = self.netD.cuda()

        # self.netF = torch.nn.DataParallel(self.netF, device_ids).cuda()
        # self.netC = torch.nn.DataParallel(self.netC, device_ids).cuda()

        self.maskrho = nn.Parameter(torch.Tensor(np.ones((1, 1))), requires_grad=True)
        # self.maskrho = torch.nn.DataParallel(self.maskrho, device_ids).cuda()

        # self.netD = torch.nn.DataParallel(self.netD,device_ids ).cuda()

        # Create optimizer
        self.optimizerF = optim.SGD(self.netF.parameters(), lr=self.config.lr, momentum=config.momentum,
                                    weight_decay=0.0005)
        self.optimizerC = optim.SGD(self.netC.parameters(), lr=self.config.lrC, momentum=config.momentum,
                                    weight_decay=0.0005)
        # self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.config.lrD, betas=(0.9, 0.999))

        # self.optimizermask = optim.SGD(self.maskrho, lr=self.config.lrC, momentum=config.momentum, weight_decay=0.0005)

        self.lr_scheduler_F = utils.InvLR(self.optimizerF, gamma=0.0001, power=0.75)
        self.lr_scheduler_C = utils.InvLR(self.optimizerC, gamma=0.0001, power=0.75)
        # self.lr_scheduler_mask = utils.InvLR(self.maskrho, gamma=0.0001, power=0.75)

        # creating losses
        self.loss_fn = losses.loss_factory[config.loss]

        if self.config.weight_update_type == 'discrete':
            self.num_datapoints = len(self.target_loader.dataset)
            # self.weight_vector = torch.FloatTensor(self.num_datapoints, ).fill_(1).to(self.device)
            self.weight_vector = torch.FloatTensor(self.num_datapoints, ).fill_(1).cuda()
        else:
            self.netW = torch_models.resnet18(pretrained=True)
            self.netW.fc = nn.Linear(512, 1)
            # self.netW = self.netW.to(self.device)
            self.netW = self.netW.cuda()
            self.netW = torch.nn.DataParallel(self.netW, device_ids).cuda()
            self.optimizerW = optim.Adam(self.netW.parameters(), lr=self.config.lrD, betas=(0.9, 0.999))
            print(self.netW)

        self.weight_update_type = self.config.weight_update_type
        assert self.weight_update_type in ['cont', 'discrete']
        self.weight_thresh_list = [0, 0, 0]
        self.eps = 0.0001

        self.best_acc = 0
        self.entropy_criterion = losses.EntropyLoss()

        self.otfusion = nn.Parameter(torch.Tensor(np.ones((1, 2))), requires_grad=True)
        # self.maskrho  = nn.Parameter(torch.Tensor(np.ones((1, 1))),requires_grad = True)

        # restoring checkpoint

        '''
        print('Restoring checkpoint ...')
        try:
            ckpt_path = os.path.join(config.logdir, 'model_state.pth')
            self.restore_state(ckpt_path)
        except:
            # If loading failed, begin from scratch
            print('Checkpoint not found. Training from scratch ...')
            self.itr = 0
            self.epoch = 0

        '''

        self.itr = 0
        self.epoch = 0

    def save_state(self):
        model_state = {}
        model_state['epoch'] = self.epoch
        model_state['itr'] = self.itr
        self.netF.eval()
        # self.netD.eval()
        self.netC.eval()
        # model_state['netD'] = self.netD.state_dict()
        model_state['netF'] = self.netF.state_dict()
        model_state['netC'] = self.netF.state_dict()
        # model_state['optimizerD'] = self.optimizerD.state_dict()
        model_state['optimizerF'] = self.optimizerF.state_dict()
        model_state['optimizerC'] = self.optimizerC.state_dict()
        model_state['best_acc'] = self.best_acc

        if self.weight_update_type == 'discrete':
            model_state['weight_vector'] = self.weight_vector.cpu()
        else:
            model_state['netW'] = self.netW.state_dict()
            model_state['optimizerW'] = self.optimizerW.state_dict()
        torch.save(model_state, osp.join(self.config.logdir, 'model_state.pth'))

    def restore_state(self, pth):
        print('Restoring state ...')
        model_state = torch.load(pth)
        self.epoch = model_state['epoch']
        self.itr = model_state['itr']
        self.best_acc = model_state['best_acc']
        # self.netD.load_state_dict(model_state['netD'])
        self.netF.load_state_dict(model_state['netF'])
        self.netC.load_state_dict(model_state['netC'])
        # self.optimizerD.load_state_dict(model_state['optimizerD'])
        self.optimizerF.load_state_dict(model_state['optimizerF'])
        self.optimizerC.load_state_dict(model_state['optimizerC'])
        if self.weight_update_type == 'discrete':
            self.weight_vector = model_state['weight_vector']
            # self.weight_vector = self.weight_vector.to(self.device)
            self.weight_vector = self.weight_vector.cuda()

        else:
            self.netW.load_state_dict(model_state['netW'])
            self.optimizerW.load_state_dict(model_state['optimizerW'])

    def zero_grad_all(self):
        self.netF.zero_grad()
        self.netC.zero_grad()
        # self.netD.zero_grad()
        if self.weight_update_type == 'cont':
            self.netW.zero_grad()
            self.optimizerW.zero_grad()
        self.optimizerF.zero_grad()
        self.optimizerC.zero_grad()
        # self.optimizerD.zero_grad()

    def log(self, message):
        print(message)
        message = message + '\n'
        f = open("{}/log.txt".format(self.config.logdir), "a+")
        f.write(message)
        f.close()

    

    def test(self):
        self.netF.eval()
        self.netC.eval()

        correct = 0
        size = 0
        num_class = self.nclasses
        output_all = np.zeros((0, num_class))
        confusion_matrix = torch.zeros(num_class, num_class)

        with torch.no_grad():
            for batch_idx, data_t in enumerate(self.target_loader):
                imgs, labels, _ = data_t
                imgs = imgs.cuda()
                labels = labels.cuda()

                feat = self.netF(imgs)
                logits = self.netC(feat)
                output_all = np.r_[output_all, logits.data.cpu().numpy()]
                size += imgs.size(0)
                pred = logits.data.max(1)[1]  # get the index of the max log-probability
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                correct += pred.eq(labels.data).cpu().sum()

        print('\nTest set: Accuracy: {}/{} C ({:.0f}%)\n'.format(correct, size, 100. * float(correct) / size))
        mean_class_acc = torch.diagonal(confusion_matrix) / torch.sum(confusion_matrix, dim=1)
        mean_class_acc = mean_class_acc * 100.0

        print('Classwise accuracy')
        print(mean_class_acc)

        mean_class_acc = torch.mean(mean_class_acc)
        net_class_acc = 100. * float(correct) / size

        if net_class_acc > self.best_acc:
            print('confuse:', confusion_matrix)



        return mean_class_acc, net_class_acc

    

    def train(self):
        print('Start training from iter {}'.format(self.itr))
        end_flag = 0

        while True:
            self.epoch += 1
            if end_flag == 1:
                break

            if self.weight_update_type == 'discrete':
                print('Running discrete')
                print('epoch:', self.epoch)
                # print('555555555555555555')



            for i, (data_s, data_t) in enumerate(zip(self.source_loader, self.target_loader)):

            ##### tgrs补实验做nwpu之前用的
            # for i, (data_s, data_t) in enumerate(zip(self.source_loader, self.sourcey_loader)):
                self.itr += 1

                if self.itr > self.config.num_iters:
                    print('Training complete')
                    end_flag = 1
                    break



                timetrain1 = time.time()
                # print('timetrain1:', timetrain1)


                self.netF.train()
                self.netC.train()
                # self.netD.train()
                # if self.weight_update_type == 'cont':
                #     self.netW.train()

                inp_s, lab_s, indices_src = data_s

                # print('inp_s:', inp_s.mean())
                # print('lab_s:', lab_s)

                # print('shape:', indices_src)

                # inp_s, lab_s = inp_s.to(self.device), lab_s.to(self.device)
                inp_s, lab_s = inp_s.cuda(), lab_s.cuda()

                inp_t, lab_t, indices_tgt = data_t
                # inp_t = inp_t.to(self.device)
                inp_t = inp_t.cuda()

                # lab_t = lab_t.cuda()

                self.zero_grad_all()

                feat_s = self.netF(inp_s, dom_id=0)
                feat_t = self.netF(inp_t, dom_id=1)

                # print('feat_s:',feat_s.mean())

                self.optimizerF.zero_grad()
                self.optimizerC.zero_grad()

            #
                

                #########################################################################################
                logits_t = self.netC(feat_t)
                pred_xt = F.softmax(logits_t, 1)

                # print('pred_xt.size:', pred_xt.size())

                # print('lab_s.size:', lab_s.size())
                ys_oh = F.one_hot(lab_s, num_classes=self.nclasses).float()
                # print('ys_oh:', ys_oh)
                # print('ys_oh.size:',ys_oh.size())

                #### For predict xs
                logits_s = self.netC(feat_s)
                pred_xs = F.softmax(logits_s, 1)

                M_sce = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))

                M_embed = torch.cdist(feat_s, feat_t) ** 2

                # M_sce = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt), 0, 1))

                ####################################################################
                mask0 = torch.mm(pred_xs, torch.transpose(pred_xt, 0, 1))
                ones1 = torch.ones(feat_s.size(0), feat_t.size(0))
                ones1 = ones1.cuda()
                mask1 = ones1 - mask0


                ######################     -H
                maskH = torch.exp(-mask1)
                # print('maskH:', maskH.mean())



                #########dishuxuanze

                mask1 = torch.exp(mask1)
                # print('mask1:', mask1.mean())




                


                ########### for Barycentres NeW    ######################################################################################
                # print('center')
                M_embedself = torch.cdist(feat_t, feat_t) ** 2

                onebb = torch.ones(feat_t.size(0), self.nclasses)
                onebb = onebb.cuda()

                formax = torch.zeros(1, self.nclasses)
                formax = formax.cuda()


                if (feat_s.size(0) == self.config.batchSize) & (feat_t.size(0) == self.config.batchSize):
                    maskdis = torch.zeros(feat_t.size(0), self.nclasses)
                    maskdis = maskdis.cuda()

                    distantot = torch.zeros(feat_t.size(0), self.nclasses)
                    distantot = distantot.cuda()

                    tongjitot = torch.zeros(feat_t.size(0), self.nclasses)
                    tongjitot = tongjitot.cuda()


                    maskdisself = torch.zeros(feat_t.size(0), self.nclasses)
                    maskdisself = maskdis.cuda()





                    for i in range(feat_t.size(0)):
                        distan = torch.zeros(self.nclasses)
                        distan = distan.cuda()
                        tongji = torch.zeros(self.nclasses)
                        tongji = tongji.cuda()
                        # jieguo = torch.zeros(self.nclasses)

                        for j in range(feat_s.size(0)):
                            distan[lab_s[j]] += M_embed[j, i]
                            tongji[lab_s[j]] += 1

                            distantot[i,lab_s[j]] += M_embed[j, i]
                            tongjitot[i,lab_s[j]] += 1


                        for jj in range(len(tongji)):
                            if tongji[jj] == 0:
                                # tongji[jj] = max(tongji)
                                tongji[jj] = 1

                        jieguo = distan / tongji

                        for ii in range(len(jieguo)):
                            if jieguo[ii] == 0:
                                jieguo[ii] = max(jieguo)





                        maskdis = torch.zeros(feat_t.size(0), self.nclasses)

                        # print('jieguo:',jieguo)
                        # print('max:',max(jieguo))


                        jieguo1 = formax*max(jieguo)-jieguo

                        maskdis[i, :] = jieguo1


                    maskdis1 = F.softmax(maskdis, 1)



                    # maskdis1 = onebb - maskdis1


                    _, preds = torch.max(maskdis1, 1)

                        # M_embedself = torch.cdist(feat_t, feat_t) ** 2

                    for aa in range(len(preds)):
                        for bb in range(len(preds)):

                            distantot[aa, preds[bb]] += M_embedself[bb,  aa ]
                            tongjitot[aa, preds[bb]] += 1


                    for cc in range(len(preds)):
                        for dd in range(self.nclasses):
                            if tongjitot[cc, dd] == 0:
                                tongjitot[cc, dd] += 1


                    maskdisgl = distantot / tongjitot
                    # maskdisglcopy = maskdisgl
                    maskdisglcopy =torch.zeros(feat_t.size(0), self.nclasses)
                    maskdisglcopy = maskdisglcopy.cuda()

                    for nn in range(feat_t.size(0)):
                        maxline = max(maskdisgl[nn,:])
                        for mm in range(self.nclasses):
                            maskdisglcopy[nn,mm] = maxline - maskdisgl[nn,mm]



                    maskdisgl2 = F.softmax(maskdisglcopy, 1)





                    pred_xt1 = (maskdisgl2 + pred_xt) / 2

                    maskdis44 = torch.mm(ys_oh, torch.transpose(pred_xt1, 0, 1))






                    maskdis55 = torch.exp(maskdis44)
                    
                    M = 0.1 * M_embed * maskdis55

                    M_cpu = M.double().detach().cpu().numpy()


                    a, b = ot.unif(feat_s.size()[0]).astype('float64'), ot.unif(feat_t.size()[0]).astype('float64')

                    
                    pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_cpu, 0.01, 0.1)
                    

                    pi = torch.from_numpy(pi).float().cuda()

                    

                    transfer_loss = torch.sum(pi * 0.01 * M)


                else:
                    transfer_loss = 0

                #####################################################################################################







                logits = self.netC(feat_s)

                ############ cross entropy
                lossC = F.cross_entropy(logits, lab_s)

                


                lossT = lossC + transfer_loss
                lossT.backward()

                

                timetrain2 = time.time()
               
                #
                timetrain = timetrain2 - timetrain1

                if self.itr % self.config.log_interval == 0:
                    print('timetrain:', timetrain)


                # self.optimizerD.step()
                self.optimizerF.step()
                self.optimizerC.step()
                # self.optimizermask.step()

                self.lr_scheduler_F.step()
                self.lr_scheduler_C.step()
                # self.lr_scheduler_mask.step()

                lr = self.optimizerF.param_groups[0]['lr']

                

                if self.itr % self.config.log_interval == 0:
                    log_train = 'Train iter: {}, Epoch: {}, lr{} \t Loss Classification: {:.6f} ' \
                                'Method {}'.format(self.itr, self.epoch, lr, lossC.item(), self.config.method)
                    self.log(log_train)


                # if (self.itr > 0.6 * self.config.num_iters) and (self.itr % self.config.save_interval == 0):
                if  (self.itr % self.config.save_interval == 0):


                    timet1 = time.time()
                    # print('timet1:', timet1)

                    mean_class_acc, net_class_acc = self.test()
                    if net_class_acc > self.best_acc:
                        self.best_acc = net_class_acc

                 

                    msg = 'Mean class acc: {}, Net class acc: {}'.format(mean_class_acc, net_class_acc)
                    self.log(msg)
                    msg = 'Best class acc: {}'.format(self.best_acc)
                    self.log(msg)

                    print('Saving model')
                    # print('rho:',self.maskrho[0,0])

                    self.save_state()
                    self.netF.train()
                    self.netC.train()




                    # if self.itr > (self.config.num_iters-20):


                    flopsF, paramsF = profile(self.netF, inputs=(inp_s,))
                    flopsF, paramsF = clever_format([flopsF, paramsF], "%.3f")

                    flopsC, paramsC = profile(self.netC, inputs=(feat_s,))
                    flopsC, paramsC = clever_format([flopsC, paramsC], "%.3f")

                    flops = flopsF + flopsC
                    params = paramsF + paramsC

                    print('flops:', flops)
                    print('params:', params)


             
