import torch
import numpy as np
from torch import optim
import cal_utils as utils
import torch.nn.functional as F
import data_loader
import scipy.io as sio
import copy

import time

class Solver(object):
    def __init__(self, config):

        self.output_shape = config.output_shape
        data = data_loader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        (self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels) = data

        self.n_view = len(self.train_data)
        for v in range(self.n_view):
            if min(self.train_labels[v].shape) == 1:
                self.train_labels[v] = self.train_labels[v].reshape([-1])
            if min(self.val_labels[v].shape) == 1:
                self.val_labels[v] = self.val_labels[v].reshape([-1])
            if min(self.test_labels[v].shape) == 1:
                self.test_labels[v] = self.test_labels[v].reshape([-1])

        if len(self.train_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.train_labels).reshape([-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.train_labels[0].shape[1]

        if self.output_shape == -1:
            self.output_shape = self.num_classes

        self.dropout_prob = 0.5
        self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.alpha1 = config.alpha1
        self.beta1 = config.beta1
        self.alpha2 = config.alpha2
        self.beta2 = config.beta2
        self.view_id = config.view_id
        self.gpu_id = config.gpu_id
        self.epochs = config.epochs
        self.sample_interval = config.sample_interval
        self.just_val = config.just_val

        print("datasets: %s, batch_size: %d, output_shape: %d, hyper-alpha1: %f，hyper-beta1： %f，hyper-alpha2：%f，hyper-beta2: %f"% \
                (config.datasets, self.batch_size, self.output_shape, self.alpha1, self.beta1, self.alpha2, self.beta2))

        Wfile = 'classify_' + str(self.output_shape) + 'X' + str(self.num_classes) + self.datasets +'.mat'
        try:
            self.W = sio.loadmat( './classifyW/' + Wfile)['W']
        except Exception as e:
            W = torch.Tensor(self.output_shape, self.output_shape)
            self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: self.num_classes]
            sio.savemat(Wfile, {'W': self.W.cpu().data.numpy()})
            
        self.runing_time = config.running_time


    def to_one_hot(self, x):
        if len(x.shape) == 1 or x.shape[1] == 1:
            one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
            labels = one_hot
            y = torch.tensor(labels).cuda()
        else:
            y = torch.tensor(x.astype('float32')).cuda()
        return y

    def view_result(self, _acc):
        res = ''
        res += ((' - mean: %.5f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
        for _i in range(self.n_view):
            for _j in range(self.n_view):
                if _i != _j:
                    res += ('%.5f' % _acc[_i, _j]) + ','
        return res


    def train(self):
        if self.view_id >= 0:
            start = time.time()
            self.train_view(self.view_id)
        else:
            start = time.time()
            for v in range(self.n_view):
                self.train_view(v)

        end = time.time()
        runing_time = end - start
        if self.runing_time:
            print('runing_time: ' + str(runing_time))
            return runing_time


        test_fea, test_lab = [], []
        for v in range(self.n_view):
            tmp = sio.loadmat('features/' + self.datasets + '_' + str(v) + '.mat')
            test_fea.append(tmp['test_fea'])
            test_lab.append(tmp['test_lab'].reshape([-1,]) if min(tmp['test_lab'].shape) == 1 else tmp['test_lab'])

        test_results = utils.multi_test(test_fea, test_lab)
        print("test resutls:" + self.view_result(test_results))
        sio.savemat('features/' + self.datasets + '_MARS_test_feature_results.mat', {'test': test_fea, 'test_labels': test_lab})
        return test_results


    def pair_loss(self, x, y, lab1, lab2):
        cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) * 2.
        theta12 = cos(x, y)
        theta11 = cos(x, x)
        theta22 = cos(y, y)
        sim12 = (lab1.float().mm(lab2.float().t()) > 0.5).float()
        sim11 = (lab1.float().mm(lab1.float().t()) > 0.5).float()
        sim22 = (lab2.float().mm(lab2.float().t()) > 0.5).float()

        return  [(torch.log( 1. + torch.exp( theta12 )) - sim12 * theta12).mean(), 
                 (torch.log( 1. + torch.exp( theta11 )) - sim11 * theta11).mean(),
                 (torch.log( 1. + torch.exp( theta22 )) - sim22 * theta22).mean()]


    def orthogonal_loss(self, W1, W2):
        weight_squared = torch.mm(W1, W2.t())  # (N * C) * H * H
        ones = torch.ones(W1.shape[0], W1.shape[0], dtype=torch.float32)  # (N * C) * H * H
        weight_squared = (torch.mm(W1, W2.t())**2).sum()
        return ((weight_squared** 2)).sum()

    def train_view(self, view_id):

        from to_seed import to_seed
        to_seed(seed=0)
        import os
        import torch
        ds_criterion = lambda x, y: ( -torch.sum(y * torch.log(F.softmax(x, dim=1)), dim=1) ).mean()
        de_criterion = lambda x: torch.mean( torch.log(torch.max(F.softmax(x, dim=1), dim=1)[0]) )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        from model import Data_Net
        from model import Lab_Net
        #labnetend = False
        '''
        sio.savemat('features/' + self.datasets + '_raw' + str(view_id) + '.mat', {'val_fea': self.val_data[view_id], 'val_lab': self.val_labels[view_id],
                                                                                  'test_fea': self.test_data[view_id], 'test_lab': self.test_labels[view_id]})
        '''
        if view_id == 0:
            Labnet = Lab_Net(input_dim=self.num_classes, out_dim=self.output_shape).cuda()
            best_model_wts = copy.deepcopy(Labnet.state_dict())
            Datanet = Data_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape).cuda()

            W = torch.tensor(self.W).cuda()
            W.requires_grad=False

            #stage 1 for Labnet
            get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
            params_lnet = get_grad_params(Labnet)
            params_dnet = get_grad_params(Datanet)

            optimizer_lnet = optim.Adam(params_lnet, self.lr[view_id], [0.5, 0.999])
            optimizer_dnet = optim.Adam(params_dnet, self.lr[view_id], [0.5, 0.999])

            discriminator_losses, losses, val_results = [], [], []

            tr_ml_loss, tr_d_loss, val_ml_loss, val_d_loss = [], [], [], []
            val_loss_min = 1e9

            for epoch in range(self.epochs):
                print(('ViewID: %d, Epoch %d/%d') % (view_id, epoch + 1, self.epochs))

                rand_lidx = np.arange(self.train_labels[view_id].shape[0])
                np.random.shuffle(rand_lidx)
                batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

                rand_didx = np.arange(self.train_data[view_id].shape[0])
                np.random.shuffle(rand_didx)

                k = 0
                mean_loss = []
                mean_tr_ml_loss = []
                for batch_idx in range(batch_nums):
                    lidx = rand_lidx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                    didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                    view1_labs = self.to_one_hot(self.train_labels[view_id][lidx])
                    view1_data = self.to_one_hot(self.train_labels[view_id][lidx])

                    view2_labs = self.to_one_hot(self.train_labels[view_id][didx])
                    view2_data = self.to_one_hot(self.train_data[view_id][didx])

                    optimizer_dnet.zero_grad()
                    optimizer_lnet.zero_grad()

                    out_fc1 = Labnet(view1_data)[-1]
                    pred1 = out_fc1.view([out_fc1.shape[0], -1]).mm(W)

                    out_fc2 = Datanet(view2_data)[-1]
                    pred2 = out_fc2.view([out_fc2.shape[0], -1]).mm(W)

                    pld_loss, pll_loss, pdd_loss = self.pair_loss(out_fc1, out_fc2, view1_labs, view2_labs)

                    view1_labs_idx = view1_labs.sum(1) > 0
                    view2_labs_idx = view2_labs.sum(1) > 0

                    ctl_loss = ds_criterion(pred1[view1_labs_idx], view1_labs[view1_labs_idx])

                    ctd_loss = ds_criterion(pred2[view2_labs_idx], view2_labs[view2_labs_idx])

                    out_fc2_ex = Datanet(view2_data)[-2]
                    pred2_ex = out_fc2_ex.view([out_fc2_ex.shape[0], -1]).mm(W)
                    cte_loss = de_criterion(pred2_ex[view2_labs_idx])

                    loss = (pld_loss + pll_loss + pdd_loss) * 1. + ctd_loss * self.alpha1 + cte_loss * self.beta1

                    loss.backward()
                    optimizer_lnet.step()
                    optimizer_dnet.step()

                    mean_loss.append(loss.item())
                    mean_tr_ml_loss.append(pdd_loss.item())

                    if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                        Labnet.eval()
                        Datanet.eval()

                        losses.append(np.mean(mean_loss))

                        val_labs = self.to_one_hot(self.val_labels[view_id])
                    
                        val_lfea = utils.predict(lambda x: Labnet(x)[-1].view([x.shape[0], -1]), val_labs, self.batch_size).reshape([val_labs.shape[0], -1])
                        val_dfea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]), self.val_data[view_id], self.batch_size).reshape([self.val_data[view_id].shape[0], -1])

                        val_lpre = utils.predict(lambda x: Labnet(x)[-1].view([x.shape[0], -1]).mm(W).view([x.shape[0], -1]), val_labs, self.batch_size).reshape( [val_labs.shape[0], -1])
                        val_dpre = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]).mm(W).view([x.shape[0], -1]), self.val_data[view_id], self.batch_size).reshape(
                                           [self.val_data[view_id].shape[0], -1])

                        vld_loss, vll_loss, vdd_loss= self.pair_loss((torch.from_numpy(val_lfea)).cuda(), (torch.from_numpy(val_dfea)).cuda(), val_labs, val_labs)

                        cvl_loss = ds_criterion((torch.from_numpy(val_lpre)).cuda(), val_labs)
                        cvd_loss = ds_criterion((torch.from_numpy(val_dpre)).cuda(), val_labs)

                        print('Train_ViewID: %d, Epoch %d/%d, Batch %02d/%d, pld_loss:%.4f, pll_loss:%.4f, pdd_loss:%.4f, ctl_loss:%.4f, ctd_loss:%.4f' % \
                        	                (view_id, epoch + 1, self.epochs, batch_idx+1, batch_nums, pld_loss, pll_loss, pdd_loss, ctl_loss, ctd_loss))

                        bval_fea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]), self.val_data[view_id], self.batch_size).reshape([self.val_data[view_id].shape[0], -1])
                        btest_fea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]), self.test_data[view_id], self.batch_size).reshape( [self.test_data[view_id].shape[0], -1])

                        if val_loss_min > vld_loss and not self.just_val:
                            val_fea = bval_fea
                            test_fea = btest_fea
                            best_model_wts = copy.deepcopy(Labnet.state_dict())

                        elif self.just_val:
                            tr_ml_loss.append(np.mean(mean_tr_ml_loss))
                            val_d_loss.append((cvl_loss.item() + cvd_loss.item() )/2)
                    k += 1

            print('save name: %s'%('features/' + self.datasets + '_' + 'lab' + '.mat'))
            sio.savemat('features/'+self.datasets+'_'+str(view_id)+'.mat', {'val_fea': val_fea, 'val_lab': self.val_labels[view_id], 'test_fea': test_fea, 'test_lab': self.test_labels[view_id]})
            torch.save( Labnet.state_dict(), 'models/' + self.datasets + '_' + 'labnet_params' + '.pth')
            torch.save( Datanet.state_dict(), 'models/' + self.datasets + 'v' +str(view_id) + '_params' + '.pth' )

        ### Model 1 2 3 ...
        #labnetend = True
        if view_id >0:
            Labnet = Lab_Net(input_dim=self.num_classes, out_dim=self.output_shape).cuda()
            Labnet.load_state_dict(torch.load('models/' + self.datasets + '_' + 'labnet_params' + '.pth'))
            Datanet = Data_Net(input_dim=self.input_shape[view_id], out_dim=self.output_shape).cuda()

            W = torch.tensor(self.W).cuda()
            W.requires_grad=False

            #stage 1 for Labnet
            get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
            params = get_grad_params(Datanet)
            optimizer = optim.Adam(params, self.lr[view_id], [0.5, 0.999])

            discriminator_losses, losses, val_results = [], [], []

            tr_ml_loss, tr_d_loss, val_ml_loss, val_d_loss = [], [], [], []
            val_loss_min = 1e9

            for epoch in range(self.epochs):
                print(('ViewID: %d, Epoch %d/%d') % (view_id, epoch + 1, self.epochs))
                f_idx = np.arange(self.train_labels[view_id].shape[0])
                d_idx = np.arange(self.train_labels[view_id].shape[0])
                np.random.shuffle(f_idx)
                np.random.shuffle(d_idx)

                batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

                k = 0
                mean_loss = []
                mean_tr_ml_loss, mean_tr_d_loss = [], []
                for batch_idx in range(batch_nums):
                    fidx = f_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                    didx = d_idx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                    slect_labs = self.to_one_hot(self.train_labels[view_id][fidx])
                    slect_feas = utils.predict(lambda x: Labnet(x)[-1].view([x.shape[0], -1]), slect_labs, self.batch_size).reshape([slect_labs.shape[0], -1])

                    view2_labs = self.to_one_hot(self.train_labels[view_id][didx])
                    view2_data = torch.tensor(self.train_data[view_id][didx]).cuda()

                    optimizer.zero_grad()

                    out_fc = Datanet(view2_data)[-1]
                    pred = out_fc.view([out_fc.shape[0], -1]).mm(W)

                    pld_loss, pll_loss, pdd_loss = self.pair_loss(torch.tensor(slect_feas).cuda(), out_fc, slect_labs, view2_labs)

                    view2_labs_idx = view2_labs.sum(1) > 0
                    ctd_loss = ds_criterion(pred[view2_labs_idx], view2_labs[view2_labs_idx])

                    out_fc_ex = Datanet(view2_data)[-2]
                    pred_ex = out_fc_ex.view([out_fc_ex.shape[0], -1]).mm(W)
                    cte_loss = de_criterion(pred_ex[view2_labs_idx])

                    loss = (pld_loss + pdd_loss)*1. + self.alpha2 * ctd_loss + self.beta2 * cte_loss

                    loss.backward()
                    optimizer.step()

                    mean_loss.append(loss.item())
                    mean_tr_ml_loss.append(pld_loss.item())
                    mean_tr_d_loss.append(ctd_loss.item())

                    if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                        Datanet.eval()
                        losses.append(np.mean(mean_loss))

                        val_labs = self.to_one_hot(self.val_labels[view_id])

                        val_dfea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]), self.val_data[view_id], self.batch_size).reshape([self.val_data[view_id].shape[0], -1])
                        val_dpre = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]).mm(W).view([x.shape[0], -1]), self.val_data[view_id], self.batch_size).reshape(
                                           [self.val_data[view_id].shape[0], -1])

                        val_lfea = utils.predict(lambda x: Labnet(x)[-1].view([x.shape[0], -1]), val_labs, self.batch_size).reshape([val_labs.shape[0], -1])

                        vld_loss, vll_loss, vdd_loss = self.pair_loss(torch.from_numpy(val_lfea).cuda(), torch.from_numpy(val_dfea).cuda(), val_labs, val_labs)

                        cvd_loss = ds_criterion(torch.tensor(val_dpre).cuda(), val_labs)

                        print('Train_ViewID: %d, Epoch %d/%d, Batch %02d/%d, pld_loss:%.4f, pll_loss:%.4f, pdd_loss:%.4f,  ctd_loss:%.4f' % \
                                             (view_id, epoch + 1, self.epochs, batch_idx+1, batch_nums, pld_loss, pll_loss, pdd_loss, ctd_loss))

                        bval_fea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]), self.val_data[view_id],self.batch_size).reshape([self.val_data[view_id].shape[0], -1])
                        btest_fea = utils.predict(lambda x: Datanet(x)[-1].view([x.shape[0], -1]),self.test_data[view_id], self.batch_size).reshape([self.test_data[view_id].shape[0], -1])

                        if val_loss_min > vdd_loss and not self.just_val:
                            val_fea = bval_fea
                            test_fea = btest_fea

                        elif self.just_val:
                            tr_ml_loss.append(np.mean(mean_tr_ml_loss))
                            tr_d_loss.append(np.mean(mean_tr_d_loss))
                            val_d_loss.append(cvd_loss.item())
                    k += 1

            print('save name: %s'%('features/' + self.datasets + '_' +str(view_id)+ '.mat'))
            sio.savemat('features/'+self.datasets+'_'+str(view_id)+'.mat', {'val_fea': val_fea, 'val_lab': self.val_labels[view_id], 'test_fea': test_fea, 'test_lab': self.test_labels[view_id]})
            torch.save(Datanet.state_dict(), 'models/' + self.datasets + 'v' + str(view_id) + '_params' + '.pth')
            

