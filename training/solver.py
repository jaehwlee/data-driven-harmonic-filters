# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import datetime
import tqdm
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import Model, WaveEncoder, WaveProjector, SupervisedClassifier, TagEncoder, TagDecoder


class Solver(object):
    def __init__(self, data_loader, config):
        # data loader
        self.input_length = config.input_length
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.stage1_step = 100

        # model hyper-parameters
        self.conv_channels = config.conv_channels
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.n_harmonic = config.n_harmonic
        self.semitone_scale = config.semitone_scale
        self.learn_bw = config.learn_bw

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size

        # cuda
        self.is_cuda = torch.cuda.is_available()
        
        # my moidel
        self.wave_encoder_path = './checkpoints/wave_encoder'
        self.classifier_path = './checkpoints/classifier'
        self.classifier_laod_path = '.'
        # Build model
        self.get_dataset()

    def get_dataset(self):
        if self.dataset == 'mtat':
            self.valid_list = np.load(os.path.join(self.data_path, 'mtat', 'valid.npy'))
            self.binary = np.load(os.path.join(self.data_path, 'mtat', 'binary.npy'))
        elif self.dataset == 'dcase':
            df = pd.read_csv(os.path.join(self.data_path, 'dcase', 'df.csv'), delimiter='\t', names=['file', 'start', 'end', 'path', 'split', 'label'])
            df = df[df['split'] == 'val']
            self.valid_list = list(df['path'])
            self.binary = list(df['label'])
        elif self.dataset == 'keyword':
            from data_loader.keyword_loader import get_audio_loader
            self.valid_loader = get_audio_loader(self.data_path, self.batch_size, input_length = self.input_length, tr_val='val')

    
    def load_wave_encoder(self, filename):
        S = torch.load(filename)
        self.wave_encoder.load_state_dict(S)

    def load_classifier(self, filename):
        S = torch.load(filename)
        self.classifier.load_state_dict(S)


    def get_model(self):
        return Model()

    def get_encoder(self):
        return WaveEncoder(conv_channels=self.conv_channels,
                     sample_rate=self.sample_rate,
                     n_fft=self.n_fft,
                     n_harmonic=self.n_harmonic,
                     semitone_scale=self.semitone_scale,
                     learn_bw=self.learn_bw,
                     dataset=self.dataset)



    def get_projector(self):
        return WaveProjector()

    def get_classifier(self):
        return SupervisedClassifier()
    

    def get_tag_encoder(self):
        return TagEncoder()

    def get_tag_decoder(self):
        return TagDecoder()



    def build_model(self):
        # model
        self.model = self.get_model()
        self.wave_encoder = self.get_encoder()
        self.wave_projector = self.get_projector()
        self.tag_encoder = self.get_tag_encoder()
        self.tag_decoder = self.get_tag_decoder()
        self.classifier = self.get_classifier()

        # cuda
        if self.is_cuda:
            self.wave_encoder.cuda()
            self.wave_projector.cuda()
            self.classifier.cuda()
            self.tag_encoder.cuda()
            self.tag_decoder.cuda()


        # load pretrained model
        #if len(self.wave_encoder_path) > 1:
        #    print('load_model!')
        #    self.load_wave_encoder(self.wave_encoder_path)
        #else:
            #print('No model')


        # optimizers
        self.stage1_optimizer = torch.optim.Adam(chain(self.tag_encoder.parameters(), self.tag_decoder.parameters(), self.wave_encoder.parameters(), self.wave_projector.parameters()), self.lr, weight_decay=1e-4)

        self.tag_optimizer = torch.optim.Adam(chain(self.tag_encoder.parameters(), self.tag_decoder.parameters()), self.lr, weight_decay=1e-4)
        self.wave_optimizer = torch.optim.Adam(chain(self.wave_encoder.parameters(), self.wave_projector.parameters()), self.lr, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), self.lr, weight_decay=1e-4)


    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_loss_function(self):
        if self.dataset == 'mtat' or self.dataset == 'dcase':
            return nn.BCELoss()
        elif self.dataset == 'keyword':
            return nn.CrossEntropyLoss()

    def get_ae_loss(self):
        return nn.MSELoss()


    def get_pairwise_loss(self):
        return nn.MSELoss()



    def stage1_train(self):
        # Start training
        self.build_model()
        start_t = time.time()
        current_optimizer = 'adam'
        ae_loss = self.get_ae_loss()
        pairwise_loss = self.get_pairwise_loss()
        best_metric = 0
        drop_counter = 0
        for epoch in range(self.stage1_step):
            # train
            ctr = 0
            drop_counter += 1
            #self.model.cuda()
            #self.model.train()
            self.tag_encoder.cuda()
            self.tag_decoder.cuda()
            self.wave_encoder.cuda()
            self.wave_projector.cuda()
            self.tag_encoder.train()
            self.tag_decoder.train()
            self.wave_encoder.train()
            self.wave_projector.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)

                z1 = self.tag_encoder(y)
                recon_tag = self.tag_decoder(z1)

                r1 = self.wave_encoder(x)
                latent = self.wave_projector(r1)
                #out = self.model(x)

                # Backward
                loss1 = ae_loss(recon_tag, y)
                loss2 = pairwise_loss(latent, z1)
                loss = loss1 + loss2

                #loss = reconst_loss(out, y)
                #self.tag_optimizer.zero_grad()
                #self.wave_optimizer.zero_grad()
                self.stage1_optimizer.zero_grad()
                loss.backward()
                self.stage1_optimizer.step()
                #self.wave_optimizer.step()
                #self.tag_optimizer.step()

                # Log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                epoch+1, self.stage1_step, ctr, len(self.data_loader), loss.item(),
                                datetime.timedelta(seconds=time.time()-start_t)))


            # schedule optimizer
            torch.save(self.wave_encoder.state_dict(), os.path.join(self.wave_encoder_path, 'best_model.pth'))
            current_optimizer, drop_counter = self.stage1_opt_schedule(current_optimizer, drop_counter)


        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def stage1_opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 60:
            self.load_wave_encoder = os.path.join(self.wave_encoder_path, 'best_model.pth')
            self.tag_optimizer = torch.optim.SGD(chain(self.tag_encoder.parameters(), self.tag_decoder.parameters()), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            self.wave_optimizer = torch.optim.SGD(chain(self.wave_encoder.parameters(), self.wave_projector.parameters()), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)

            self.stage1_optimizer = torch.optim.SGD(chain(self.wave_encoder.parameters(), self.wave_projector.parameters(), self.tag_encoder.parameters(), self.tag_decoder.parameters()), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load_wave_encoder = os.path.join(self.wave_encoder_path, 'best_model.pth')
            for pg in self.stage1_optimizer.param_groups:
                pg['lr']= 0.0001
            #for pg in self.tag_optimizer.param_groups:
            #    pg['lr'] = 0.0001
           # 
           # for pg in self.wave_optimizer.param_groups:
           #     pg['lr'] = 0.0001

            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
            '''
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load = os.path.join(self.model_save_path, 'best_model.pth')
            for pg in self.wave_optimizer.param_groups:
                pg['lr'] = 0.00001

            for pg in self.tag_optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
            '''
        return current_optimizer, drop_counter



    def stage2_opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 60:
            self.load_classifier = os.path.join(self.classifier_path, 'best_model.pth')
            self.optimizer = torch.optim.SGD(self.classifier.parameters(), 0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load_classifier = os.path.join(self.classifier, 'best_model.pth')
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load_classifier = os.path.join(self.classifier, 'best_model.pth')
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter


    def wave_encoder_save(self, filename):
        wave_encoder = self.wave_encoder.state_dict()
        torch.save({'wave_encoder': wave_encoder}, filename)

    def classifier_save(self, filename):
        classifier = self.classifier.state_dict()
        torch.save({'classifier': classifier}, filename)

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)
        return roc_aucs, pr_aucs

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'dcase':
            npy_path = fn
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        if length < self.input_length:
            nnpy = np.zeros(self.input_length)
            ri = int(np.floor(np.random.random(1) * (self.input_length - length)))
            nnpy[ri:ri+length] = raw
            raw = nnpy
            length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_validation_score(self):
        self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.get_loss_function()
        index = 0
        for line in tqdm.tqdm(self.valid_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'dcase':
                fn = line

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            if self.dataset == 'mtat':
                ground_truth = self.binary[int(ix)]
            elif self.dataset == 'dcase':
                ground_truth = np.fromstring(self.binary[index][1:-1], dtype=np.float32, sep=' ')

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
            z1 = self.wave_encoder(x)
            out = self.classifier(z1)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)

            gt_array.append(ground_truth)
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        if self.dataset == 'mtat':
            roc_auc, pr_auc = self.get_auc(est_array, gt_array)
            return roc_auc, pr_auc, loss
        elif self.dataset == 'dcase':
            prd_array = (est_array > 0.1).astype(np.float32)
            f1 = metrics.f1_score(gt_array, prd_array, average='samples')
            print('f1: %.4f' % f1)
            return f1, loss

    def get_validation_acc(self):
        self.wave_encoder.eval()
        self.classifier.eval()
        reconst_loss = self.get_loss_function()
        est_array = []
        gt_array = []
        losses = []
        for x, y in tqdm.tqdm(self.valid_loader):
            x = self.to_var(x)
            y = self.to_var(y)
            z1 = self.wave_encoder(x)
            out = self.classifier(z1)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()
            y = y.detach().cpu()
            _prd = [int(np.argmax(prob)) for prob in out]
            for i in range(len(_prd)):
                est_array.append(_prd[i])
                gt_array.append(y[i])
        est_array = np.array(est_array)
        gt_array = np.array(gt_array)
        acc = metrics.accuracy_score(gt_array, est_array)
        loss = np.mean(losses)
        print('accuracy: %.4f' % acc)
        print('loss: %.4f' % loss)
        return acc, loss


    def stage2_train(self):
        # Start training
        self.build_model()
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0
        #for epoch in range(2):
        for epoch in range(self.n_epochs):
            # train
            ctr = 0
            drop_counter += 1
            self.wave_encoder.cuda()
            self.wave_encoder.train(False)
            self.classifier.cuda()
            self.classifier.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                z1 = self.wave_encoder(x)
                out = self.classifier(z1)

                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                                datetime.timedelta(seconds=time.time()-start_t)))

            # validation
            if self.dataset == 'mtat':
                roc_auc, pr_auc, loss = self.get_validation_score()
                score = 1 - loss
                if score > best_metric:
                    print('best model!')
                    best_metric = score
                    torch.save(self.classifer.state_dict(), os.path.join(self.classifier_path, 'best_model.pth'))
            elif self.dataset == 'dcase':
                if epoch > 10:
                    f1, loss = self.get_validation_score()
                    score = 1 - loss
                    score = f1
                else:
                    score = 0
                if score > best_metric:
                    print('best model!')
                    best_metric = score
                    #torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
            elif self.dataset == 'keyword':
                acc, loss = self.get_validation_acc()
                score = 1 - loss
                if score > best_metric:
                    print('best model: %.4f' % acc)
                    best_metric = score
                    #torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))

            # schedule optimizer
            current_optimizer, drop_counter = self.stage2_opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

