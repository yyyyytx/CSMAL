from active_trainers.BaseTrainer import BaseActiveTrainer
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
import torch.optim as optim

from active_utils.ema import EMA
from active_utils.lr_scheduler import WarmupCosineLrScheduler



import torch.nn.functional as F

import numpy as np
import torch.nn as nn
from arch.resnetv1 import ResNet18, ResNet18_metric
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from active_utils.losses import CenterSeperateMarginLoss
from torch.utils.data import Dataset
from data_utils import caltech101

class kCenterGreedy():

  def __init__(self, features, metric='euclidean'):
    # self.X = X
    # self.y = y
    # self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = features
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.features.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:

      x = self.features[cluster_centers]

      dist = pairwise_distances(self.features, x, metric=self.metric)
      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)
    # print('min_distances:',self.min_distances)

  def select_batch_(self, already_selected, N, **kwargs):
    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      # print('Getting transformed features...')
      # self.features = model.transform(self.X)
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for i in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.

      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))

    self.already_selected = already_selected

    return new_batch





def mixup_data(x1, y1, alpha=1.0, use_cuda=True, device='cuda:0'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    # l = max(lam, 1-lam)

    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x1 + (1 - lam) * x1[index, :]
    y_a, y_b = y1, y1[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MetricTrainer(BaseActiveTrainer):
    def __init__(self, config, device=torch.device('cuda:0'), logger=None, positive_margin=0.3,start_epoch=5):
        self.config = config
        self.device = device
        self.logger = logger
        self.module_lr = 0.01

        self.embeding_dim=128
        self.weight = 0.5
        self.metric_epoch = 120

        # wether random take 10000 samples
        self.is_random_sample = False
        self.subset_size = 10000
        self.ema_decay = 0.999
        self.negative_margin = 0.01
        self.start_epoch = start_epoch
        self.positive_margin = positive_margin
        self.pesudo_idx = None
        self.u_weight = 1.



    # def build_model(self, base_net):
    def start_loop(self, net, train_dataset, test_dataset, unlabel_dataset, n_select, n_iteration, index_loop=1, is_semi=False, is_semi_metric=False, pesudo_count=1000, semi_ind=None, is_mixup=True):
        method_str = '====================Metric Method with pesudo label==============='
        self.logger.info(method_str)
        self.loop_num = index_loop
        self.pesudo_count=pesudo_count
        total_train_ind = np.arange(6471)
        self.semi_ind = semi_ind
        self.label_ind = np.random.permutation(total_train_ind)[:n_select]
        self.unlabel_ind = np.setdiff1d(total_train_ind, self.label_ind)
        self.n_cls = int(self.config['n_cls'])
        self.n_select = n_select
        self.is_semi_metric = is_semi_metric
        self.is_mixup = is_mixup

        print(self.n_cls)

        self.ema_iter = 0


        self.best_acc = 0.
        self.best_meric_acc = 0.

        # self.lb_guessor = LabelGuessor(thresh=0.95)
        if is_semi is True:
        #     model = WideResnet(n_classes=10,
        #                        k=2,
        #                        n=28)
            self.in_feats = 128
        else:
        #     model = ResNet18_metric(num_classes=10).to(self.device)
        #     model.apply(self.init_weights)
            self.in_feats = 512
        # model.train()
        # model.to(self.device)
        #
        #
        #
        #
        # ema = EMA(model, 0.999)
        self.pesudo_idx, self.pesudo_label = None, None

        for i in range(n_iteration):
            self.old_mean_embdedding = None
            self.mean_embedding = torch.zeros(self.config['n_cls'], self.in_feats)
            self.is_start_semi = False
            self.index_iteration = i
            str = '[Iteration %d] select index:' % i
            self.logger.info(str)
            # self.logger.info(self.label_ind.tolist())
            # summary_log = '[LOOP %d]Iteration %d labeled num %d, unlabel num %d' % (
            #     index_loop, i, len(self.label_ind), len(self.unlabel_ind))
            # print(summary_log)
            # self.logger.info(summary_log)
            # self.start_train(net, train_dataset)
            # self.train_pesudo_data(net, train_dataset)
            # dlval = caltech101.get_dataloader(train_datasets=caltech101.get_test_dataset(), batch_size=self.config['test_batch_size'])
            #
            # # dlval = cifar.get_val_loader(dataset=self.config['ds_name'], batch_size=64, num_workers=2)
            # print(len(dlval.dataset))

            # dl_x, dl_u = cifar.get_sub_train_loader(dataset=self.config['ds_name'],
            #                                         batch_size=self.config['train_batch_size'],
            #                                         ind_x=self.label_ind,
            #                                         ind_u=self.unlabel_ind)
            ds_u = caltech101.get_unlabel_dataset()
            ds_u = torch.utils.data.Subset(ds_u, self.unlabel_ind)
            dl_u = DataLoader(dataset=ds_u,
                                        batch_size=self.config['test_batch_size'],
                                        shuffle=False,
                                        num_workers=2)

            if is_semi is True:
                model = WideResnet(n_classes=self.config['n_cls'],
                                   k=2,
                                   n=28)
                self.in_feats = 128
                model.train()
                model.to(self.device)
                ema = EMA(model, 0.999)
                self.train_semi_data(model, ema)

            else:
                model = ResNet18_metric(num_classes=self.config['n_cls']).to(self.device)
                model.apply(self.init_weights)
                self.in_feats = 512
                model.train()
                model.to(self.device)

                self.train_pesudo_data(model)
                # self.start_train(model, criterions)


            # acc = self.model_accuracy(net, test_loader)
            # acc_info = '[LOOP %d]Iteration %d Accuracy %.4f' % (self.loop_num, i, acc)
            # print(acc_info)
            # self.logger.info(acc_info)
            # acc = self.model_accuracy(self.model_ema.model, test_loader)
            # acc_info = '[LOOP %d]Iteration %d EMA Accuracy %.4f' % (self.loop_num, i, acc)
            # print(acc_info)
            # self.logger.info(acc_info)
            # break
            # self.plot_trained_data(models, unlabel_dataset)
            #self.plot_unlabel_data(models, unlabel_dataset)
            # self.plot_test_data(models, test_dataset)

            self.pesudo_idx, self.pesudo_label = self.get_pesudo_idx(model, dl_u, self.positive_margin)

            acc, metric_acc = self.model_accuracy(model)

            # if acc > self.best_acc:
            #     self.best_acc = acc
            # if metric_acc > self.best_meric_acc:
            #     self.best_meric_acc = metric_acc
            acc_info = '[Iteration]epoch %d Accuracy %.4f metric acc %.4f' % (i, acc, metric_acc)
            print(acc_info)
            self.logger.info(acc_info)
            if is_semi is True:
                break

            query_ind = self.get_query_idx(model, dl_u, n_select)
            self.label_ind = np.append(self.label_ind, query_ind)
            self.label_ind = np.unique(self.label_ind)
            self.label_ind = np.random.permutation(self.label_ind)
            self.unlabel_ind = np.setdiff1d(total_train_ind, self.label_ind)


    def train_pesudo_data(self, model):


        # dl_x, dl_g = cifar.get_combine_train_loader(self.config['ds_name'], self.config['train_batch_size'], self.label_ind, self.pesudo_idx, self.pesudo_label)

        ds_x = caltech101.get_train_dataset()
        ds_x = torch.utils.data.Subset(ds_x, self.label_ind)
        dl_x = DataLoader(dataset=ds_x,
                          batch_size=self.config['train_batch_size'],
                          shuffle=True,
                          num_workers=2)

        ds_g = caltech101.get_train_dataset()
        ds_g = torch.utils.data.Subset(ds_g, self.label_ind)
        dl_g = DataLoader(dataset=ds_g,
                          batch_size=self.config['train_batch_size'],
                          shuffle=True,
                          num_workers=2)
        # dl_x, dl_g = cifar.get_guess_train_loader('CIFAR10', self.config['train_batch_size'], self.label_ind, self.pesudo_idx, self.pesudo_label)



        optim_backbone = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                                   weight_decay=self.config['weight_decay'])
        sched_backbone = torch.optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=self.config['milestones'])
        # sched_backbone = WarmupCosineLrScheduler(optim_backbone, max_iter=np.inf, warmup_iter=0)

        criterion_backbone = nn.CrossEntropyLoss().to(self.device)
        criterion_metric = CenterSeperateMarginLoss(in_feats=self.in_feats, n_classes=int(self.config['n_cls']),
                                                    margin=self.positive_margin,
                                                    distance=1.,
                                                    device=self.device)
        criterion_u = nn.CrossEntropyLoss(reduction='none').to(self.device)
        criterions = {'cls': criterion_backbone, 'metric': criterion_metric, 'semi': criterion_u}


        str = 'label len : %d' % (len(dl_x.dataset))
        print('combine len : %d' % (len(dl_g.dataset)))

        print(str)
        self.logger.info(str)
        for epoch in range(self.config['n_epoch']):
            # if self.pesudo_idx is not None:
            #     self.train_pesudo_each_epoch(model, optim_backbone, criterions, train_loader, combine_loader, epoch)
            # else:
            #     self.train_each_epoch(model, optim_backbone, criterions, train_loader, epoch, 0)
            self.train_pesudo_each_epoch(model, optim_backbone, criterions, dl_x, dl_g, epoch)

            sched_backbone.step()

    def train_pesudo_each_epoch(self, model, optimizer, criterions, dl_x, dl_g, epoch):
        model.train()
        total_uloss = 0.0
        total_loss = 0.0
        total_cls_loss = 0.0
        total_metric_loss = 0.0
        batch_count = 0
        negative_samples = 0


        labeled_train_iter = iter(dl_x)
        combine_train_iter = iter(dl_g)
        n_iteration_per_epoch = len(dl_g.dataset) // self.config['train_batch_size']
        for n_i in range(n_iteration_per_epoch):
            try:
                Xs, ys, _ = labeled_train_iter.next()
                Xs, ys, _ = Xs.to(self.device), ys.to(self.device)
            except:
                labeled_train_iter = iter(dl_x)
                Xs, ys, _ = labeled_train_iter.next()
                Xs, ys, _ = Xs.to(self.device), ys.to(self.device)

            try:
                Xs_u, ys_u, _ = combine_train_iter.next()
                Xs_u, ys_u, _ = Xs_u.to(self.device), ys_u.to(self.device)
            except:
                combine_train_iter = iter(dl_g)
                Xs_u, ys_u, _ = combine_train_iter.next()
                Xs_u, ys_u, _ = Xs_u.to(self.device), ys_u.to(self.device)



            #mixup
            if self.is_mixup is True:
                inputs, targets_a, targets_b, lam = mixup_data(Xs_u, ys_u, alpha=0.75)
                y_uhats, _ = model(inputs)
                cls_loss = mixup_criterion(criterions['cls'], y_uhats, targets_a, targets_b, lam)
            else:
            #non mixup
                y_uhats, _ = model(Xs_u)
                cls_loss = criterions['cls'](y_uhats, ys_u)


            y_hats, feat_x = model(Xs)
            # y_hats_u, feat_u = model(Xs_u)
            # cls_loss = criterions['cls'](y_hats, ys)

            # ---------   metric loss   ---------------
            metric_loss = criterions['metric'](feat_x, ys)
            # metric_loss = criterions['metric'](feat_u, ys_u)


            total_cls_loss += cls_loss
            total_metric_loss += metric_loss
            loss = cls_loss + metric_loss
            # print('metric cls loss %.4f metric loss %.4f' % (cls_loss, metric_loss))
            total_loss += loss
            batch_count += 1

            # print(metric_loss)
            optimizer.zero_grad()
            # loss.backward()
            loss.backward()

            # for name, param in model.named_parameters():
            #     print(' name:', name, ' grad:', param.grad)
            #
            #
            # exit(0)
            optimizer.step()
            # ema.update_params()
        self.mean_embedding = criterions['metric'].mean_feats.detach().cpu().numpy()

        # ema.update_buffer()
        str = '[LOSS] epoch %d loss %.4f cls_loss %.4f metric_loss %.4f negative samples %d)' % (
            epoch, total_loss / batch_count, total_cls_loss / batch_count, total_metric_loss / batch_count,
            negative_samples)
        print(str)
        # logger.info(str)




    def model_accuracy(self, net):
        net.eval()
        correct_num = 0
        dlval = caltech101.get_dataloader(train_datasets=caltech101.get_test_dataset(), train_batch_size=self.config['test_batch_size'],
                                          shuffle=False)


        embedding_features = []
        true_labels = []
        for Xs, ys, _ in dlval:
            Xs = Xs.to(self.device)
            ys = ys.to(self.device)

            with torch.set_grad_enabled(False):
                y_hats,avg_feat = net(Xs)
                _, preds = torch.max(y_hats, 1)

            true_labels.append(ys)

            correct_num += torch.sum(preds == ys.data)

            embedding_features.append(avg_feat)

        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        distanc_matrix = pairwise_distances(embedding_features, self.mean_embedding, metric='l2')
        argsort = np.argsort(distanc_matrix, axis=1)
        argdist = np.take_along_axis(distanc_matrix, argsort, axis=1)
        metric_cls = np.argmin(distanc_matrix, axis=1)

        # metric_correct_idx = np.where(metric_cls == true_labels)[0]

        # net.eval()
        # dlval = get_val_loader(batch_size=128, num_workers=0)
        # matches = []
        # for ims, lbs in dlval:
        #     ims = ims.cuda()
        #     lbs = lbs.cuda()
        #     with torch.no_grad():
        #         logits,_ = net(ims)
        #         scores = torch.softmax(logits, dim=1)
        #         _, preds = torch.max(scores, dim=1)
        #         match = lbs == preds
        #         matches.append(match)
        # matches = torch.cat(matches, dim=0).float()
        # acc = torch.mean(matches)
        # return acc


        return (float(correct_num) / float(len(dlval.dataset))), np.sum(true_labels == metric_cls) / len(embedding_features)

    def get_pesudo_idx(self, net, dl_u, thresh):

        net.eval()

        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in dl_u:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = net(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                embedding_features.append(avg_feat)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        predict_cls = torch.cat(predict_cls, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        distanc_matrix = pairwise_distances(embedding_features, self.mean_embedding, metric='l2')

        argsort = np.argsort(distanc_matrix, axis=1)
        argdist = np.take_along_axis(distanc_matrix, argsort, axis=1)
        metric_cls = np.argmin(distanc_matrix, axis=1)

        metric_distance = np.min(distanc_matrix, axis=1)
        quarter_distance = metric_distance.max()

        # t = (self.index_iteration + 1) * 0.1
        # if t > self.positive_margin:
        #     t = self.positive_margin
        pesudo_idx = np.where(metric_distance < self.positive_margin)[0]

        pesudo_arg = np.argsort(metric_distance[pesudo_idx])[:self.pesudo_count]

        select_pesudo = pesudo_idx[pesudo_arg]
        pesudo_label = metric_cls[select_pesudo]

        # pesudo_label = metric_cls[pesudo_idx]
        str = 'pesudo count %d correct %d %.4f' % (len(select_pesudo), np.sum(true_labels[select_pesudo] == metric_cls[select_pesudo]),
                                                   len(select_pesudo) / np.sum(true_labels[select_pesudo] == metric_cls[select_pesudo]))
        print(str)
        self.logger.info(str)
        return self.unlabel_ind[select_pesudo], pesudo_label

        # print('abs 0.2 short correct %d/%d' % (
        #     np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx)))




    def get_query_idx(self, net, unlabel_loader, n_select):
        net.eval()

        embedding_features = []
        predict_cls = []
        predict_outputs = []

        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in unlabel_loader:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = net(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                predict_outputs.append(y_hats)

                true_labels.append(ys)
                embedding_features.append(avg_feat)


        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()

        predict_outputs = torch.cat(predict_outputs, dim=0)
        predict_outputs = torch.softmax(predict_outputs, dim=1)
        predict_scores = torch.max(predict_outputs, dim=1)[0].cpu().numpy()
        predict_entropy = (- predict_outputs * torch.log2(predict_outputs)).sum(dim=1).cpu().numpy()

        predict_cls = torch.cat(predict_cls, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        distanc_matrix = pairwise_distances(embedding_features, self.mean_embedding, metric='l2')

        argsort = np.argsort(distanc_matrix, axis=1)
        argdist = np.take_along_axis(distanc_matrix, argsort, axis=1)
        metric_dist = argdist[:,0]
        metric_cls = np.argmin(distanc_matrix, axis=1)


        query_idx = np.where(predict_cls != metric_cls)

        unconsistency_idx = query_idx[0]
        unconsist_dist = np.take(metric_dist, unconsistency_idx)
        idx = np.argsort(unconsist_dist)[:n_select]

        aa = 2
        # if self.index_iteration > 4:
        #     aa = 2
        tmp_ind = np.argsort(metric_dist)[-aa*n_select:]
        query = kCenterGreedy(features=embedding_features[tmp_ind])
        select_ind = query.select_batch_(already_selected=[], N=n_select)


        # query_ind = np.argsort(metric_dist)[-n_select:]
        query_ind = self.unlabel_ind[tmp_ind][select_ind]



        metric_distance = np.min(distanc_matrix, axis=1)
        # target_distance = self.positive_margin

        score_thresh = 0.95
        # large_entropy_ind = np.where(predict_entropy > entropy_thresh)[0]
        # info = 'entropy %.2f large correct %d/%d %.4f' % (
        #     entropy_thresh,
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]), len(large_entropy_ind),
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]) / len(large_entropy_ind),
        # )
        # print(info)
        # self.logger.info(info)
        #
        # entropy_thresh = 0.90
        # large_entropy_ind = np.where(predict_entropy > entropy_thresh)[0]
        # info = 'entropy %.2f large correct %d/%d %.4f' % (
        #     entropy_thresh,
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]), len(large_entropy_ind),
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]) / len(large_entropy_ind),
        # )
        # print(info)
        # self.logger.info(info)
        #
        # entropy_thresh = 0.85
        # large_entropy_ind = np.where(predict_entropy > entropy_thresh)[0]
        # info = 'entropy %.2f large correct %d/%d %.4f' % (
        #     entropy_thresh,
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]), len(large_entropy_ind),
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]) / len(large_entropy_ind),
        # )
        # print(info)
        # self.logger.info(info)
        #
        # entropy_thresh = 0.80
        # large_entropy_ind = np.where(predict_entropy > entropy_thresh)[0]
        # info = 'entropy %.2f large correct %d/%d %.4f' % (
        #     entropy_thresh,
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]), len(large_entropy_ind),
        #     np.sum(predict_cls[large_entropy_ind] == true_labels[large_entropy_ind]) / len(large_entropy_ind),
        # )
        # print(info)
        # self.logger.info(info)


        quarter_distance = metric_distance.max() / 3
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = 'third one %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx), np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx])/len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx), np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx])/len(short_dist_idx))
        print(info)
        self.logger.info(info)

        quarter_distance = metric_distance.max() / 3 * 2
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = 'third two %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx), np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx])/len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx), np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx])/len(short_dist_idx))
        print(info)
        self.logger.info(info)
        #

        quarter_distance = metric_distance.max() / 4
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = 'quarter one %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx),
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]) / len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        half_distance = metric_distance.max() / 2
        large_dist_idx = np.where(metric_distance > half_distance)[0]
        short_dist_idx = np.where(metric_distance < half_distance)[0]
        info = 'half_distance large %.4f correct %d/%d %.4f short correct %d/%d %.4f' % (
            half_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx),
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]) / len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        quarter_distance = metric_distance.max() / 4 * 3
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = 'quarter three %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx),
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]) / len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        quarter_distance = metric_distance.max() / 5 * 1
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = '1/5 %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx),
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]) / len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        quarter_distance = metric_distance.max() / 5 * 2
        large_dist_idx = np.where(metric_distance > quarter_distance)[0]
        short_dist_idx = np.where(metric_distance < quarter_distance)[0]
        info = '2/5 %.4f large correct %d/%d %.4f short correct %d/%d %.4f' % (
            quarter_distance,
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]), len(large_dist_idx),
            np.sum(metric_cls[large_dist_idx] == true_labels[large_dist_idx]) / len(large_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        abs_distance1 = 0.05
        short_dist_idx = np.where(metric_distance < abs_distance1)[0]
        info = 'abs 0.05 short correct %d/%d' % (
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx))
        print(info)
        self.logger.info(info)

        abs_distance1 = 0.1
        short_dist_idx = np.where(metric_distance < abs_distance1)[0]
        info = 'abs 0.1 short correct %d/%d' % (
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx))
        print(info)
        self.logger.info(info)

        abs_distance1 = 0.2
        short_dist_idx = np.where(metric_distance < abs_distance1)[0]
        info = 'abs 0.2 short correct %d/%d' % (
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx))
        print(info)
        self.logger.info(info)

        abs_distance1 = 1.0
        short_dist_idx = np.where(metric_distance < abs_distance1)[0]
        info = 'abs 1.0 short correct %d/%d  %.4f' % (
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx])/len(short_dist_idx))
        print(info)
        self.logger.info(info)

        abs_distance1 = 1.3
        short_dist_idx = np.where(metric_distance < abs_distance1)[0]
        info = 'abs 1.3 short correct %d/%d  %.4f' % (
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]) / len(short_dist_idx))
        print(info)
        self.logger.info(info)

        short_dist_idx = np.where(metric_distance < self.positive_margin)[0]
        info = 'abs %.4f short correct %d/%d %.4f' % (self.positive_margin,
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx]), len(short_dist_idx),
            np.sum(metric_cls[short_dist_idx] == true_labels[short_dist_idx])/len(short_dist_idx))
        print(info)
        self.logger.info(info)
        #
        predict_correct_idx = np.where(predict_cls == true_labels)[0]
        metric_correct_idx = np.where(metric_cls == true_labels)[0]
        #
        consistency_idx = np.where(predict_cls == metric_cls)[0]
        consistency_correct_idx = np.where(predict_cls[consistency_idx] == true_labels[consistency_idx])[0]
        #
        unconsistency_predict_correct_idx = np.where(predict_cls[unconsistency_idx] == true_labels[unconsistency_idx])[0]
        unconsistency_metric_correct_idx = np.where(metric_cls[unconsistency_idx] == true_labels[unconsistency_idx])[0]
        #
        str = 'unlabel total number %d \n' \
              'metric correct number %d accuracy %.4f \n' \
              'consistency count %d consistency correct %d \n' \
              'unconsistency count %d predcit correct %d  metric correct %d '% \
              (len(embedding_features),
               len(metric_correct_idx), np.sum(true_labels == metric_cls) / len(embedding_features),
               len(consistency_idx), len(consistency_correct_idx),
               len(unconsistency_idx),len(unconsistency_predict_correct_idx),len(unconsistency_metric_correct_idx))
        self.logger.info(str)
        print(str)


        #np.random.shuffle(query_ind)

        return query_ind

    def lb_guessor(self, feat_u):
        with torch.no_grad():
            distanc_matrix = pairwise_distances(feat_u.cpu().numpy(), self.mean_embedding, metric='l2')
            lbs_u_guess = np.argmin(distanc_matrix, axis=1)
            lbs_u_guess = torch.tensor(lbs_u_guess).to(self.device)
            metric_distance = torch.tensor(np.min(distanc_matrix, axis=1)).to(self.device)
            # mask = 1. - torch.ge(1.).float()
            # half_distance = metric_distance.max() / 2
            mask = metric_distance < self.positive_margin
        return  mask, lbs_u_guess


        #     model.train()
        #     # print(ims_u_weak)
        #     logits, avg_feat = model(ims_u_weak)
        #     # print(self.mean_embedding)
        #     # print(avg_feat.cpu().numpy())
        #     distanc_matrix = pairwise_distances(avg_feat.cpu().numpy(), self.mean_embedding, metric='l2')
        #     lbs = np.argmin(distanc_matrix, axis=1)
        #     lbs = torch.tensor(lbs).to(self.device)
        #     metric_distance = torch.tensor(np.min(distanc_matrix, axis=1))
        #     half_distance = metric_distance.max() / 2
        #     idx = metric_distance < half_distance
        #     # probs = torch.softmax(logits, dim=1)
        #     # scores, lbs = torch.max(probs, dim=1)
        #     # idx = scores > self.thresh
        #     lbs = lbs[idx]
        # model.load_state_dict(org_state)
        #
        # if is_train:
        #     model.train()
        # else:
        #     model.eval()
        # return lbs.detach(), idx

    def lb_score_guess(self, logits):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(0.95).float()
        return mask, lbs_u_guess



    def plot_trained_data(self, models, train_dataset):
        subdataset = torch.utils.data.Subset(train_dataset, self.label_ind)

        unlabel_loader = DataLoader(dataset=subdataset,
                                    batch_size=self.config['test_batch_size'],
                                    shuffle=False,
                                    num_workers=4)

        models['backbone'].eval()
        models['module'].eval()

        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in unlabel_loader:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = models['backbone'](Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(embedding)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/train_%d.pdf' % self.index_iteration
        self.plot_classfication(embedding_features, self.mean_embedding, true_labels, str)

    def plot_unlabel_data(self, models, train_dataset):
        subdataset = torch.utils.data.Subset(train_dataset, self.unlabel_ind)

        unlabel_loader = DataLoader(dataset=subdataset,
                                    batch_size=self.config['test_batch_size'],
                                    shuffle=False,
                                    num_workers=4)

        models['backbone'].eval()
        models['module'].eval()

        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in unlabel_loader:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = models['backbone'](Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(embedding)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/unlabel_%d.pdf' % self.index_iteration
        self.plot_classfication(embedding_features, self.mean_embedding, true_labels, str)

    def plot_test_data(self, models, test_dataset):

        unlabel_loader = DataLoader(dataset=test_dataset,
                                    batch_size=self.config['test_batch_size'],
                                    shuffle=False,
                                    num_workers=4)

        models['backbone'].eval()
        models['module'].eval()

        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in unlabel_loader:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = models['backbone'](Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(embedding)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/test_%d.pdf' % self.index_iteration
        self.plot_classfication(embedding_features, self.mean_embedding, true_labels, str)



    def plot_classfication(self, embedding, anchors, labels, path):
        colors = ['dimgray', 'lightcoral', 'chocolate', 'yellow',
                  'olive', 'palegreen', 'teal', 'deepskyblue', 'fuchsia', 'blue']
        tsne = TSNE()
        combine_feat = np.concatenate([embedding, anchors])

        show_feat = tsne.fit_transform(combine_feat)

        feat = show_feat[:len(embedding)]
        anchor_feat = show_feat[len(embedding):]

        for c in range(self.n_cls):
            c_ind = np.where(labels == c)[0]
            plt.scatter(feat[c_ind][:,0], feat[c_ind][:,1], c=colors[c], alpha=0.5, s=1)

        for c in range(self.n_cls):
            plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c=colors[c], marker='o')
            plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c='black', marker='x')
        #plt.show()
        pp = PdfPages(path)
        plt.savefig(pp, format='pdf')
        pp.close()
        plt.close()







