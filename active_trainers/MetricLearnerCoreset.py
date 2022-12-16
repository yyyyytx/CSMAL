from active_trainers.BaseTrainer import BaseActiveTrainer
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler
import torch.optim as optim

from active_utils.ema import EMA
from active_utils.lr_scheduler import WarmupCosineLrScheduler
# from arch.wide_resnet import WideResnet



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
from data_utils import cifar

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





def mixup_data(x1, x2, y1, y2, alpha=1.0, use_cuda=True, device='cuda:0'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    # print(lam)
    # lam = max(lam, 1-lam)
    l = max(lam, 1-lam)
    # print(l)
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = l * x1 + (1 - l) * x2[index, :]
    y_a, y_b = y1, y2[index]
    return mixed_x, y_a, y_b, l


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MetricTrainer(BaseActiveTrainer):
    def __init__(self, config, device=torch.device('cuda:0'), logger=None, positive_margin=0.1,start_epoch=5, distance=1., gamma=2,lam=1.0):
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
        self.distance = distance
        self.pesudo_idx = None
        self.u_weight = 1.
        self.gamma = gamma
        self.lam = lam


    # def build_model(self, base_net):
    def start_loop(self, net, train_dataset, test_dataset, unlabel_dataset, n_select, n_iteration, index_loop=1, is_semi=False, is_semi_metric=False, pesudo_count=1000, semi_ind=None, is_mixup=True, is_rubost=False):
        method_str = '====================Metric Method with pesudo label==============='
        self.logger.info(method_str)
        self.loop_num = index_loop
        self.pesudo_count=pesudo_count
        total_train_ind = np.arange(50000)
        self.semi_ind = semi_ind
        self.label_ind = np.random.permutation(total_train_ind)[:n_select]
        self.unlabel_ind = np.setdiff1d(total_train_ind, self.label_ind)
        self.n_cls = int(self.config['n_cls'])
        self.n_select = n_select
        self.is_semi_metric = is_semi_metric
        self.is_mixup = is_mixup
        self.is_rubost = is_rubost

        print(self.n_cls)

        self.ema_iter = 0


        self.best_acc = 0.
        self.best_meric_acc = 0.

        # # self.lb_guessor = LabelGuessor(thresh=0.95)
        # if is_semi is True:
        # #     model = WideResnet(n_classes=10,
        # #                        k=2,
        # #                        n=28)
        #     self.in_feats = 128
        # else:
        # #     model = ResNet18_metric(num_classes=10).to(self.device)
        # #     model.apply(self.init_weights)
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
            dlval = cifar.get_val_loader(dataset=self.config['ds_name'], batch_size=64, num_workers=2)
            print(len(dlval.dataset))

            dl_x, dl_u = cifar.get_sub_train_loader(dataset=self.config['ds_name'],
                                                    batch_size=self.config['train_batch_size'],
                                                    ind_x=self.label_ind,
                                                    ind_u=self.unlabel_ind)

            if is_semi is True:
                # model = WideResnet(n_classes=self.config['n_cls'],
                #                    k=2,
                #                    n=28)
                # self.in_feats = 128
                # model.train()
                # model.to(self.device)
                # ema = EMA(model, 0.999)
                # self.train_semi_data(model, ema)
                print('1111')

            else:
                model = ResNet18_metric(num_classes=self.config['n_cls']).to(self.device)
                model.apply(self.init_weights)
                self.in_feats = 512
                model.train()
                model.to(self.device)
                import time
                aa = time.time()
                self.train_pesudo_data(model)
                end = time.time() - aa
                str = 'train time %.4f' % end
                self.logger.info(str)


            self.pesudo_idx, self.pesudo_label = self.get_pesudo_idx(model, dl_u, self.positive_margin)

            acc, metric_acc = self.model_accuracy(model)



            acc_info = '[Iteration]epoch %d Accuracy %.4f metric acc %.4f' % (i, acc, metric_acc)
            print(acc_info)
            self.logger.info(acc_info)
            if is_semi is True:
                break
            import time
            aa = time.time()
            query_ind, self.plot_ind = self.get_query_idx(model, dl_u, n_select)
            end = time.time() - aa
            str = 'select time %.4f' % end
            self.logger.info(str)


            self.label_ind = np.append(self.label_ind, query_ind)
            self.label_ind = np.unique(self.label_ind)
            self.label_ind = np.random.permutation(self.label_ind)
            self.unlabel_ind = np.setdiff1d(total_train_ind, self.label_ind)


    def train_pesudo_data(self, model):


        dl_x, dl_g = cifar.get_combine_train_loader(self.config['ds_name'], self.config['train_batch_size'], self.label_ind, self.pesudo_idx, self.pesudo_label)
        # dl_x, dl_g = cifar.get_guess_train_loader('CIFAR10', self.config['train_batch_size'], self.label_ind, self.pesudo_idx, self.pesudo_label)

        optim_backbone = optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'],
                                   weight_decay=self.config['weight_decay'])
        sched_backbone = torch.optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=self.config['milestones'])
        # sched_backbone = WarmupCosineLrScheduler(optim_backbone, max_iter=np.inf, warmup_iter=0)

        criterion_backbone = nn.CrossEntropyLoss().to(self.device)
        criterion_metric = CenterSeperateMarginLoss(in_feats=self.in_feats, n_classes=int(self.config['n_cls']),
                                                    margin=self.positive_margin,
                                                    distance=self.distance,
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

            # if (epoch + 1)%5 == 0:
            #     self.plot_trained_data(model, epoch)
        # self.plot_trained_data(model, epoch)


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
                Xs, _, ys = labeled_train_iter.next()
                Xs, ys = Xs.to(self.device), ys.to(self.device)
            except:
                labeled_train_iter = iter(dl_x)
                Xs, _, ys = labeled_train_iter.next()
                Xs, ys = Xs.to(self.device), ys.to(self.device)

            try:
                Xs_u, _, ys_u = combine_train_iter.next()
                Xs_u, ys_u = Xs_u.to(self.device), ys_u.to(self.device)
            except:
                combine_train_iter = iter(dl_g)
                Xs_u, _, ys_u = combine_train_iter.next()
                Xs_u, ys_u = Xs_u.to(self.device), ys_u.to(self.device)



            # mixup
            # if self.is_mixup is True:
            #     inputs, targets_a, targets_b, lam = mixup_data(Xs, Xs_u, ys, ys_u, alpha=0.75)
            #     y_uhats, _ = model(inputs)
            #     cls_loss = mixup_criterion(criterions['cls'], y_uhats, targets_a, targets_b, lam)
            # else:
            # #non mixup
            #     y_uhats, _ = model(Xs_u)
            #     cls_loss = criterions['cls'](y_uhats, ys_u)


            y_hats, feat_x = model(Xs)
            # y_hats_u, feat_u = model(Xs_u)
            cls_loss = criterions['cls'](y_hats, ys)

            # ---------   metric loss   ---------------
            metric_loss = criterions['metric'](feat_x, ys)
            # metric_loss = criterions['metric'](feat_u, ys_u)


            total_cls_loss += cls_loss
            total_metric_loss += metric_loss
            loss = cls_loss + self.lam * metric_loss
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
        str = '[LOSS] epoch %d loss %.4f cls_loss %.4f metric_loss %.4f)' % (
            epoch, total_loss / batch_count, total_cls_loss / batch_count, total_metric_loss / batch_count)
        print(str)
        # logger.info(str)




    def model_accuracy(self, net):
        net.eval()
        correct_num = 0
        dlval = cifar.get_val_loader(dataset=self.config['ds_name'], batch_size=64, num_workers=2)


        embedding_features = []
        true_labels = []
        for Xs, ys in dlval:
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
        predict_pro = []

        with torch.no_grad():
            for (Xs, ys) in dl_u:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = net(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_pro.append(y_hats)

                predict_cls.append(preds)
                true_labels.append(ys)
                embedding_features.append(avg_feat)


        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()

        predict_pro = torch.cat(predict_pro, dim=0)
        predict_pro = torch.softmax(predict_pro, dim=1)
        entropy = (- predict_pro * torch.log2(predict_pro)).sum(dim=1).cpu().numpy()
        predict_cls = torch.cat(predict_cls, dim=0).cpu().numpy()

        ent_sort1 = np.argsort(entropy)[-self.pesudo_count:]

        str = 'entropy 1 pesudo count %d correct %d %.4f' % (
        len(ent_sort1), np.sum(true_labels[ent_sort1] == predict_cls[ent_sort1]),
        len(ent_sort1) / np.sum(true_labels[ent_sort1] == predict_cls[ent_sort1]))
        print(str)
        self.logger.info(str)

        ent_sort2 = np.argsort(entropy)[:self.pesudo_count]

        str = 'entropy 2 pesudo count %d correct %d %.4f' % (
            len(ent_sort2), np.sum(true_labels[ent_sort2] == predict_cls[ent_sort2]),
            len(ent_sort2) / np.sum(true_labels[ent_sort2] == predict_cls[ent_sort2]))
        print(str)
        self.logger.info(str)


        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        distanc_matrix = pairwise_distances(embedding_features, self.mean_embedding, metric='l2')

        argsort = np.argsort(distanc_matrix, axis=1)
        argdist = np.take_along_axis(distanc_matrix, argsort, axis=1)
        metric_cls = np.argmin(distanc_matrix, axis=1)

        metric_distance = np.min(distanc_matrix, axis=1)
        max_distance = metric_distance.max()

        # t = (self.index_iteration + 1) * 0.1
        # if t > self.positive_margin:
        #     t = self.positive_margin
        pesudo_idx = np.where(metric_distance < max_distance)[0]

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

        min_dis_ind = np.argsort(metric_dist)[:1000]
        max_dis_ind = np.argsort(metric_dist)[-1000:]

        min_dis_entropy = np.sum(predict_entropy[min_dis_ind]) / 1000
        min_dis_score = np.sum(predict_scores[min_dis_ind]) / 1000
        min_dis_acc = np.sum(predict_cls[min_dis_ind] == true_labels[min_dis_ind]) / 1000

        max_dis_entropy = np.sum(predict_entropy[max_dis_ind]) / 1000
        max_dis_score = np.sum(predict_scores[max_dis_ind]) / 1000
        max_dis_acc = np.sum(predict_cls[max_dis_ind] == true_labels[max_dis_ind]) / 1000

        str = 'min 1000 avg entropy %.4f score %.4f acc %.4f' % (min_dis_entropy, min_dis_score, min_dis_acc)
        print(str)
        self.logger.info(str)
        str = 'max 1000 avg entropy %.4f score %.4f acc %.4f' % (max_dis_entropy, max_dis_score, max_dis_acc)
        print(str)
        self.logger.info(str)
        query_idx = np.where(predict_cls != metric_cls)

        unconsistency_idx = query_idx[0]
        unconsist_dist = np.take(metric_dist, unconsistency_idx)
        idx = np.argsort(unconsist_dist)[:n_select]

        aa = 3
        if self.index_iteration > 4:
            aa = 2
        # aa = self.gamma
        tmp_ind = np.argsort(metric_dist)[-aa*n_select:]
        query = kCenterGreedy(features=embedding_features[tmp_ind])
        select_ind = query.select_batch_(already_selected=[], N=n_select)
        query_ind = self.unlabel_ind[tmp_ind][select_ind]


        # query_ind = np.argsort(metric_dist)[-n_select:]



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

        return query_ind, tmp_ind[select_ind]

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

    def plot_total(self, model, dl_x, dl_u):
        from shapely.geometry import Point
        from shapely.ops import cascaded_union
        import matplotlib.patches as ptc
        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in dl_u:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = model(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                # embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(avg_feat)

            for (Xs, _, ys) in dl_x:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = model(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                # embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(avg_feat)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/total_metric_%d.pdf' % (self.index_iteration)
        str1 = './plot/total_metric_%d.svg' % (self.index_iteration)

        combine_feat = np.concatenate([embedding_features, self.mean_embedding])

        colors = ['lightcoral', 'skyblue', 'sandybrown', 'antiquewhite', 'khaki', 'lightpink', 'mediumseagreen',
                  'turquoise', 'skyblue', 'mediumpurple']
        label_colors = ['brown', 'deepskyblue', 'saddlebrown', 'tan', 'darkkhaki', 'hotpink', 'seagreen',
                        'lightseagreen', 'steelblue', 'darkviolet']

        tsne = TSNE()


        feat_path = '/home/ytx/ytx/metric_active/cifar/plot/plot_features_meric_%d.npy' % self.index_iteration
        label_path = '/home/ytx/ytx/metric_active/cifar/plot/plot_labels_meric_%d.npy' % self.index_iteration
        quey_path = '/home/ytx/ytx/metric_active/cifar/plot/plot_query_meric_%d.npy' % self.index_iteration

        import os
        if os.path.exists(feat_path):
            print('load feats')
            show_feat = np.load(feat_path)
        else:
            show_feat = tsne.fit_transform(combine_feat)
            np.save(feat_path, show_feat)

        if os.path.exists(label_path):
            true_labels = np.load(label_path)
        else:
            np.save(label_path, true_labels)

        if os.path.exists(quey_path):
            self.plot_ind = np.load(quey_path)
        else:
            np.save(quey_path, self.plot_ind)
        print(show_feat.shape)
        feat_u = show_feat[:len(dl_u.dataset)]
        labels_u  = true_labels[:len(dl_u.dataset)]
        feat_l = show_feat[len(dl_u.dataset): (len(dl_u.dataset)+len(dl_x.dataset))]
        labels_l = true_labels[len(dl_u.dataset): (len(dl_u.dataset)+len(dl_x.dataset))]
        anchor_feat = show_feat[(len(dl_u.dataset)+len(dl_x.dataset)):]
        # feat = show_feat[:len(embedding)]
        # anchor_feat = show_feat[len(embedding):]

        alpha = 0.5
        size = 0.1

        import plotly.graph_objects as go
        fig = go.Figure()

        for c in range(self.n_cls):
            c_ind = np.where(labels_u == c)[0]
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=feat_u[c_ind][:, 0],
                    y=feat_u[c_ind][:, 1],
                    opacity=0.3,
                    marker=dict(
                        color=colors[c],
                        size=1,
                    )
                )
            )

        for c in range(self.n_cls):
            c_ind = np.where(labels_l == c)[0]
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    marker_symbol='triangle-up',
                    x=feat_l[c_ind][:, 0],
                    y=feat_l[c_ind][:, 1],
                    opacity=1.0,
                    marker=dict(
                        color=label_colors[c],
                        size=2,
                    )
                )
            )

        for c in range(self.n_cls):
            query_feat = feat_u[self.plot_ind]
            query_label = labels_u[self.plot_ind]
            c_ind = np.where(query_label == c)[0]
            #     plt.scatter(query_feat[c_ind][:, 0], query_feat[c_ind][:, 1], c=colors[c], alpha=1, s=2, marker='+')
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    marker_symbol='square',
                    x=query_feat[c_ind][:, 0],
                    y=query_feat[c_ind][:, 1],
                    opacity=1.0,
                    marker=dict(
                        color=colors[c],
                        size=2,
                    )
                )
            )

        # for c in range(self.n_cls):
        #     print(anchor_feat[c][0], anchor_feat[c][1])
        fig.add_trace(
            go.Scatter(
                mode='markers',
                marker_symbol='x',
                x=anchor_feat[:,0],
                y=anchor_feat[:,1],
                opacity=1.0,
                marker=dict(
                    color='black',
                    size=4,
                )
            )
        )
        # #     plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c=colors[c], marker='o')
        #     ax.scatter(anchor_feat[c][0], anchor_feat[c][1], c='black', marker='x')
        #     ax.scatter(query_feat[c_ind][:, 0], query_feat[c_ind][:, 1], c=colors[c], alpha=1, s=2, marker='+')

        # fig.show()
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.write_image(str)
        fig.write_image(str1)
        # fig = plt.figure(figsize=(5,5))
        # ax = fig.add_subplot(111, title='222')
        # for c in range(self.n_cls):
        #     c_ind = np.where(labels_u == c)[0]
        #     f_c = feat_u[c_ind]
        #     polygons = [Point(f_c[i, 0], f_c[i, 1]).buffer(size) for i in range(len(f_c))]
        #     polygons = cascaded_union(polygons)
        #     for poly in polygons:
        #         poly = ptc.Polygon(np.array(poly.exterior), facecolor=colors[c], lw=0, alpha=0.5)
        #         ax.add_patch(poly)
        #     # plt.scatter(feat_u[c_ind][:, 0], feat_u[c_ind][:, 1], c=colors[c], alpha=1, s=1, marker='o', lw=0, edgecolors='none')
        #
        #
        # for c in range(self.n_cls):
        #     c_ind = np.where(labels_l == c)[0]
        #     # f_c = feat_l[c_ind]
        #     # polygons = [Point(f_c[i, 0], f_c[i, 1]).buffer(size) for i in range(len(f_c))]
        #     # polygons = cascaded_union(polygons)
        #     # for poly in polygons:
        #     #     poly = ptc.Polygon(np.array(poly.exterior), facecolor=label_colors[c], lw=0, alpha=0.5, marker='^')
        #     #     ax.add_patch(poly)
        #     # ax.scatter
        #     ax.scatter(feat_l[c_ind][:, 0], feat_l[c_ind][:, 1], c=label_colors[c], alpha=1, s=2, marker='^', lw=0, edgecolors='none')
        #
        # for c in range(self.n_cls):
        #     query_feat = feat_u[self.plot_ind]
        #     query_label = labels_u[self.plot_ind]
        #     c_ind = np.where(query_label == c)[0]
        #     ax.scatter(query_feat[c_ind][:, 0], query_feat[c_ind][:, 1], c=colors[c], alpha=1, s=2, marker='+')
        #
        # for c in range(self.n_cls):
        #     #     plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c=colors[c], marker='o')
        #     ax.scatter(anchor_feat[c][0], anchor_feat[c][1], c='black', marker='x')
        #     # f_c = feat_l[c_ind]
        #     # polygons = [Point(anchor_feat[i, 0], anchor_feat[i, 1]).buffer(size) for i in range(len(anchor_feat))]
        #     # polygons = cascaded_union(polygons)
        #     # for poly in polygons:
        #     #     poly = ptc.Polygon(np.array(poly.exterior), facecolor='black', lw=0, alpha=1.0, marker='x')
        #     #     ax.add_patch(poly)
        # # plt.show()
        # plt.savefig(str)
        # plt.close()
        # pp = PdfPages(str)
        # fig.savefig(pp, format='pdf')
        # pp.close()
        # plt.close()

    def plot_trained_data(self, model, epoch):
        from data_utils.cifar import load_sub_data_train, Cifar
        data_x, label_x, data_u, label_u = load_sub_data_train(inds_x=self.label_ind,
                                                               inds_u=self.unlabel_ind,
                                                               dataset='CIFAR10')
        ds_x = Cifar(dataset='CIFAR10', data=data_x, labels=label_x, is_train=True)

        # subdataset = torch.utils.data.Subset(dx, self.label_ind)


        unlabel_loader = DataLoader(dataset=ds_x,
                                    batch_size=self.config['test_batch_size'],
                                    shuffle=False,
                                    num_workers=4)

        model.eval()
        print('plot data count : %d' % len(unlabel_loader.dataset))
        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs,_, ys) in unlabel_loader:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = model(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                # embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(avg_feat)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/metric_train_%d_%d_new.pdf' % (epoch, self.index_iteration)
        self.plot_classfication(embedding_features, self.mean_embedding, true_labels, str)

    def plot_unlabel_data(self, model, dl, epoch):
        model.eval()
        print('plot data count : %d' % len(dl.dataset))
        embedding_features = []
        predict_cls = []
        true_labels = []

        with torch.no_grad():
            for (Xs, ys) in dl:
                Xs = Xs.to(self.device)
                ys = ys.to(self.device)

                y_hats, avg_feat = model(Xs)
                _, preds = torch.max(y_hats, 1)

                predict_cls.append(preds)
                true_labels.append(ys)
                # embedding = models['module'](avg_feat)  # pred_loss = criterion(scores, labels) # ground truth loss
                embedding_features.append(avg_feat)

        embedding_features = torch.cat(embedding_features, dim=0).cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
        str = './plot/metric_unlabel_%d.pdf' % epoch
        self.plot_classfication(embedding_features, self.mean_embedding, true_labels, str)


    def plot_test_data(self, models, test_dataset):
        from data_utils.cifar import load_sub_data_train
        data_x, label_x, data_u, label_u = load_sub_data_train(inds_x=self.label_ind,
                                                               inds_u=self.unlabel_ind,
                                                               dataset='CIFAR10')
        unlabel_loader = DataLoader(dataset=data_x,
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
        # colors = ['dimgray', 'lightcoral', 'chocolate', 'yellow',
        #           'olive', 'palegreen', 'teal', 'deepskyblue', 'fuchsia', 'blue']
        # label_colors =['dimgray', 'lightcoral', 'chocolate', 'yellow',
        #           'olive', 'palegreen', 'teal', 'deepskyblue', 'fuchsia', 'blue']
        colors = ['brown', 'deepskyblue', 'saddlebrown', 'tan', 'darkkhaki', 'hotpink', 'seagreen',
                        'lightseagreen', 'steelblue', 'darkviolet']

        tsne = TSNE(init='pca')
        combine_feat = np.concatenate([embedding, anchors])

        show_feat = tsne.fit_transform(combine_feat)

        feat = show_feat[:len(embedding)]
        anchor_feat = show_feat[len(embedding):]

        for c in range(self.n_cls):
            c_ind = np.where(labels == c)[0]
            plt.scatter(feat[c_ind][:,0], feat[c_ind][:,1], c=colors[c], alpha=0.5, s=1)

        for c in range(self.n_cls):
        #     plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c=colors[c], marker='o')
            plt.scatter(anchor_feat[c][0], anchor_feat[c][1], c='black', marker='x')
        #plt.show()
        pp = PdfPages(path)
        plt.savefig(pp, format='pdf')
        pp.close()
        plt.close()







