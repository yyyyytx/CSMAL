import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
from torch.autograd.function import Function


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def dist_mat(x, y=None, eps=1e-16, squared=False):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j]
    is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    dist = torch.clamp(dist, 0.0, np.inf)
    if not squared:
        mask = (dist == 0).float()
        dist = dist + mask * eps
        dist = torch.sqrt(dist)
        dist = dist * (1.0 - mask)
    return dist

import threading
lock = threading.Lock()

class CenterSeperateMarginLoss(nn.Module):
    def __init__(self,
                 in_feats = 2,
                 n_classes = 10,
                 margin = 0.25,
                 distance = 1.,
                 initial_center = None,
                 device = torch.device('cuda:0')):
        super(CenterSeperateMarginLoss, self).__init__()

        self.device = device
        if initial_center is None:
            self.mean_feats = torch.zeros(n_classes, in_feats)
        else:
            self.mean_feats = torch.tensor(initial_center)
        self.old_mean_feats = None
        self.margin = margin
        self.distance = distance


        # self.ema_decay = 0.999
        self.ema_decay = 0.999
        self.ema_iteration = 0

        self.n_classes = n_classes
        self.in_feats = in_feats

    def forward(self, x, labels):
        # with lock:

        self.ema_mean(x.detach(), labels)
        self.ema_iteration += 1

        # delta = dist_mat(x, self.mean_feats.to(self.device).detach())
        delta = torch.cdist(x, self.mean_feats.to(self.device).detach())


        positive_mask = (torch.arange(self.n_classes).expand(len(x), -1).to(self.device) == labels.expand(self.n_classes, -1).transpose(0,1)).float()
        negative_mask = 1. - positive_mask


        ps = torch.clamp((delta - self.margin), min=0.) * positive_mask
        # ns = (delta - self.margin) * negative_mask
        # ns = torch.clamp(delta, min=1.) * negative_mask
        ns = torch.clamp((self.distance - delta), min=0.) * negative_mask
        # ns = delta * negative_mask


        ap = torch.clamp_min(ps.detach() + self.distance, min=0.) * positive_mask
        an = torch.clamp_min(ns.detach() + self.margin , min=0. ) * negative_mask

        # prevent divide zero
        loss_p = torch.sum(ap * ps) / (torch.sum(ps > 0.) + 1)
        loss_n = torch.sum(an * ns) / (torch.sum(ns > 0.) + 1)

        return torch.log(1 + loss_n + loss_p)


    def ema_mean(self, feats, labels):
        cls_features = {}
        mean_features = torch.zeros(self.n_classes, self.in_feats)

        for l in range(self.n_classes):
            cls_features[l] = []
        for idx in range(0, len(feats)):
            label = labels[idx]
            # print(label.item())
            # print(feats[idx])
            cls_features[label.item()].append(feats[idx])
        for l in range(self.n_classes):
            if len(cls_features[l]) != 0:
                cls_features[l] = torch.stack(cls_features[l])
                mean_features[l] = torch.mean(cls_features[l], axis=0)
            else:
                cls_features[l] = torch.zeros([1, self.in_feats])
                mean_features[l] = self.mean_feats[l]

        # ema mean feat
        alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)


        if self.old_mean_feats is None:
            self.mean_feats = mean_features
            self.old_mean_feats = mean_features
        else:
            self.mean_feats = self.old_mean_feats * alpha + (1 - alpha) * mean_features
            # self.old_mean_feats = mean_features
            self.old_mean_feats = self.mean_feats


class AblationCenterSeperateMarginLoss(nn.Module):
    def __init__(self,
                 in_feats = 2,
                 n_classes = 10,
                 margin = 0.25,
                 distance = 1.,
                 initial_center = None,
                 device = torch.device('cuda:0')):
        super(AblationCenterSeperateMarginLoss, self).__init__()

        self.device = device
        if initial_center is None:
            self.mean_feats = torch.zeros(n_classes, in_feats)
        else:
            self.mean_feats = torch.tensor(initial_center)
        self.old_mean_feats = None
        self.margin = margin
        self.distance = distance


        # self.ema_decay = 0.999
        self.ema_decay = 0.999
        self.ema_iteration = 0

        self.n_classes = n_classes
        self.in_feats = in_feats

    def forward(self, x, labels):
        # with lock:

        self.ema_mean(x.detach(), labels)
        self.ema_iteration += 1

        # delta = dist_mat(x, self.mean_feats.to(self.device).detach())
        delta = torch.cdist(x, self.mean_feats.to(self.device).detach())


        positive_mask = (torch.arange(self.n_classes).expand(len(x), -1).to(self.device) == labels.expand(self.n_classes, -1).transpose(0,1)).float()
        negative_mask = 1. - positive_mask


        ps = torch.clamp((delta - self.margin), min=0.) * positive_mask
        # ns = (delta - self.margin) * negative_mask
        # ns = torch.clamp(delta, min=1.) * negative_mask
        ns = torch.clamp((self.distance - delta), min=0.) * negative_mask
        # ns = delta * negative_mask


        ap = torch.clamp_min(ps.detach(), min=0.) * positive_mask
        an = torch.clamp_min(ns.detach() , min=0. ) * negative_mask

        if self.margin == 0.:
            ap = torch.tensor(0.).to(ap.device)

        if self.distance == 0.:
            an = torch.tensor(0.).to(an.device)



        # prevent divide zero
        loss_p = torch.sum(ap * ps) / (torch.sum(ps > 0.) + 1)
        loss_n = torch.sum(an * ns) / (torch.sum(ns > 0.) + 1)

        return torch.log(1 + loss_n + loss_p)


    def ema_mean(self, feats, labels):
        cls_features = {}
        mean_features = torch.zeros(self.n_classes, self.in_feats)

        for l in range(self.n_classes):
            cls_features[l] = []
        for idx in range(0, len(feats)):
            label = labels[idx]
            # print(label.item())
            # print(feats[idx])
            cls_features[label.item()].append(feats[idx])
        for l in range(self.n_classes):
            if len(cls_features[l]) != 0:
                cls_features[l] = torch.stack(cls_features[l])
                mean_features[l] = torch.mean(cls_features[l], axis=0)
            else:
                cls_features[l] = torch.zeros([1, self.in_feats])
                mean_features[l] = self.mean_feats[l]

        # ema mean feat
        alpha = min(1 - 1 / (self.ema_iteration + 1), self.ema_decay)


        if self.old_mean_feats is None:
            self.mean_feats = mean_features
            self.old_mean_feats = mean_features
        else:
            self.mean_feats = self.old_mean_feats * alpha + (1 - alpha) * mean_features
            # self.old_mean_feats = mean_features
            self.old_mean_feats = self.mean_feats

class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.3,
                 s=15):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        # if lb_view.is_cuda: lb_view = lb_view.cpu()
        lb_view = lb_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        # if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, lb)
        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        # print(feat.size(1), self.feat_dim)
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


################################################################
## Triplet related loss
################################################################
def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else (res + eps).sqrt() + eps

from torch.autograd import Variable

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=10):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_classes))

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max())  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min())  # mask[i]==0: negative samples of sample i
        # print(torch.tensor(dist_ap).shape)
        # dist_ap = torch.cat(dist_ap, dim=0)
        # dist_an = torch.cat(dist_an)
        dist_ap = torch.tensor(dist_ap)
        dist_an = torch.tensor(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  # normalize data by batch size
        return loss, prec