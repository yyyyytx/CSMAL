'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from active_utils.batchbald_redux import consistent_mc_dropout


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, (out4.shape[2], out4.shape[3]))

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_features(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_features, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.apply(self.init_weights)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, (out4.shape[2], out4.shape[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, [out1, out2, out3, out4]


    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

class ResNet_Bayesian(consistent_mc_dropout.BayesianModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear_drop = consistent_mc_dropout.ConsistentMCDropout()

        self.apply(self.init_weights)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def mc_forward_impl(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(self.layer3_drop(out2))
        out4 = self.layer4(self.layer4_drop(out3))
        out = F.avg_pool2d(out4, (out4.shape[2], out4.shape[3]))


        out = out.view(out.size(0), -1)
        out = self.linear_drop(out)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)


class ResNet_metric(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_metric, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, (out4.shape[2], out4.shape[3]))

        out_avg = out.view(out.size(0), -1)
        out = self.linear(out_avg)
        return out, out_avg

def cosine_distance_func(feat1, feat2):
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Cosine Distance
    distance = torch.matmul(F.normalize(feat1), F.normalize(feat2).t())
    return distance

class ResNet_ADS(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_metric, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)



        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, bias=False)

        self.classifier1 = nn.Sigmoid()
        self.classifier2 = nn.Sigmoid()

        self.protos1 = nn.Parameter(
            torch.randn(self.num_proto * self.num_classes, 512),
            requires_grad=True
        )
        self.protos2 = nn.Parameter(
            torch.randn(self.num_proto * self.num_classes, 512),
            requires_grad=True
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _compute_prob1(self, feat):
        cos_dist = cosine_distance_func(feat, self.protos1)
        cls_score, _ = cos_dist.view(-1, self.num_proto, self.num_classes).max(1)

        return self.classifier1(cls_score / self.alpha)

    def _compute_prob2(self, feat):
        cos_dist = cosine_distance_func(feat, self.protos2)
        cls_score, _ = cos_dist.view(-1, self.num_proto, self.num_classes).max(1)

        return self.classifier2(cls_score / self.alpha)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, (out4.shape[2], out4.shape[3]))

        out_avg = out.view(out.size(0), -1)
        out = self.linear(out_avg)
        return out, out_avg

import torchvision
torchvision.models.googlenet()

def ResNet18_features(num_classes = 10):
    return ResNet_features(BasicBlock, [2,2,2,2], num_classes)


def ResNet18_Bayesian(num_classes = 10):
    return ResNet_Bayesian(BasicBlock, [2,2,2,2], num_classes)

def ResNet18_metric(num_classes = 10):
    return ResNet_metric(BasicBlock, [2,2,2,2], num_classes)


def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

# class ResNet50_metric(ResNet_metric):
#     def __init__(self, num_classes=10):
#         super(ResNet50_metric, self).__init__(Bottleneck, [3,4,6,3], num_classes)

def ResNet50(num_classes = 10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet50_metric(num_classes = 10):
    return ResNet_metric(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes = 10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes = 10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

if __name__ == '__main__':
    net = ResNet50_metric()
    print(net)