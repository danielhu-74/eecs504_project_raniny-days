import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = (
            torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_right = (
            torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_up = (
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_down = (
            torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda()
            + 10000
            * torch.min(
                org_pool - torch.FloatTensor([0.3]).cuda(),
                torch.FloatTensor([0]).cuda(),
            ),
            torch.FloatTensor([0.5]).cuda(),
        )
        E_1 = torch.mul(
            torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()),
            enhance_pool - org_pool,
        )

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):

        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, : h_x - 1, :])).mean()
        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, : w_x - 1])).mean()
        return self.TVLoss_weight * (h_tv + w_tv)


class L_contrast(nn.Module):
    def __init__(self):
        super(L_contrast, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        contrast = torch.sqrt(torch.mean(torch.pow(x - x_mean, 2), dim=[2, 3]))
        loss = torch.mean(torch.pow(contrast - 0.2, 2))
        return loss


class L_struct(nn.Module):
    def __init__(self):
        super(L_struct, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, org, enhance):
        org_mean = self.pool(torch.mean(org, 1, keepdim=True))
        enhance_mean = self.pool(torch.mean(enhance, 1, keepdim=True))

        loss = 1 - torch.mean(torch.cos(org_mean - enhance_mean))
        return loss


class Sa_Loss(nn.Module):
    def __init__(self):
        super(L_struct, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, org, enhance):
        # Compare the mean of the original and enhanced images to approximate the correlation
        org_mean = self.pool(torch.mean(org, 1, keepdim=True))
        enhance_mean = self.pool(torch.mean(enhance, 1, keepdim=True))

        loss = 1 - torch.mean(torch.cos(org_mean - enhance_mean))
        return loss


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
