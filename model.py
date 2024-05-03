# coding: utf-8
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from util.util import *
import torch.nn.functional as F

from ipdb import set_trace as debug


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ImgFeature(nn.Module):
    def __init__(self):
        super(ImgFeature, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(307200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.model(x)
        # print(out.shape)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


img_feature = ImgFeature()


class Actor(nn.Module):
    def __init__(self, dim_states, dim_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        # self.conv = img_feature
        self.fc1 = nn.Linear(dim_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc2_1 = nn.Linear(hidden2, hidden2)
        self.fc3 = nn.Linear(hidden2, dim_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2_1.weight.data = fanin_init(self.fc2_1.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        # print(x.shape)
        # print(img.shape)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc2_1(out)
        out = self.relu(out)
        out = self.fc3(out)
        # print(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, dim_states, dim_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        # self.conv = img_feature
        self.fc1 = nn.Linear(dim_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + dim_actions, hidden2)
        self.fc2_1 = nn.Linear(hidden2, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc2_1.weight.data = fanin_init(self.fc2_1.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        x, a = xs

        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc2_1(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def print_para(net: nn.Module):
    for para in net.parameters():
        print(para)


def update_test():
    img = torch.ones((1, 4, 640, 480)).cuda()
    conv = ImgFeature().cuda()
    with torch.no_grad():
        out = conv(img)

    # print(out)
    a = (1, 2, 3, 4, 5, 6)
    a = torch.cat((to_tensor(np.array([a])), out), 1)
    # print(a)
    target = ()
    for num in range(16):
        target += (num * 10000000,)
    critic = Actor(16, 2)
    print('更新前')
    print_para(conv)
    print('------------------------------------')

    critic = critic.cuda()
    critic1_optim = Adam(critic.parameters(), lr=0.1)
    conv_optim = Adam(conv.parameters(), lr=0.1)
    # print(target)
    out = critic(a)
    target = critic(to_tensor(np.array([target])))
    criterion = nn.MSELoss()
    value_loss_1 = criterion(out, target)
    value_loss_1.backward()
    critic1_optim.step()
    conv_optim.step()
    # print(target)
    print('更新后')
    print_para(conv)


if __name__ == '__main__':
    # update_test()
    img = torch.ones((32, 3, 480, 640)).cuda()
    conv = ImgFeature().cuda()
    print(conv(img))
