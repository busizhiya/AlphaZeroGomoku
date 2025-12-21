import torch
import torch.nn as nn
import torch.nn.functional as F

import conf

class ResidualBlock(nn.Module):
    """标准的 ResNet 残差块
    Conv-BN-ReLU - Conv-BN + skip - ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AlphaZeroNet(nn.Module):
    """
    Conv_in - BN_in - ReLU - ResTower - Policy_head
                                      - Value_head
    Policy_head:
        Conv - BN - ReLU --(resize)-> FC
    Value_head:
        Conv - BN - ReLU --(resize)-> FC1 - ReLU - FC2 - tanh
    输入：
        state: (batch, 3, 7, 7)
    输出：
        policy_logits: (batch, 49)
        value: (batch, 1)
    """
    def __init__(self):
        super().__init__()
        self.num_row = conf.num_row
        self.num_col = conf.num_col
        self.action_size = conf.num_row * conf.num_col
        # ---------- 1. 输入层 ----------
        self.conv_in = nn.Conv2d(3,conf.channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(conf.channels)

        # ---------- 2. 残差塔 ----------
        self.res_towel = nn.Sequential(*[ResidualBlock(conf.channels) for _ in range(conf.num_res_blocks)])

        # ---------- 3. Policy head ----------
        self.policy_conv = nn.Conv2d(conf.channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*self.num_row * self.num_col, self.action_size)

        # ---------- 4. Value head ----------
        self.value_conv = nn.Conv2d(conf.channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.num_row * self.num_col, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # state: (batch, 3, 7, 7)
        x = F.relu(self.bn_in(self.conv_in(x))) # (B, 64, 7, 7)
        x = self.res_towel(x)   # (B, 64, 7, 7)

        p = F.relu(self.policy_bn(self.policy_conv(x))) # (B, 2, 7, 7)
        p = p.view(p.size(0), -1) # (B, 2*49)
        policy_logits = self.policy_fc(p) # (B, action_size)

        v = F.relu(self.value_bn(self.value_conv(x))) # (B, 1, 7, 7)
        v = v.view(v.size(0), -1) # (B, 49)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

alphaZeroNet = AlphaZeroNet()
if conf.load_model :
    print("load",conf.load_model)
    alphaZeroNet.load_state_dict(torch.load(conf.load_model))
alphaZeroNet.to(conf.device)
if __name__ == '__main__':
    state = torch.zeros(1, 3, 7, 7)

    p, v = alphaZeroNet(state)
    print(p.shape)
    print(v.shape)