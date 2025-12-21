import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import model as m
import simple_gui
import conf
import env
import MCTS
def draw_loss(records):
    # # [ {'train': tl, 'val': vl, 'iter':i}, ]
    # records = np.array(records)
    # plt.xlabel('iter')
    # plt.ylabel('loss')
    # x = [i['iter'] for i in records]
    # tl = [i['train'].item() for i in records]
    # vl = [i['val'].item() for i in records]
    # plt.plot(x, tl,label='train_loss')
    # plt.plot(x, vl,label='val_loss')
    # plt.legend()
    # plt.show()
    records = np.array(records)
    x = [i[0] for i in records]
    tl = [i[1] for i in records]
    plt.plot(x,tl, label='train_loss')
    plt.legend()
    plt.show()


def save_model(model, optimizer, iter, records, path, version):
    """保存模型检查点"""
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'records': records,
        'version': version
    }
    torch.save(checkpoint, path)
    print(f"模型已保存到 {path}")

def load_model(path, device='cpu'):
    """加载模型检查点"""
    checkpoint = torch.load(path, map_location=device)
    # 重新创建模型
    model = m.alphaZeroNet
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter = checkpoint['iter']
    records = checkpoint['records']
    version = checkpoint['version']
    print(path,' Loaded')
    return model, optimizer, iter, records, version

def arg_parse():
    parser = argparse.ArgumentParser(description='神经网络参数与操作')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--train', action='store_true')
    res = parser.parse_args()
    return res, res.resume or res.play or res.draw or res.train

def play_with_ai():
    conf.temperature = 0
    conf.epsilon = 0
    conf.num_searches = 200
    gui = simple_gui.GomokuGUI(conf.num_row, conf.num_col, env.gomoku, MCTS.mcts)
    gui.loop(gui.game.get_init_state(), -1)

