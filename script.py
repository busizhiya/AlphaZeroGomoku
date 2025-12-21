import os
import sys
import utils
import model as m
import conf
import numpy as np
import torch
from train import TrainManager
import env
import MCTS
import model
from datetime import datetime
device = conf.device

args, para = utils.arg_parse()
checkpoint_path = None
iter_start = 0
records = []

if not para:
    print('Usage:\n\t--resume <filename>\n\t--play\n\t--train\n\t--draw')
    sys.exit(1)
if args.resume:
    if os.path.exists(args.resume):
        model, optimizer, iter_start, records, version = utils.load_model(args.resume, device)
    else:
        print("请提供检查点路径: python script.py --resume path/to/checkpoint.pth")
        sys.exit(1)
else:
    print('init model & optimizer')
    model = m.alphaZeroNet
    model = model.to(device)
    version=0
    optimizer = torch.optim.AdamW(m.alphaZeroNet.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
print(sum([p.numel() for p in model.parameters()])/1e6,'M parameters')
print(f'device={device}')
if args.play:
    utils.play_with_ai()
    sys.exit(1)
if args.draw:
    utils.draw_loss(records)
    sys.exit(1)
if args.train:
    train_worker = TrainManager(game=env.gomoku,
                                model=model,
                                mcts=MCTS.mcts_parallel,
                                optimizer=optimizer,
                                T=conf.temperature,
                                loss_records=records,
                                version=version
                                )
    train_worker.learn(conf.num_learn_iters,
                        conf.num_parallel_games,
                        iter_start
                        )
    