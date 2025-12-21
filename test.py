import env
import MCTS
import model as m
import numpy as np
model = m.alphaZeroNet
game = env.gomoku
mcts = MCTS.mcts

root = MCTS.Node(game.get_init_state(), -1)
pi = mcts.search(root)
movable = game.get_valid_moves(root.state)
for action, child in root.children.items():
    print("action", action, "P(prior)=", child.P, "N=", child.N, "W/N=", (child.W / child.N if child.N>0 else None))
print("pi sum", pi.sum())