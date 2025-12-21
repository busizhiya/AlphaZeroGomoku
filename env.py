import numpy as np
import conf
import torch

class Gomoku:
    def __init__(self,num_row,num_col,win_k):
        self.num_row = num_row
        self.num_col = num_col
        self.win_k = win_k
        self.action_size = num_col * num_row
    def get_init_state(self):
        return np.zeros((self.num_row,self.num_col)).astype(np.int8)

    def get_next_state(self, state, action):
        row = action // self.num_col
        col = action % self.num_col
        if state[row][col] != 0:
            raise RuntimeError(f"state[{row}][{col}] is occupied")
        state[row][col] = 1
        return state
    
    def get_valid_moves(self, state):
        if len(state.shape) == 2: # (7, 7)
            return np.array(state.reshape(-1) == 0).astype(np.int8)
        # (B, 7, 7)
        return np.array(state.reshape(state.shape[0],-1) == 0).astype(np.int8)
    
    def check_win(self, state, action):
        '''Check whether the player who made action A win.
        state S is the state after action A'''
        row = action // self.num_col
        col = action % self.num_col
        player = state[row][col]
        B_r, B_c = self.num_row, self.num_col
        directions = [(0,1), (1,0), (1,1), (1,-1)]  # horiz, vert, diag, anti-diag
        for dr, dc in directions:
            cnt = 1
            # forward direction
            rr, cc = row + dr, col + dc
            while 0 <= rr < B_r and 0 <= cc < B_c and state[rr][cc] == player:
                cnt += 1
                rr += dr; cc += dc
            # backward direction
            rr, cc = row - dr, col - dc
            while 0 <= rr < B_r and 0 <= cc < B_c and state[rr][cc] == player:
                cnt += 1
                rr -= dr; cc -= dc
            if cnt >= self.win_k:
                return True
        return False

    def get_opponent_state(self, state):
        return -1 * state
    def is_terminal(self, state, action):
        '''
        return is_terminal, value(won=1, draw/continue=0)
        '''
        if action == -1: # for root node
            return False, 0
        won = self.check_win(state, action)
        draw = False
        if not won:
            # draw?
            draw = np.sum(self.get_valid_moves(state)) == 0
        return won or draw, 1 if won else 0
    def encode(self, state):
        # From np.array(7, 7) -> tensor(1, 3, 7, 7)
        if len(state.shape) == 2:
            return torch.tensor(np.array([(state==1).astype(np.int8),(state==-1).astype(np.int8), np.ones((self.num_row,self.num_col))]),device=conf.device,dtype=torch.float32).unsqueeze(0)
        return torch.tensor(np.array([(state==1).astype(np.float32),(state==-1).astype(np.float32), np.ones((state.shape[0],self.num_row,self.num_col),dtype=np.float32)]),device=conf.device,dtype=torch.float32).transpose(0,1)

gomoku = Gomoku(conf.num_row,conf.num_col,conf.win_k)
if __name__ == '__main__':
    s = gomoku.get_init_state()
    #print(gomoku.get_next_state(s0, 0))
    #print(gomoku.get_valid_moves(s0))
    player = 1
    while True:
        print(s)
        try:
            row,col = [int(i) for i in input("please input row & col:").split(' ')]
            if row < 1 or row > gomoku.num_row or col < 1 or col > gomoku.num_col:
                print("invalid moves...")
                continue
        except ValueError:
            print("please input numbers:  row  col")
            continue
        action = gomoku.num_col * (row-1) + col-1
        movable = gomoku.get_valid_moves(s)
        if not movable[action]:
            print("invalid moves...")
            continue
        s = gomoku.get_next_state(s, action)
        is_terminal, value = gomoku.is_terminal(s, action)
        
        if is_terminal:
            if value == 1:
                print(f"player {player} Won!")
                break
            else:
                print("Draw...")
                break
        player *= -1
        s = gomoku.get_opponent_state(s)