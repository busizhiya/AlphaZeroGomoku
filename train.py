import numpy as np
import torch
import model as m
import conf
import MCTS
import env
import random
import torch.nn.functional as F
from tqdm import trange, tqdm
import utils
from datetime import datetime
import copy
import gc
from profiler import profiler
device = conf.device
profiler.set_enabled(conf.enable_profiling)



class SPG:
    def __init__(self, game):
        self.memory = []
        # root是当前真实对局的信息
        self.root = MCTS.Node(game.get_init_state(),-1)
        # node是mcts中存储的模拟搜索信息
        self.node = self.root

class TrainManager:
    def __init__(self, game: env.Gomoku, model, mcts, optimizer, T, loss_records=[], version=0):
        self.game = game
        self.model = model
        self.mcts = mcts
        self.optimizer = optimizer
        self.batch_size = conf.batch_size
        self.T = T
        self.iter_T = T
        self.loss_records = loss_records
        self.best_model = copy.deepcopy(model)
        self.version = version
        self.data_augmentation_enabled = True

    def selfplay_parallel(self, num_selfplay_rounds, num_parallel_games):
        memoryReturn = [] 
        round_bar = trange(num_selfplay_rounds,desc="", leave=False, position=1)
        for round in round_bar:
            T = self.iter_T
            spGames =[SPG(self.game) for _ in range(num_parallel_games) ]
            round_bar.set_description(f"Round {round}/{num_selfplay_rounds}, spg={len(spGames)}")
            step = 0
            while len(spGames) > 0:
                step += 1
                with profiler.section("selfplay_parallel/mcts"):
                    pis = self.mcts.search(spGames)
                movables = self.game.get_valid_moves(np.array([spg.root.state for spg in spGames]))
                if step >= conf.annealing_steps:
                    T = 0
                for i in range(len(spGames))[::-1]:
                    round_bar.set_description(f"Round {round}/{num_selfplay_rounds}, spg_{len(spGames)}/s_{step}")
                    spg, pi, movable = spGames[i], pis[i], movables[i]
                    pi[movable==0] = 0
                    pi /= np.sum(pi)
                    spg.memory.append([spg.root.state, pi])
                    if T == 0:
                        action = np.argmax(pi)
                        #print(f"action={action}")
                    else:
                        pi = pi ** 1/T
                        pi /= np.sum(pi)
                        action = np.random.choice(self.game.action_size,p=pi)
                    new_node = spg.root.get_child(action)
                    spg.root.clear_child()
                    new_node.clear_parent()
                    is_terminal, value = self.game.is_terminal(new_node.state, action)
                    spg.root = new_node
                    if is_terminal:
                        for sample in reversed(spg.memory):
                            state, pi, value = sample[0], sample[1], value
                            memoryReturn.append((sample[0],sample[1],value))
                            if self.data_augmentation_enabled:
                                # 使用全部 8 种对称变换，提升稳定性与泛化
                                pi_matrix = pi.reshape(conf.num_row, conf.num_col)
                                transforms = [
                                    (state, pi_matrix),  # 原始
                                    (np.rot90(state, 1), np.rot90(pi_matrix, 1)),  # 旋转90度
                                    (np.rot90(state, 2), np.rot90(pi_matrix, 2)),  # 旋转180度
                                    (np.rot90(state, 3), np.rot90(pi_matrix, 3)),  # 旋转270度
                                    (np.fliplr(state), np.fliplr(pi_matrix)),  # 水平翻转
                                    (np.flipud(state), np.flipud(pi_matrix)),  # 垂直翻转
                                    (np.transpose(state), np.transpose(pi_matrix)),  # 转置
                                    (np.fliplr(np.rot90(state, 1)), np.fliplr(np.rot90(pi_matrix, 1))),  # 旋转90度+水平翻转
                                ]
                                for aug_state, aug_pi_matrix in transforms:
                                    memoryReturn.append(
                                        (aug_state.copy(), aug_pi_matrix.flatten(), value)
                                    )
                            value *= -1
                        del spGames[i]
        round_bar.close()
        return memoryReturn
    def selfplay(self, num_selfplay_rounds):
        # (s, π, z)
        memoryReturn = []
        self.best_model.eval()
        for _ in trange(num_selfplay_rounds):
            memory = []
            root = MCTS.Node(self.game.get_init_state(), -1)
            T = self.iter_T
            while True:
                with profiler.section("selfplay/mcts"):
                    pi = self.mcts.search(root)
                movables = self.game.get_valid_moves(root.state)
                pi[movables==0] = 0
                pi /= np.sum(pi)
                memory.append([root.state, pi])
                if T == 0:
                    action = np.argmax(pi)
                    #print(f"action={action}")
                else:
                    pi = pi ** 1/T
                    pi /= np.sum(pi)
                    action = np.random.choice(self.game.action_size,p=pi)
                new_node = root.get_child(action)
                root.clear_child()
                new_node.clear_parent()
                
                is_terminal, value = self.game.is_terminal(new_node.state, action)
                root = new_node
                if is_terminal:
                    for sample in reversed(memory):
                        memoryReturn.append((sample[0],sample[1],value))
                        value *= -1
        return memoryReturn
    def annealing(self,iter,num_learn_iters):
        ratio = iter / num_learn_iters
        self.iter_T = self.T * max(0, 1 - ratio)
        self.mcts.set_epsilon(conf.epsilon * max(0, 1 - ratio))
    def train(self, data):
        # sample :(s, pi, z)
        random.shuffle(data)
        self.model.train()
        total_loss = 0
        #print(data)
        train_bar = tqdm(range(0,len(data), self.batch_size),leave=False,position=1)
        for i in train_bar:
            train_bar.set_description(f"Train: [{i}/{len(data)}]")
            batch = data[i:min(i+self.batch_size, len(data))]
            states = np.array([sample[0] for sample in batch])
            pi = torch.tensor(np.array([sample[1] for sample in batch]),device=conf.device)
            z = torch.tensor([sample[2] for sample in batch],device=conf.device,dtype=torch.float).reshape(-1, 1)
            with profiler.section("train/nn"):
                P, v = self.model(self.game.encode(states))
            logp = F.log_softmax(P, dim=1)
            policy_loss = -(pi * logp).sum(dim=1).mean()
            loss = policy_loss + F.mse_loss(v, z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    
            total_loss += loss.item()        
        return total_loss / ((len(data)-1) // self.batch_size + 1)
    # def eval(self):
    #     num_eval_rounds = 1
    #     # self.model VS self.best_model
    #     self.model.eval()
    #     self.best_model.eval()
    #     best_mcts = MCTS.MCTS_Parallel(conf.num_eval_searches,num_eval_rounds,self.best_model,epsilon=0)
    #     cur_mcts = MCTS.MCTS_Parallel(conf.num_eval_searches,num_eval_rounds,self.model,epsilon=0)
    #     cmp_mcts = [cur_mcts,best_mcts]
    #     round_bar = trange(2,desc="", leave=False, position=1)
    #     win_cnt = [0,0]
    #     for round in round_bar:
    #         T = 0
    #         spGames =[SPG(self.game) for _ in range(num_eval_rounds) ]
    #         round_bar.set_description(f"[Eval] {round}/2 games={len(spGames)}")
    #         step = 0
    #         while len(spGames) > 0:
    #             step += 1
    #             side = (round+step)%2
    #             pis = cmp_mcts[side].search(spGames)
    #             movables = self.game.get_valid_moves(np.array([spg.root.state for spg in spGames]))
    #             if step == conf.annealing_steps:
    #                 T = 0
    #             for i in range(len(spGames))[::-1]:
    #                 rate = 0.5 if (win_cnt[0] + win_cnt[1]==0) else win_cnt[0] / (win_cnt[0] + win_cnt[1])
    #                 round_bar.set_description(f"[Eval] {round}/2 games_{len(spGames)}/s_{step} rate={rate:.2f}")
    #                 spg, pi, movable = spGames[i], pis[i], movables[i]
    #                 pi[movable==0] = 0
    #                 pi /= np.sum(pi)
    #                 if T == 0:
    #                     action = np.argmax(pi)
    #                     #print(f"action={action}")
    #                 else:
    #                     pi = pi ** 1/T
    #                     pi /= np.sum(pi)
    #                     action = np.random.choice(self.game.action_size,p=pi)
    #                 new_node = spg.root.get_child(action)
    #                 #spg.memory.append([spg.root.state, pi])
    #                 is_terminal, value = self.game.is_terminal(new_node.state, action)
    #                 spg.root = new_node
    #                 if is_terminal:
    #                     if value==1:
    #                         win_cnt[side]+=1
    #                     del spGames[i]  
    #     round_bar.close()
    #     return 1 if (win_cnt[0] + win_cnt[1])==0 else win_cnt[0] / (win_cnt[0] + win_cnt[1])
    def eval(self):
        num_eval_rounds = conf.num_eval_rounds
        self.model.eval()
        self.best_model.eval()

        # 创建两个 MCTS_Parallel 实例
        cur_mcts = MCTS.MCTS_Parallel(conf.num_eval_searches, num_eval_rounds, self.model, epsilon=0)
        best_mcts = MCTS.MCTS_Parallel(conf.num_eval_searches, num_eval_rounds, self.best_model, epsilon=0)

        # 创建两局 SPG
        spGames = []
        for _ in range(num_eval_rounds):
            spGames.append(SPG(self.game))  # 新模型先手
            spGames.append(SPG(self.game))  # 旧模型先手

        first_player = [0, 1] * num_eval_rounds
        win_results = [-1] * (2 * num_eval_rounds)
        step = 0
        train_bar = tqdm(range(1),leave=False,position=1,desc=f"Evaling...")
        while any(w == -1 for w in win_results):
            step += 1
            train_bar.set_description(f"Evaling, step={step}")
            # 根据 step 和 first_player 决定每局走棋方
            sides = [(first_player[i] + step - 1) % 2 for i in range(len(spGames))]

            # 哪些局还没结束
            active_indices = [i for i, w in enumerate(win_results) if w == -1]
            if not active_indices:
                break

            # 生成当前活跃局的 SPG 列表
            active_spgs = [spGames[i] for i in active_indices]
            active_sides = [sides[i] for i in active_indices]

            # 分组：同一模型走的局可以一次性 batch 搜索
            cur_spgs = [spg for spg, side in zip(active_spgs, active_sides) if side == 0]
            best_spgs = [spg for spg, side in zip(active_spgs, active_sides) if side == 1]

            # 批量搜索
            with profiler.section("eval/mcts"):
                cur_pis = cur_mcts.search(cur_spgs) if cur_spgs else []
                best_pis = best_mcts.search(best_spgs) if best_spgs else []

            # 合并结果，顺序和 active_indices 对齐
            pi_dict = {}
            cur_idx = 0
            best_idx = 0
            for i, side in enumerate(active_sides):
                if side == 0:
                    pi_dict[active_indices[i]] = cur_pis[cur_idx]
                    cur_idx += 1
                else:
                    pi_dict[active_indices[i]] = best_pis[best_idx]
                    best_idx += 1

            # 执行落子并更新状态
            for idx in active_indices:
                spg = spGames[idx]
                pi = pi_dict[idx]
                movables = self.game.get_valid_moves(spg.root.state)
                pi[movables == 0] = 0
                pi /= np.sum(pi)
                action = np.argmax(pi)
                new_node = spg.root.get_child(action)
                spg.root = new_node

                # 检查终局
                is_terminal, value = self.game.is_terminal(new_node.state, action)
                if is_terminal:
                    #print(spg.root.state)
                    if value == 1:
                        if sides[idx] == 0:
                            win_results[idx] = 1
                        else:
                            win_results[idx] = 0
                    else:
                        win_results[idx] = 0.5  # 平局
        return sum(win_results)/len(win_results)
    

    def update(self, iter, rate):
        if rate >= conf.update_rate or (iter < conf.warmup_iter and rate >= conf.warmup_update_rate):
            self.best_model.load_state_dict(self.model.state_dict())
            self.version += 1
        
            
        
    def learn(self, num_learn_iters, num_parallel_games=None, iter_start = 0):
        try:
            loss = 0
            iter_bar = tqdm(range(iter_start,num_learn_iters),leave=False, position=0, desc=f"Iter {iter_start}/{num_learn_iters} v={self.version}")
            win_rates = []
            rate=0
            data=[]
            for iter in iter_bar:
                iter_bar.set_description(f"Iter {iter}/{num_learn_iters},len={len(data)},loss={loss:.2f}, v={self.version} rate={rate:.2f}")
                #退火
                self.annealing(iter,num_learn_iters)
                if num_parallel_games:
                    data = self.selfplay_parallel(conf.num_selfplay_rounds,num_parallel_games)
                else:
                    data = self.selfplay(conf.num_selfplay_rounds)
                iter_bar.set_description(f"Iter {iter}/{num_learn_iters},len={len(data)},loss={loss:.2f}, v={self.version} rate={rate:.2f}")
                loss = self.train(data)
                iter_bar.set_description(f"Iter {iter}/{num_learn_iters},len={len(data)},loss={loss:.2f}, v={self.version} rate={rate:.2f}")
                self.loss_records.append((iter,loss))
                win_rates.append(self.eval())
                if iter < conf.warmup_iter:
                    rate = np.array(win_rates[-conf.warmup_num_eval_K:]).mean()
                else:
                    rate = np.array(win_rates[-conf.num_eval_K:]).mean()
                self.update(iter, rate)
                iter_bar.set_description(f"Iter {iter}/{num_learn_iters},len={len(data)},loss={loss:.2f}, v={self.version} rate={rate:.2f}")
                if conf.enable_profiling and (iter + 1) % conf.profile_every == 0:
                    profiler.dump(conf.profile_output)
                    profiler.reset()
                if (iter+1) % conf.num_eval_internal == 0:
                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"model_{conf.game_name}_{current_time}_iter_{iter}.pth"
                    iter_bar.clear()
                    if(rate >= conf.update_rate):
                        utils.save_model(self.best_model,self.optimizer,iter,self.loss_records,filename,self.version)
                        iter_bar.set_description(f"Iter {iter}/{num_learn_iters},len={len(data)},loss={loss:.2f}, v={self.version} rate={rate:.2f}")

            filename = f"model_{conf.game_name}_trained_{num_learn_iters}_v{self.version}_with_{sum([p.numel() for p in self.model.parameters()])/1e6}M_parameters.pth"
            utils.save_model(self.best_model,self.optimizer,iter,self.loss_records,filename,self.version)
        except KeyboardInterrupt:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{conf.game_name}_{current_time}_iter_{iter}_v{self.version}.pth"
            utils.save_model(self.best_model,self.optimizer,iter,self.loss_records,filename,self.version)

        
    

if __name__ == '__main__':
    train_worker = TrainManager(game=env.gomoku,
                                model=m.alphaZeroNet,
                                mcts=MCTS.mcts_parallel,
                                optimizer=torch.optim.AdamW(m.alphaZeroNet.parameters(), lr=conf.lr, weight_decay=conf.weight_decay),
                                T=conf.temperature
                                )


    train_worker.learn(conf.num_learn_iters,conf.num_parallel_games)
