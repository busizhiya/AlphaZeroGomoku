import conf
import numpy as np
import env
import model as m
import torch
import torch.nn.functional as F
'''
首先明确几点:
    1.神经网络的输出P,v 分别代表着: 输入状态后的各个子动作的胜率, 当前状态的胜率
    2.当我们到达一个叶节点时, 要做的就是: 判断在当前状态下, 各个子动作带来的胜率, 并赋值给子节点的prior
        同时返回当前叶节点的胜率
    3.当我们在SELECT环节中, 我们想找一条PUCT最大的路, 直至到达一个叶节点----即未预估子动作结果的节点
    4.每个节点的state, 都是从自己的视角出发, 默认下一步是自己下棋, 且自己执1
'''
game = env.gomoku
model = m.alphaZeroNet
class Node:
    __slots__ = (
        'N', 'W', 'P',
        'children',
        'parent',
        'state',
        'action_taken',
        'is_root_expanded'
    )
    def __init__(self, state, action_taken = 0, parent=None):
        self.W = 0
        self.N = 0
        self.P = 0
        self.parent = parent
        self.state = state
        self.action_taken = action_taken
        self.children = [None] * conf.action_size
        self.is_root_expanded = False

    def clear_child(self):
        self.children = [None] * conf.action_size
    def clear_parent(self):
        self.parent= None
    def is_expanded(self):
        return any(child is not None for child in self.children)

    def select(self):
        '''self is a parent node.
        Now we need to select one action based on their PUCT score
        return best_child, best_action
        '''
        # (s,a) = argmax(PUCT(s,·))
        # PUCT = Q(s,a) + U(s,a)
        # U(s,a) = C_puct * P(s,a) * (Parent's N) ** 0.5 / (1 + N(s,a))
        C_puct = conf.C_puct
        best_child = None
        best_puct = float("-inf")
        for child in self.children:
            if child is None:
                continue
            if child.N == 0:
                Q = 0
            else:
                # 孩子的价值越大, 对方越容易赢
                Q = -(child.W / child.N)
            U = C_puct * child.P * (self.N**0.5) / (1+child.N)
            if Q+U > best_puct:
                best_puct = Q+U
                best_child = child
        return best_child
    
    def backup(self, v):
        # if is_terminal, that means action taken by self can lead him to success
        # and self should get reward for taking this action.
        # the child, which is the terminal lead node, should get more visited, 
        # thus we add more visit count and value to that child,
        # making sure it gets higher PUCT score
        
        # node's value V is meaningful to its parent, since we only use V in
        # parent selecting child based on PUCT
        # if a leaf node won, them that very leaf node should have higher V, 
        # in order to select it more often. 
        node = self
        while node:
            node.W += v
            node.N += 1
            v = -v
            node = node.parent
            
    def get_child(self, action):
        return self.children[action]


class MCTS:
    def __init__(self,num_searches,model,epsilon=conf.epsilon):
        self.num_searches = num_searches
        self.model = model
        self.epsilon = epsilon
    def expand(self,node):
        '''We are sure self is movable, since we've checked it's not terminal yet
        Find all movable action, get (P,v), create child nodes, set prior and return v
        '''
        
        # self is a leaf node, which has the opposite perspective of parent node
        # However, we are predicting from child's perspective, so that's correct.
        P,v = self.model(game.encode(node.state))
        P = P.squeeze(0)
        v = v.squeeze(0)
        v = float(v.cpu().item())
        movable = torch.tensor(game.get_valid_moves(node.state),dtype=torch.bool,device=P.device) # mask
        P[~movable] = 1e-9
        P = F.softmax(P,dim=0).cpu().numpy()
        for action,is_valid in enumerate(movable):
            if is_valid == 1:
                # every node holds the state from its perspective
                child = Node(game.get_opponent_state(game.get_next_state(node.state.copy(),action)),action,node)
                child.P = P[action]
                node.children[action] = child
        return v

    @torch.no_grad()
    def search(self, root: Node):
        # 先对根节点做expand & 噪声处理
        is_terminal, v = game.is_terminal(root.state, root.action_taken)
        if not is_terminal:
            if not root.is_expanded():
                # EXPAND
                v = self.expand(root)
                # 取所有 children 的 action
                actions = [i  for i in range(len(root.children)) if root.children[i] is not None]
                # 创建 Dirichlet 噪声
                noise = np.random.dirichlet([conf.alpha] * len(actions))
                # 混合噪声
                if not root.is_root_expanded:
                    root.is_root_expanded = True
                    for a, n in zip(actions, noise):
                        old_p = root.children[a].P
                        root.children[a].P = (1 - self.epsilon) * old_p + self.epsilon * n
        for i in range(self.num_searches):
            # SELECT
            node = root
            # make some noise~
            while node.is_expanded():
                node = node.select()
            # Now, node is a leaf node

            # terminal?
            is_terminal, v = game.is_terminal(node.state, node.action_taken)
            v *= -1

            if not is_terminal:
                # EXPAND
                v = self.expand(node)

            # BACKUP    
            node.backup(v)
        
        pi = game.get_valid_moves(root.state).astype(np.int32)
        for action, child in enumerate(root.children):
            if child is not None:
                pi[action] = child.N
        pi = pi / np.sum(pi) # turning it into probs
        return pi

class MCTS_Parallel:
    def __init__(self,num_searches, num_parallel_games,model,epsilon=conf.epsilon):
        self.num_searches = num_searches
        self.num_parallel_games = num_parallel_games
        self.model = model
        self.epsilon=epsilon
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon

    @torch.no_grad()
    def expand(self,spGames):
        Ps,vs = self.model(game.encode(np.array([spg.node.state for spg in spGames])))
        vs = vs.cpu().numpy()
        movables = game.get_valid_moves(np.array([spg.node.state for spg in spGames]))
        mask = torch.tensor(movables, device=Ps.device, dtype=torch.bool)
        Ps = Ps.masked_fill(~mask, -1e9)
        Ps = torch.softmax(Ps, dim=1)
        Ps = Ps.detach().cpu().numpy()
        for i, spg in enumerate(spGames):
            for action, is_valid in enumerate(movables[i]):
                if is_valid and spg.node.children[action] is None:
                    child = Node(game.get_opponent_state(game.get_next_state(spg.node.state.copy(), action)),action,spg.node)
                    child.P = Ps[i][action]
                    spg.node.children[action] = child
        return vs
    @torch.no_grad()
    def search(self, spGames):
        #roots = np.stack([spg.root for spg in spGames])
        # 先对根节点做expand & 噪声处理
        expand_spgs = []
        for spg in spGames:
            spg.node = spg.root
            if not spg.root.is_expanded():
                expand_spgs.append(spg)

        if expand_spgs:
            self.expand(expand_spgs)
        # 取所有 children 的 action
        for spg in spGames:
            if spg.root.is_root_expanded:
                continue
            spg.root.is_root_expanded = True
            actions = [i  for i in range(len(spg.root.children)) if spg.root.children[i] is not None]
            # 创建 Dirichlet 噪声
            noise = np.random.dirichlet([conf.alpha] * len(actions))
            # 混合噪声
            for a, n in zip(actions, noise):
                old_p = spg.root.children[a].P
                spg.root.children[a].P = (1 - self.epsilon) * old_p + self.epsilon * n
        
        for i in range(self.num_searches):
            # Selecting
            for spg in spGames[::-1]:
                spg.node = None
                node = spg.root
                depth = 0
                while node.is_expanded() and depth < conf.max_select_depth:
                    node = node.select()
                    depth += 1

                is_terminal, v = game.is_terminal(node.state, node.action_taken)
                v *= -1
                if not is_terminal:
                    spg.node = node
                else:
                    # Reaching terminal, backpropagate
                    node.backup(v)
            expandable_spGames = [spg for spg in spGames if spg.node is not None]
            if len(expandable_spGames) > 0:
                # Expanding
                vs = self.expand(expandable_spGames)
                # BACKUP  
                for v,spg in zip(vs, expandable_spGames):
                    spg.node.backup(v)
                  
        pis = []
        for spg in spGames:
            pi = game.get_valid_moves(spg.root.state).astype(np.int32)
            for action, child in enumerate(spg.root.children):
                if child is not None:
                    pi[action] = child.N
            pi = pi / np.sum(pi) # turning it into probs
            pis.append(pi)
        return pis



mcts = MCTS(conf.num_searches,m.alphaZeroNet)
mcts_parallel = MCTS_Parallel(conf.num_searches,conf.num_parallel_games,m.alphaZeroNet)
if __name__ == '__main__':
    root = Node(game.get_init_state(), -1)
    player = 1
    T = conf.temperature
    cnt = 0
    while True:
        pi = mcts.search(root)
        movables = game.get_valid_moves(root.state)
        pi[movables==0] = 0
        pi /= np.sum(pi)
        
        if T == 0:
            action = np.argmax(pi)
            #print(f"action={action}")
        else:
            pi = pi ** 1/T
            pi /= np.sum(pi)
            action = np.random.choice(game.action_size,p=pi)
        new_node = root.get_child(action)
        cnt += 1
        print(f"---step {cnt} ---")
        print(new_node.state)

        is_terminal, value = game.is_terminal(new_node.state, action)
        root = new_node
        if is_terminal:
            print()
            if value == 1:
                print(f"player {player} Won!")
                break
            else:
                print("Draw...")
                break   
        player *= -1