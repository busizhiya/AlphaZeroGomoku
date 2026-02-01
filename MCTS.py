import conf
import numpy as np
import env
import model as m
import torch
import torch.nn.functional as F
from profiler import profiler

try:
    if conf.enable_numba:
        from numba import njit
    else:
        njit = None
except Exception:
    njit = None
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
profiler.set_enabled(conf.enable_profiling)

_nn_cache = {}
_nn_cache_order = []


def _state_key(state):
    return np.ascontiguousarray(state).tobytes()


def _cache_get(key):
    return _nn_cache.get(key)


def _cache_put(key, p_np, v_np):
    if not conf.enable_nn_cache:
        return
    if key in _nn_cache:
        return
    _nn_cache[key] = (p_np, v_np)
    _nn_cache_order.append(key)
    if len(_nn_cache_order) > conf.nn_cache_max:
        old_key = _nn_cache_order.pop(0)
        _nn_cache.pop(old_key, None)


if njit is not None:
    @njit(cache=True)
    def _puct_select(N, W, P, parent_n_sqrt, c_puct):
        best_idx = 0
        best_puct = -1e18
        for i in range(N.shape[0]):
            if N[i] == 0:
                Q = 0.0
            else:
                Q = -(W[i] / N[i])
            U = c_puct * P[i] * parent_n_sqrt / (1.0 + N[i])
            puct = Q + U
            if puct > best_puct:
                best_puct = puct
                best_idx = i
        return best_idx
else:
    _puct_select = None
class Node:
    __slots__ = (
        'N', 'W', 'P',
        'children',
        'child_list',
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
        self.child_list = []
        self.is_root_expanded = False

    def clear_child(self):
        self.children = [None] * conf.action_size
        self.child_list = []
    def clear_parent(self):
        self.parent= None
    def is_expanded(self):
        return len(self.child_list) > 0

    def select(self):
        '''self is a parent node.
        Now we need to select one action based on their PUCT score
        return best_child, best_action
        '''
        # (s,a) = argmax(PUCT(s,·))
        # PUCT = Q(s,a) + U(s,a)
        # U(s,a) = C_puct * P(s,a) * (Parent's N) ** 0.5 / (1 + N(s,a))
        C_puct = conf.C_puct
        parent_n_sqrt = self.N ** 0.5
        if _puct_select is not None:
            N = np.fromiter((c.N for c in self.child_list), dtype=np.float32)
            W = np.fromiter((c.W for c in self.child_list), dtype=np.float32)
            P = np.fromiter((c.P for c in self.child_list), dtype=np.float32)
            idx = _puct_select(N, W, P, parent_n_sqrt, C_puct)
            return self.child_list[idx]

        best_child = None
        best_puct = float("-inf")
        for child in self.child_list:
            if child.N == 0:
                Q = 0
            else:
                # 孩子的价值越大, 对方越容易赢
                Q = -(child.W / child.N)
            U = C_puct * child.P * parent_n_sqrt / (1 + child.N)
            puct = Q + U
            if puct > best_puct:
                best_puct = puct
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
        key = _state_key(node.state)
        cached = _cache_get(key) if conf.enable_nn_cache else None
        if cached is not None:
            P = torch.tensor(cached[0], device=conf.device)
            v = torch.tensor(cached[1], device=conf.device)
        else:
            with profiler.section("mcts/expand/nn"):
                with torch.inference_mode():
                    if conf.enable_amp_inference and conf.device == "cuda":
                        dtype = torch.float16 if conf.amp_dtype == "float16" else torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            P, v = self.model(game.encode(node.state))
                    else:
                        P, v = self.model(game.encode(node.state))
            _cache_put(
                key,
                P.squeeze(0).detach().float().cpu().numpy(),
                v.squeeze(0).detach().float().cpu().numpy(),
            )
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
                node.child_list.append(child)
        return v

    @torch.no_grad()
    def search(self, root: Node):
        with profiler.section("mcts/search"):
            # 先对根节点做expand & 噪声处理
            is_terminal, v = game.is_terminal(root.state, root.action_taken)
            if not is_terminal:
                if not root.is_expanded():
                    # EXPAND
                    with profiler.section("mcts/expand/root"):
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
                with profiler.section("mcts/select"):
                    while node.is_expanded():
                        node = node.select()
                # Now, node is a leaf node

                # terminal?
                is_terminal, v = game.is_terminal(node.state, node.action_taken)
                v *= -1

                if not is_terminal:
                    # EXPAND
                    with profiler.section("mcts/expand"):
                        v = self.expand(node)

                # BACKUP
                with profiler.section("mcts/backup"):
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
        Ps_list = [None] * len(spGames)
        vs_list = [None] * len(spGames)
        miss_states = []
        miss_indices = []
        miss_keys = []
        for i, spg in enumerate(spGames):
            key = _state_key(spg.node.state)
            cached = _cache_get(key) if conf.enable_nn_cache else None
            if cached is not None:
                Ps_list[i] = torch.tensor(cached[0], device=conf.device)
                vs_list[i] = torch.tensor(cached[1], device=conf.device)
            else:
                miss_states.append(spg.node.state)
                miss_indices.append(i)
                miss_keys.append(key)

        if miss_states:
            with profiler.section("mcts/expand_parallel/nn"):
                with torch.inference_mode():
                    if conf.enable_amp_inference and conf.device == "cuda":
                        dtype = torch.float16 if conf.amp_dtype == "float16" else torch.bfloat16
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            Ps_miss, vs_miss = self.model(game.encode(np.array(miss_states)))
                    else:
                        Ps_miss, vs_miss = self.model(game.encode(np.array(miss_states)))
            for j, i in enumerate(miss_indices):
                Ps_list[i] = Ps_miss[j]
                vs_list[i] = vs_miss[j]
                _cache_put(
                    miss_keys[j],
                    Ps_miss[j].detach().float().cpu().numpy(),
                    vs_miss[j].detach().float().cpu().numpy(),
                )

        Ps = torch.stack(Ps_list, dim=0)
        vs = torch.stack(vs_list, dim=0)
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
                    spg.node.child_list.append(child)
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
            with profiler.section("mcts/expand_parallel/root"):
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
        
        expand_queue = []
        batch_size = conf.expand_batch_size if conf.enable_async_expand else 0
        for i in range(self.num_searches):
            # Selecting
            with profiler.section("mcts/select_parallel"):
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
                        if batch_size > 0:
                            expand_queue.append(spg)
                    else:
                        # Reaching terminal, backpropagate
                        node.backup(v)

            if batch_size > 0:
                if len(expand_queue) >= batch_size:
                    with profiler.section("mcts/expand_parallel"):
                        vs = self.expand(expand_queue)
                    with profiler.section("mcts/backup_parallel"):
                        for v, spg in zip(vs, expand_queue):
                            spg.node.backup(v)
                    expand_queue = []
            else:
                expandable_spGames = [spg for spg in spGames if spg.node is not None]
                if len(expandable_spGames) > 0:
                    # Expanding
                    with profiler.section("mcts/expand_parallel"):
                        vs = self.expand(expandable_spGames)
                    # BACKUP
                    with profiler.section("mcts/backup_parallel"):
                        for v, spg in zip(vs, expandable_spGames):
                            spg.node.backup(v)

        if batch_size > 0 and expand_queue:
            with profiler.section("mcts/expand_parallel"):
                vs = self.expand(expand_queue)
            with profiler.section("mcts/backup_parallel"):
                for v, spg in zip(vs, expand_queue):
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
