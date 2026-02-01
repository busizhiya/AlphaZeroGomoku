
# import torch

# PLAY = False
# game_name = 'Gomoku'
# num_row = 7
# num_col = 7
# action_size = num_col * num_row
# win_k = 4  # 四子棋

# # ===== 模型架构 =====
# num_res_blocks = 3  # 增加到3个残差块，更强的表达能力
# channels = 64  # 增加通道数，提高模型容量

# # ===== MCTS 搜索参数 =====
# C_puct = 1.5  # 降低探索权重，更快收敛
# device = "cuda" if torch.cuda.is_available() else "cpu"
# num_searches = 200  # 训练时搜索次数增加到200，确保搜索质量
# num_eval_searches = 400  # 评估时用更多搜索

# # ===== 训练参数 =====
# num_eval_internal = 10  # 每10轮评估一次
# num_eval_K = 5  # 用最近5次评估平均
# warmup_iter = 50  # 延长热身期
# warmup_update_rate = 0.55  # 热身期更新阈值
# warmup_num_eval_K = 3  # 热身期用最近3次评估
# annealing_steps = 15  # 延长退火步数
# max_select_depth = action_size
# update_rate = 0.55  # 胜率达到55%就更新
# temperature = 1.0
# T = temperature

# # ===== 自对弈参数 =====
# num_selfplay_rounds = 1  # 增加每轮自对弈次数，收集更多数据
# num_parallel_games = 96  # 增加并行游戏数，提高效率
# num_learn_iters = 500  # 增加到500轮训练

# # ===== 优化参数 =====
# batch_size = 256  # 增大批次大小
# epsilon = 0.25  # Dirichlet噪声强度
# alpha = 1.5  # 降低Dirichlet噪声参数，更稳定
# lr = 2e-4  # 增大学习率
# weight_decay = 5e-5  # 降低权重衰减

# if PLAY:
#     load_model = None
#     temperature = 0
#     epsilon = 0.2
#     num_searches = 800  # 对弈时用更多搜索
# else:
#     load_model = None

# import torch

# PLAY = False
# game_name = '5x5_for_in_row'
# num_row = 5
# num_col = 5
# action_size = num_col * num_row
# win_k = 4  # 四子棋

# # ===== 模型架构 =====
# num_res_blocks = 3  # 增加到3个残差块，更强的表达能力
# channels = 64  # 增加通道数，提高模型容量

# # ===== MCTS 搜索参数 =====
# C_puct = 1.5  # 降低探索权重，更快收敛
# device = "cuda" if torch.cuda.is_available() else "cpu"
# num_searches = 200  # 训练时搜索次数增加到200，确保搜索质量
# num_eval_searches = 400  # 评估时用更多搜索

# # ===== 训练参数 =====
# num_eval_internal = 10  # 每10轮评估一次
# num_eval_K = 5  # 用最近5次评估平均
# warmup_iter = 50  # 延长热身期
# warmup_update_rate = 0.55  # 热身期更新阈值
# warmup_num_eval_K = 3  # 热身期用最近3次评估
# annealing_steps = 10  # 延长退火步数
# max_select_depth = action_size
# update_rate = 0.55  # 胜率达到55%就更新
# temperature = 1.0
# T = temperature

# # ===== 自对弈参数 =====
# num_selfplay_rounds = 1  # 增加每轮自对弈次数，收集更多数据
# num_parallel_games = 96  # 增加并行游戏数，提高效率
# num_learn_iters = 500  # 增加到500轮训练

# # ===== 优化参数 =====
# batch_size = 256  # 增大批次大小
# epsilon = 0.25  # Dirichlet噪声强度
# alpha = 1.5  # 降低Dirichlet噪声参数，更稳定
# lr = 2e-4  # 增大学习率
# weight_decay = 5e-5  # 降低权重衰减

# if PLAY:
#     load_model = None
#     temperature = 0
#     epsilon = 0.2
#     num_searches = 800  # 对弈时用更多搜索
# else:
#     load_model = None

# import torch

# PLAY = False
# game_name = '15x15_gomoku'
# num_row = 15
# num_col = 15
# action_size = num_col * num_row
# win_k = 5  # 五子棋

# # ===== 模型架构 =====
# # 五子棋相对复杂，需要更强的模型
# num_res_blocks = 5
# channels = 192

# # ===== MCTS 搜索参数 =====
# C_puct = 1.25
# device = "cuda" if torch.cuda.is_available() else "cpu"
# num_searches = 500
# num_eval_searches = 1000

# # ===== 训练参数 =====
# num_eval_internal = 16
# num_eval_rounds = 1  # 每次评估双方各先手的局数
# num_eval_K = 10
# warmup_iter = 150
# warmup_update_rate = 0.53
# warmup_num_eval_K = 5
# annealing_steps = 15
# max_select_depth = action_size
# update_rate = 0.53
# temperature = 1.0
# T = temperature

# # ===== 自对弈参数 =====
# num_selfplay_rounds = 1
# num_parallel_games = 80
# num_learn_iters = 200

# # ===== 优化参数 =====
# if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16e9:
#     batch_size = 512
# else:
#     batch_size = 256

# epsilon = 0.15
# alpha = 0.8
# lr = 2e-4
# weight_decay = 3e-5

# # ===== Profiling（可选）=====
# enable_profiling = False
# profile_output = "profile.txt"
# profile_every = 1

# # ===== 对战参数 =====
# if PLAY:
#     load_model = "best_model_15x15_gomoku.pth"
#     temperature = 0.05
#     epsilon = 0
#     num_searches = 2000
# else:
#     load_model = None

import torch

PLAY = False
game_name = '15x15_gomoku'
num_row = 15
num_col = 15
action_size = num_col * num_row
win_k = 5  # 五子棋

# ===== 模型架构 =====
num_res_blocks = 10
channels = 256

# ===== MCTS 搜索参数 =====
C_puct = 1.25
device = "cuda" if torch.cuda.is_available() else "cpu"
num_searches = 800
num_eval_searches = 1600

# ===== 训练参数 =====
num_eval_internal = 16
num_eval_rounds = 2
num_eval_K = 10
warmup_iter = 150
warmup_update_rate = 0.53
warmup_num_eval_K = 5
annealing_steps = 15
max_select_depth = action_size
update_rate = 0.53
temperature = 1.0
T = temperature

# ===== 自对弈参数 =====
num_selfplay_rounds = 1
num_parallel_games = 96
num_learn_iters = 1500

# ===== 优化参数 =====
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16e9:
    batch_size = 512
else:
    batch_size = 256
epsilon = 0.15
alpha = 0.8
lr = 2e-4
weight_decay = 5e-5

# ===== 对战参数 =====
if PLAY:
    load_model = "best_model_15x15_gomoku.pth"
    temperature = 0.05
    epsilon = 0
    num_searches = 800
else:
    load_model = None

# ===== Profiling（可选）=====
enable_profiling = True
profile_output = "profile.txt"
profile_every = 1

# ===== 推理缓存（可选）=====
enable_nn_cache = True
nn_cache_max = 200000

# ===== MCTS 加速（可选）=====
enable_numba = True
enable_async_expand = True
expand_batch_size = 64

# ===== GPU 推理加速（可选）=====
enable_torch_compile = True
enable_amp_inference = True
amp_dtype = "float16"
