
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

import torch

PLAY = False
game_name = '8x8_gomoku'
num_row = 8
num_col = 8
action_size = num_col * num_row
win_k = 5  # 五子棋

# ===== 模型架构 =====
# 五子棋相对复杂，需要更强的模型
num_res_blocks = 5  # 增加到5个残差块
channels = 192  # 增加到256通道

# ===== MCTS 搜索参数 =====
C_puct = 1.25  # 9x9需要更多探索，略降C_puct
device = "cuda" if torch.cuda.is_available() else "cpu"
# 9x9五子棋搜索树很大，需要大幅增加搜索次数
num_searches = 500  # 训练时至少600次搜索
num_eval_searches = 1000  # 评估时1200次

# ===== 训练参数 =====
num_eval_internal = 16  # 9x9训练慢，评估间隔可以大些
num_eval_K = 10  # 用最近10次评估平均
warmup_iter = 150  # 延长热身期到150轮
warmup_update_rate = 0.53  # 热身期要求稍低
warmup_num_eval_K = 5  # 热身期用最近5次评估
annealing_steps = 15  # 退火步数
max_select_depth = action_size
update_rate = 0.53  # 胜率达到53%更新（9x9更难达到高胜率）
temperature = 1.0
T = temperature

# ===== 自对弈参数 =====
# 9x9游戏时间较长，不要太多轮次
num_selfplay_rounds = 1  # 减少到2轮，但增加并行游戏数
num_parallel_games = 80  # 增加到120个并行游戏
num_learn_iters = 200  # 增加到1500轮（9x9需要更长时间训练）

# ===== 优化参数 =====
# 根据GPU内存调整
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16e9:  # 16GB以上
    batch_size = 512
else:
    batch_size = 256  # 保守选择

epsilon = 0.15  # 减少Dirichlet噪声（9x9更需要精确搜索）
alpha = 0.8  # Dirichlet参数
lr = 2e-4  # 学习率可以稍大，前期学习更快
weight_decay = 3e-5  # 权重衰减

# ===== 对战参数 =====
if PLAY:
    load_model = "best_model_9x9_gomoku.pth"
    temperature = 0.05  # 非常小的温度
    epsilon = 0
    num_searches = 2000  # 对战时2000次搜索
else:
    load_model = None

# import torch

# PLAY = False
# game_name = '9x9_gomoku'
# num_row = 9
# num_col = 9
# action_size = num_col * num_row
# win_k = 5  # 五子棋

# # ===== 模型架构 =====
# # 五子棋相对复杂，需要更强的模型
# num_res_blocks = 5  # 增加到5个残差块
# channels = 256  # 增加到256通道

# # ===== MCTS 搜索参数 =====
# C_puct = 1.25  # 9x9需要更多探索，略降C_puct
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # 9x9五子棋搜索树很大，需要大幅增加搜索次数
# num_searches = 600  # 训练时至少600次搜索
# num_eval_searches = 1200  # 评估时1200次

# # ===== 训练参数 =====
# num_eval_internal = 16  # 9x9训练慢，评估间隔可以大些
# num_eval_K = 10  # 用最近10次评估平均
# warmup_iter = 150  # 延长热身期到150轮
# warmup_update_rate = 0.53  # 热身期要求稍低
# warmup_num_eval_K = 5  # 热身期用最近5次评估
# annealing_steps = 15  # 退火步数
# max_select_depth = action_size
# update_rate = 0.53  # 胜率达到53%更新（9x9更难达到高胜率）
# temperature = 1.0
# T = temperature

# # ===== 自对弈参数 =====
# # 9x9游戏时间较长，不要太多轮次
# num_selfplay_rounds = 1  # 减少到2轮，但增加并行游戏数
# num_parallel_games = 150  # 增加到120个并行游戏
# num_learn_iters = 200  # 增加到1500轮（9x9需要更长时间训练）

# # ===== 优化参数 =====
# # 根据GPU内存调整
# if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 16e9:  # 16GB以上
#     batch_size = 512
# else:
#     batch_size = 256  # 保守选择

# epsilon = 0.15  # 减少Dirichlet噪声（9x9更需要精确搜索）
# alpha = 0.8  # Dirichlet参数
# lr = 2e-4  # 学习率可以稍大，前期学习更快
# weight_decay = 3e-5  # 权重衰减

# # ===== 对战参数 =====
# if PLAY:
#     load_model = "best_model_9x9_gomoku.pth"
#     temperature = 0.05  # 非常小的温度
#     epsilon = 0
#     num_searches = 2000  # 对战时2000次搜索
# else:
#     load_model = None
