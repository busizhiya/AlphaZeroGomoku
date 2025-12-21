import pygame
import sys
import numpy as np
import conf
import MCTS
import env
import torch.nn.functional as F
# GUI 参数
CELL_SIZE = 60
MARGIN = 20
SIDEBAR_WIDTH = 300
BOARD_COLOR = (230, 200, 150)
SIDEBAR_COLOR = (240, 240, 240)
LINE_COLOR = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TEXT_COLOR = (50, 50, 50)
HIGHLIGHT_COLOR = (255, 80, 80)
class GomokuGUI:
    def __init__(self, num_row, num_col, game, mcts):
        pygame.init()
        self.num_row = num_row
        self.num_col = num_col
        self.game = game
        self.mcts = mcts
        # 计算窗口大小
        self.board_width = MARGIN * 2 + CELL_SIZE * num_col
        self.board_height = MARGIN * 2 + CELL_SIZE * num_row
        total_width = self.board_width + SIDEBAR_WIDTH
        total_height = self.board_height

        self.screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("Gomoku")
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont(None, 24)
        self.font_medium = pygame.font.SysFont(None, 28)
        self.font_large = pygame.font.SysFont(None, 36)

        # 侧边栏
        self.text_lines = []
        self.highlighted_lines = set()

        # 按钮
        self.reset_button_rect = None

        # 最近落子位置
        self.last_row_col = None

    # --- 侧边栏文本 ---
    def clear_text(self):
        self.text_lines = []
        self.highlighted_lines = set()

    def add_text(self, text, highlight=False):
        self.text_lines.append(text)
        if highlight:
            self.highlighted_lines.add(len(self.text_lines) - 1)

    # --- 绘制棋盘 ---
    def show_board(self, state):
        self.screen.fill(BOARD_COLOR)

        # 绘制棋盘网格
        for r in range(self.num_row):
            y = MARGIN + r * CELL_SIZE + CELL_SIZE//2
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN + CELL_SIZE//2, y),
                (MARGIN + CELL_SIZE//2 + CELL_SIZE*(self.num_col-1), y), 2
            )
        for c in range(self.num_col):
            x = MARGIN + c * CELL_SIZE + CELL_SIZE//2
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (x, MARGIN + CELL_SIZE//2),
                (x, MARGIN + CELL_SIZE//2 + CELL_SIZE*(self.num_row-1)), 2
            )

        # 绘制棋子
        for r in range(self.num_row):
            for c in range(self.num_col):
                val = state[r][c]
                if val == 0:
                    continue

                center = (MARGIN + c * CELL_SIZE + CELL_SIZE//2,
                          MARGIN + r * CELL_SIZE + CELL_SIZE//2)
                radius = CELL_SIZE//2 - 5

                if val == 1:
                    pygame.draw.circle(self.screen, WHITE, center, radius)
                    pygame.draw.circle(self.screen, LINE_COLOR, center, radius, 2)
                else:
                    pygame.draw.circle(self.screen, BLACK, center, radius)

        # --- 显示最近落子（红色高亮） ---
        if self.last_row_col is not None:
            rr, cc = self.last_row_col
            rect = pygame.Rect(
                MARGIN + cc * CELL_SIZE,
                MARGIN + rr * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 3)

        # --- sidebar ---
        self.draw_sidebar()

        pygame.display.flip()

    # --- 绘制侧边栏及 reset 按钮 ---
    def draw_sidebar(self):
        x0 = self.board_width
        pygame.draw.rect(self.screen, SIDEBAR_COLOR, (x0, 0, SIDEBAR_WIDTH, self.board_height))

        # Title
        title = self.font_large.render("Info", True, TEXT_COLOR)
        self.screen.blit(title, (x0 + 20, 20))

        # Text
        y = 80
        for i, text in enumerate(self.text_lines):
            color = HIGHLIGHT_COLOR if i in self.highlighted_lines else TEXT_COLOR
            font = self.font_medium if i in self.highlighted_lines else self.font_small
            surface = font.render(text, True, color)
            self.screen.blit(surface, (x0 + 20, y))
            y += 35

        # --- Reset 按钮 ---
        button_w = SIDEBAR_WIDTH - 60
        button_h = 45
        button_x = x0 + 30
        button_y = self.board_height - 70

        self.reset_button_rect = pygame.Rect(button_x, button_y, button_w, button_h)
        pygame.draw.rect(self.screen, (200, 50, 50), self.reset_button_rect)
        reset_text = self.font_medium.render("Reset", True, WHITE)
        self.screen.blit(
            reset_text,
            (button_x + button_w // 2 - reset_text.get_width() // 2,
             button_y + button_h // 2 - reset_text.get_height() // 2)
        )

    # --- 坐标转换 ---
    def get_click(self, pos):
        x, y = pos
        x -= MARGIN
        y -= MARGIN
        if x < 0 or y < 0:
            return None
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        if 0 <= row < self.num_row and 0 <= col < self.num_col:
            return row, col
        return None

    # --- 主循环 ---
    def loop(self, state, player):
        # player先手方: 人类 = +1，AI = -1
        self.is_won = False
        while True:
            self.clock.tick(30)
            self.show_board(state)
            if player == -1:
                # ------ AI 自动走 ------
                if not self.is_won and player == -1:
                    
                    ai_state = self.game.get_opponent_state(state)
                    state, action = self.play_by_ai(ai_state)            
                    self.last_row_col = (action // conf.num_col, action % conf.num_col)

                    if self.check_end(state, action, player):
                        self.is_won = True
                        continue
                    self.show_board(state)
                    player = 1
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos

                    # --- 点击 Reset ---
                    if self.reset_button_rect.collidepoint(pos):
                        state[:] = 0
                        self.clear_text()
                        player = 1
                        self.last_row_col = None
                        self.is_won = False
                        continue

                    # --- 人类落子 ---
                    if not self.is_won and player == 1:
                        rc = self.get_click(pos)
                        if rc is None:
                            continue
                        r, c = rc
                        if state[r][c] != 0:
                            continue
                        # 落子
                        state[r][c] = 1
                        self.last_row_col = (r, c)

                        # 检查终局
                        if self.check_end(state, r*conf.num_col + c, player):
                            self.is_won = True
                            continue
                        self.show_board(state)
                        # 轮到 AI
                        player = -1
            
            # ------ AI 自动走 ------
            if not self.is_won and player == -1:

                ai_state = self.game.get_opponent_state(state)
                state, action = self.play_by_ai(ai_state)            
                self.last_row_col = (action // conf.num_col, action % conf.num_col)

                if self.check_end(state, action, player):
                    self.is_won = True
                    continue
                self.show_board(state)
                player = 1

    def play_by_ai(self,state):
        root = MCTS.Node(state, -1)
        pi = self.mcts.search(root)

        movable = self.game.get_valid_moves(root.state)
        pi[movable == 0] = 0
        pi /= pi.sum()
        if conf.temperature == 0:
            action = np.argmax(pi)
        else:
            pi = pi ** (1/conf.temperature)
            pi /= pi.sum()
            action = np.random.choice(len(pi), p=pi)

        new_node = root.get_child(action)
        return new_node.state, action

    def check_end(self,state, action, player):
        is_terminal, value = self.game.is_terminal(state, action)
        if is_terminal:
            text = "Human Won!" if player == 1 else "AI Won!"
            if value == 0:
                text = "Draw"
            self.add_text(text, highlight=True)
            self.show_board(state)
            is_won = True
        return is_terminal


# ======================
# ======= 主程序 =========
# ======================

if __name__ == "__main__":
    game = env.gomoku
    mcts = MCTS.mcts
    state = game.get_init_state()
    gui = GomokuGUI(conf.num_row, conf.num_col, game, mcts)
    gui.loop(state,1)
