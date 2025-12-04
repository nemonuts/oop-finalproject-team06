import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame  # 引入 pygame 繪圖庫

class GomokuEnv(gym.Env):
    """
    【五子棋環境 GomokuEnv - Pygame GUI 版】
    這是升級版的環境，使用 Pygame 彈出視窗來顯示畫面，
    不再是在 Terminal 裡面印符號。
    """
    # 設定渲染模式與 FPS
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, board_size=9, win_streak=5, render_mode=None):
        self.board_size = board_size
        self.win_streak = win_streak
        self.render_mode = render_mode
        
        # 0: 空位, 1: 黑棋 (Player 1), 2: 白棋 (Player 2)
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1 
        
        # 動作空間與觀察空間
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(board_size, board_size), dtype=int)

        # Pygame 視窗設定
        self.window_size = 512  # 視窗大小 (像素)
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        
        # 如果是 human 模式，重置時順便渲染第一幀
        if self.render_mode == "human":
            self._render_frame()
            
        return self.board, {}

    def step(self, action):
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            return self.board, -10, False, False, {"error": "Invalid move"}

        self.board[row, col] = self.current_player

        terminated = False
        reward = 1
        info = {}

        if self.check_win(row, col):
            reward = 100
            terminated = True
            info["winner"] = self.current_player
        elif np.all(self.board != 0):
            reward = 0
            terminated = True
            info["winner"] = 0 # 和局

        self.current_player = 3 - self.current_player 
        
        if self.render_mode == "human":
            self._render_frame()

        return self.board, reward, terminated, False, info

    def check_win(self, row, col):
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] 

        for dr, dc in directions:
            count = 1
            for i in range(1, self.win_streak):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else: break
            for i in range(1, self.win_streak):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else: break
            
            if count >= self.win_streak: return True
        return False

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        # 初始化視窗 (只執行一次)
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Gomoku AI Arena") # 設定視窗標題
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 建立畫布
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((221, 187, 136)) # 背景色：木頭顏色

        pix_square_size = self.window_size / self.board_size

        # 1. 畫格線
        for x in range(self.board_size):
            # 畫橫線
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pix_square_size / 2, (x + 0.5) * pix_square_size),
                (self.window_size - pix_square_size / 2, (x + 0.5) * pix_square_size),
                width=2,
            )
            # 畫直線
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                ((x + 0.5) * pix_square_size, pix_square_size / 2),
                ((x + 0.5) * pix_square_size, self.window_size - pix_square_size / 2),
                width=2,
            )

        # 2. 畫棋子
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:
                    continue
                
                # 計算中心點位置
                center_x = (c + 0.5) * pix_square_size
                center_y = (r + 0.5) * pix_square_size
                radius = pix_square_size / 2.5

                if self.board[r, c] == 1: # 黑棋
                    pygame.draw.circle(canvas, (0, 0, 0), (center_x, center_y), radius)
                elif self.board[r, c] == 2: # 白棋
                    pygame.draw.circle(canvas, (255, 255, 255), (center_x, center_y), radius)
                    # 白棋加個黑框比較好看
                    pygame.draw.circle(canvas, (0, 0, 0), (center_x, center_y), radius, width=2)

        if self.render_mode == "human":
            # 更新視窗內容
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump() # 處理視窗事件，避免當機
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def get_valid_moves(self):
        return np.where(self.board.flatten() == 0)[0]

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()