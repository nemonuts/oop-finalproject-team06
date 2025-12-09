import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame  # 引入 pygame 繪圖庫
import os

class GomokuEnv(gym.Env):
    """
    【五子棋環境 GomokuEnv - Pygame GUI 版】
    使用 Pygame 彈出視窗顯示畫面，並修正了背景平鋪邏輯，以達到無縫效果。
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
        self.tile_img = None # <-- 修正：用來暫存單個地板圖片

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.board, {}

    def step(self, action):
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            # 給予懲罰，但避免中斷
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
            pygame.display.set_caption("Gomoku AI Arena") 
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        
        pix_square_size = self.window_size / self.board_size # 計算單一格子像素大小

        # --- 圖片載入與縮放邏輯 (只執行一次) ---
        if self.tile_img is None:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                img_path = os.path.join(current_dir, "sprites", "floor.png")
                
                # 備用路徑檢查 (如果 floor.png 就在 env 檔案同層)
                if not os.path.exists(img_path):
                    img_path = os.path.join(current_dir, "floor.png")
                
                if os.path.exists(img_path):
                    print(f"成功載入單一地板圖片: {img_path}")
                    loaded_img = pygame.image.load(img_path)
                    # 關鍵修正：將圖片縮放成單個格子的大小
                    self.tile_img = pygame.transform.scale(loaded_img, (pix_square_size, pix_square_size))
                else:
                    print(f"警告：找不到圖片檔案，使用預設顏色。搜尋路徑: {img_path}")
                    self.tile_img = "DEFAULT_COLOR"

            except Exception as e:
                print(f"載入圖片發生錯誤: {e}")
                self.tile_img = "DEFAULT_COLOR"

        # --- 繪製背景 (平鋪邏輯) ---
        if self.tile_img != "DEFAULT_COLOR" and self.tile_img is not None:
            # 關鍵修正：使用雙層迴圈平鋪圖片
            for r in range(self.board_size):
                for c in range(self.board_size):
                    # 計算平鋪位置
                    pos_x = c * pix_square_size
                    pos_y = r * pix_square_size
                    canvas.blit(self.tile_img, (pos_x, pos_y))
        else:
            # 預設背景色
            canvas.fill((221, 187, 136)) 

        # 1. 畫格線 (如果需要，目前看起來圖片 floor.png 裡已經有格線了，建議移除)
        # 由於您的 floor.png 圖片看起來已經有邊框，Pygame 的線條可能會重複
        # 如果要完美無縫，通常不畫線，讓圖片自己連起來
        # 暫時保留原有的畫線邏輯，如果產生問題請再移除。
        # 由於 floor.png 是一塊帶有邊緣的圖，這裡不應該再畫線。
        
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
            pygame.event.pump() 
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def get_valid_moves(self):
        return np.where(self.board.flatten() == 0)[0]

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()