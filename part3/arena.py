# arena.py - 最終修復版 (適用於 Pygame 環境)

import time
# 引用您的五子棋環境檔 (請確保您的環境檔名是 oop_project_env.py)
# 假設您已將 tempCodeRunnerFile.py 重新命名為 oop_project_env.py
from oop_project_env import GomokuEnv 

class GomokuArena:
    """
    【競技場類別 GomokuArena】
    負責管理兩個 AI 之間的對戰流程。
    """
    # 移除 master 參數
    def __init__(self, agent1, agent2, board_size=9, win_streak=5, render=True):
        
        # 直接呼叫 GomokuEnv (現在是 Pygame 版本)
        self.env = GomokuEnv(
            board_size=board_size, 
            win_streak=win_streak, 
            render_mode='human' if render else None
        )
            
        self.agent1 = agent1
        self.agent2 = agent2
        self.render = render

    def play_match(self, delay=0.5): # <-- 關鍵修正：恢復 play_match 函式！
        """
        開始一場比賽
        delay: 每步暫停的秒數，方便人類觀看
        """
        obs, _ = self.env.reset()
        terminated = False
        
        print(f"--- 比賽開始: {self.agent1.name} (黑棋 ●) vs {self.agent2.name} (白棋 ○) ---")
        if self.render:
            self.env.render()

        while not terminated:
            # 1. 判斷現在輪到誰 (環境裡的 current_player 是 1 或 2)
            if self.env.current_player == 1:
                current_agent = self.agent1
            else:
                current_agent = self.agent2
            
            # 2. 獲取合法步數
            valid_moves = self.env.get_valid_moves()
            
            # 3. AI 思考決定下一步
            action = current_agent.choose_action(self.env.board, valid_moves)
            
            # 4. 執行動作 (下子)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 5. 顯示棋盤與資訊
            if self.render:
                row = action // self.env.board_size
                col = action % self.env.board_size
                print(f"\n[{current_agent.name}] 下在 ({row}, {col})")
                self.env.render()
                time.sleep(delay) # 暫停一下方便觀看

        # 6. 遊戲結束，宣佈結果
        winner_id = info.get("winner", 0)
        print("\n" + "="*30)
        if winner_id == 1:
            print(f"🏆 獲勝者是: {self.agent1.name} (黑棋)！")
        elif winner_id == 2:
            print(f"🏆 獲勝者是: {self.agent2.name} (白棋)！")
        else:
            print("🤝 平手 (和局)！")
        print("="*30)
        
        # 額外：如果使用 Pygame，結束後需要呼叫 close
        self.env.close()