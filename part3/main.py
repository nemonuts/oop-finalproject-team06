# main.py - 最終版本 (使用 SmartAgent 互相對戰)

from agents import RandomAgent, GreedyAgent, SmartAgent # 確保導入 SmartAgent
from arena import GomokuArena

def main():
    print("=== 初始化五子棋對戰系統 ===")

    # 1. 設定遊戲參數
    # 建議先用 9x9 測試，標準是 15x15，太小會很容易平手
    BOARD_SIZE = 9  
    WIN_STREAK = 5  

    # 2. 建立兩個 AI 選手
    # 選手 1 (黑棋，先手)：使用智慧型策略
    player1 = SmartAgent(name="AI_Black", board_size=BOARD_SIZE, win_streak=WIN_STREAK)

    # 選手 2 (白棋，後手)：使用智慧型策略
    player2 = SmartAgent(name="AI_White", board_size=BOARD_SIZE, win_streak=WIN_STREAK)

    # 3. 建立競技場 (Arena)
    # 不再傳遞 root，因為 GomokuEnv 現在是 Pygame 版本
    arena = GomokuArena(player1, player2, board_size=BOARD_SIZE, win_streak=WIN_STREAK, render=True)

    # 4. 開始比賽
    # delay=0.5 代表每走一步會暫停 0.5 秒
    arena.play_match(delay=0.5)

if __name__ == "__main__":
    main()