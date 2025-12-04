from agents import RandomAgent, GreedyAgent
from arena import GomokuArena

def main():
    print("=== 初始化五子棋對戰系統 ===")
    
    # 1. 設定遊戲參數
    # 建議先用 9x9 測試，標準是 15x15，太小會很容易平手
    BOARD_SIZE = 9  
    WIN_STREAK = 5  

    # 2. 建立兩個 AI 選手
    # 選手 1 (黑棋，先手)：使用貪婪策略 (比較聰明)
    player1 = GreedyAgent(name="SmartBot", board_size=BOARD_SIZE, win_streak=WIN_STREAK)
    
    # 選手 2 (白棋，後手)：使用隨機策略 (只會亂下)
    player2 = RandomAgent(name="RandomBot")
    
    # [進階玩法] 如果你想看兩個聰明的 AI 互打，請把上面 player2 那行刪掉，換成下面這行：
    # player2 = GreedyAgent(name="SmartBot_2", board_size=BOARD_SIZE, win_streak=WIN_STREAK)

    # 3. 建立競技場 (Arena)
    # 把兩個選手和參數傳進去，render=True 代表要在終端機畫出棋盤
    arena = GomokuArena(player1, player2, board_size=BOARD_SIZE, win_streak=WIN_STREAK, render=True)
    
    # 4. 開始比賽
    # delay=1.0 代表每走一步會暫停 1 秒，讓你有時間看清楚棋局
    arena.play_match(delay=1.0)

if __name__ == "__main__":
    main()