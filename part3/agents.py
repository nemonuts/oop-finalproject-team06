from abc import ABC, abstractmethod
import numpy as np
import random

class BaseAgent(ABC):
    """
    【抽象類別 BaseAgent】
    定義所有五子棋 AI 的共同介面。
    所有的 AI 都必須繼承這個類別，並實作 choose_action 方法。
    這展現了 OOP 的「抽象 (Abstraction)」與「繼承 (Inheritance)」。
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def choose_action(self, board, valid_moves):
        """
        輸入目前的棋盤狀態和合法步數，回傳決定要下的位置。
        """
        pass

class RandomAgent(BaseAgent):
    """
    【隨機 AI】
    完全不思考，從合法步數中隨機選一個。
    適合用來測試，或是當作初學者的對手。
    """
    def choose_action(self, board, valid_moves):
        return random.choice(valid_moves)

class GreedyAgent(BaseAgent):
    """
    【貪婪 AI】
    這是一個稍微聰明的 AI，具備基本的攻守邏輯。
    這展現了 OOP 的「多型 (Polymorphism)」，雖然都是 Agent，但思考方式不同。
    
    策略優先順序：
    1. 進攻：如果下一步能贏，立刻下！
    2. 防守：如果對手下一步會贏，立刻擋！
    3. 中間：如果都沒危險，優先搶佔棋盤中心。
    4. 隨機：剩下就隨便下。
    """
    def __init__(self, name, board_size, win_streak):
        super().__init__(name)
        self.board_size = board_size
        self.win_streak = win_streak

    def choose_action(self, board, valid_moves):
        # 1. 進攻檢查：檢查自己 (AI) 是否能一步致勝
        # 簡單判斷：如果棋盤上 1 比較多，那 AI 就是 2；反之 AI 是 1 (假設黑先手)
        my_id = 2 if np.sum(board == 1) > np.sum(board == 2) else 1
        
        winning_move = self.find_winning_move(board, valid_moves, my_id)
        if winning_move is not None:
            # print(f"[{self.name}] 發現致勝機會！進攻！")
            return winning_move

        # 2. 防守檢查：檢查對手 (人類/敵方 AI) 是否快贏了
        opponent_id = 3 - my_id # 1變2，2變1
        blocking_move = self.find_winning_move(board, valid_moves, opponent_id)
        if blocking_move is not None:
            # print(f"[{self.name}] 發現危險！防守！")
            return blocking_move

        # 3. 策略：優先下在最中間 (天元)
        center = self.board_size // 2
        center_move = center * self.board_size + center
        if center_move in valid_moves:
            return center_move

        # 4. 如果都沒事做，就隨機下
        return random.choice(valid_moves)

    def find_winning_move(self, board, valid_moves, player_id):
        """
        輔助方法：模擬下棋
        嘗試每一個合法的步，看看下在那裡會不會連成線。
        """
        for move in valid_moves:
            r, c = move // self.board_size, move % self.board_size
            
            # 假裝下在這裡
            board[r, c] = player_id
            
            # 檢查是否連線
            if self.check_win_simulation(board, r, c, player_id):
                board[r, c] = 0 # 記得復原棋盤！
                return move
            
            # 復原棋盤，試下一個位置
            board[r, c] = 0
            
        return None

    def check_win_simulation(self, board, row, col, player):
        """
        檢查在 (row, col) 落子後是否達成連線。
        這段邏輯跟環境 (Env) 裡的判斷是一樣的，但這裡是給 AI 腦袋裡模擬用的。
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # 橫, 直, 右斜, 左斜

        for dr, dc in directions:
            count = 1
            # 正向延伸檢查
            for i in range(1, self.win_streak):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            # 反向延伸檢查
            for i in range(1, self.win_streak):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_streak:
                return True
        return False