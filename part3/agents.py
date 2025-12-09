import random
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def choose_action(self, board, valid_moves):
        pass

class RandomAgent(BaseAgent):
    def choose_action(self, board, valid_moves):
        # 修正：檢查 NumPy 陣列是否為空
        if valid_moves.size == 0:
            return None
        return random.choice(valid_moves.tolist()) # 確保這裡使用列表

class GreedyAgent(BaseAgent):
    def __init__(self, name, board_size, win_streak):
        super().__init__(name)
        self.board_size = board_size
        self.win_streak = win_streak

    def choose_action(self, board, valid_moves):
        # 修正：檢查 NumPy 陣列是否為空
        if valid_moves.size == 0:
            return None
            
        valid_moves_list = valid_moves.tolist()
        my_id = 2 if np.sum(board == 1) > np.sum(board == 2) else 1
        
        # 1. 進攻檢查 (一步致勝)
        winning_move = self._find_winning_move(board, valid_moves_list, my_id)
        if winning_move is not None:
            return winning_move

        # 2. 防守檢查 (擋住對手致勝)
        opponent_id = 3 - my_id 
        blocking_move = self._find_winning_move(board, valid_moves_list, opponent_id)
        if blocking_move is not None:
            return blocking_move

        # 3. 策略：優先下在最中間 (天元)
        center = self.board_size // 2
        center_move = center * self.board_size + center
        if center_move in valid_moves_list:
            return center_move

        # 4. 隨機下
        return random.choice(valid_moves_list)

    def _find_winning_move(self, board, valid_moves_list, player_id):
        """ 輔助方法：模擬下棋，檢查是否能一步致勝或防守 """
        for move in valid_moves_list: # <-- 確保迭代列表
            r, c = move // self.board_size, move % self.board_size
            
            board[r, c] = player_id
            
            if self._check_win_simulation(board, r, c, player_id):
                board[r, c] = 0 # 復原
                return move
            
            board[r, c] = 0 # 復原
            
        return None

    def _check_win_simulation(self, board, row, col, player):
        """ 檢查在 (row, col) 落子後是否達成連線。 """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] 

        for dr, dc in directions:
            count = 1
            for i in range(1, self.win_streak):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, self.win_streak):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_streak:
                return True
        return False


class SmartAgent(GreedyAgent):
    def __init__(self, name, board_size, win_streak):
        super().__init__(name, board_size, win_streak)
        self.scores = {
            5: 10000000, 
            4: 100000,   
            '活三': 10000, 
            '眠三': 1000,  
            '活二': 100,
            '眠二': 10,
            1: 1,
            0: 0
        }

    def choose_action(self, board, valid_moves):
        # 修正：檢查 NumPy 陣列是否為空
        if valid_moves.size == 0: 
            return None
        
        valid_moves_list = valid_moves.tolist() # <-- 轉為列表，確保後續操作安全
        my_id = 2 if np.sum(board == 1) > np.sum(board == 2) else 1
        opponent_id = 3 - my_id 

        best_score = -99999999
        best_move = random.choice(valid_moves_list) 

        for move in valid_moves_list: # <-- 確保迭代列表
            r, c = move // self.board_size, move % self.board_size
            
            # 1. 模擬自己下子 (進攻評估)
            board[r, c] = my_id
            score = self._evaluate_board(board, my_id)
            
            # 2. 模擬對手下子 (防守評估，Minimax 概念)
            opponent_score = self._evaluate_board(board, opponent_id)
            score -= opponent_score * 0.9 

            board[r, c] = 0 # 復原
            
            if score > best_score:
                best_score = score
                best_move = move
            elif score == best_score and random.random() < 0.2:
                best_move = move

        # 確保一步必勝和一步必擋的策略優先級最高
        winning_move = self._find_winning_move(board, valid_moves_list, my_id)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_winning_move(board, valid_moves_list, opponent_id)
        if blocking_move is not None:
            return blocking_move
            
        return best_move

    def _evaluate_board(self, board, player_id):
        total_score = 0
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] == player_id:
                    total_score += self._get_position_score(board, r, c, player_id)
                    
        return total_score

    def _get_position_score(self, board, r, c, player_id):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] 
        score = 0
        opponent_id = 3 - player_id
        
        for dr, dc in directions:
            for i in range(-self.win_streak + 1, 1):
                start_r, start_c = r + dr * i, c + dc * i
                
                if not (0 <= start_r < self.board_size and 0 <= start_c < self.board_size):
                    continue

                line = []
                for k in range(self.win_streak):
                    check_r, check_c = start_r + dr * k, start_c + dc * k
                    
                    if 0 <= check_r < self.board_size and 0 <= check_c < self.board_size:
                        line.append(board[check_r, check_c])
                    else:
                        line.append(-1) 

                score += self._evaluate_line(line, player_id, opponent_id)

        return score
        
    def _evaluate_line(self, line, player_id, opponent_id):
        if -1 in line:
            return 0
        
        mine = line.count(player_id)
        opponent = line.count(opponent_id)
        empty = line.count(0)
        
        if opponent > 0 and mine > 0:
            return 0

        if opponent > 0:
            return 0 
        
        if mine == 0:
            return 0 

        if mine == self.win_streak:
            return self.scores[self.win_streak]

        if mine > 0 and (mine + empty) >= self.win_streak:
            
            if mine == 4:
                if empty == 1:
                    return self.scores[4] 
                
            elif mine == 3:
                if empty >= 2:
                    return self.scores['活三'] 

            elif mine == 2:
                if empty >= 3:
                    return self.scores['活二']

            elif mine == 1:
                if empty >= 4:
                    return self.scores[1]
        
        return 0