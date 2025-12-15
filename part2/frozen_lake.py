import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# ---------------------------------------------------------
# 輔助函式：計算並顯示成功率
# ---------------------------------------------------------
def print_success_rate(rewards_per_episode):
    """
    計算 Agent 的勝率。
    原理：將所有回的獎勵加總（成功為1，失敗為0），除以總回合數。
    """
    total_episodes = len(rewards_per_episode) # 總回合數
    success_count = np.sum(rewards_per_episode) #計算成功了幾場（把陣列裡的 1 全部加起來）
    success_rate = (success_count / total_episodes) * 100 # 計算成功率百分比
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

# ---------------------------------------------------------
# 主程式：執行訓練或測試
# episodes: 總回合數
# is_training: True 代表訓練模式(會更新Q表)，False 代表測試模式(只讀取Q表)
# render: 是否要畫出畫面 (測試時通常設為 True)
# ---------------------------------------------------------
def run(episodes, is_training=True, render=False):
    
    # 定義檔案名稱：分開儲存「地圖」與「Q-table(大腦)」
    map_filename = 'frozen_lake_map.pkl'
    q_table_filename = 'frozen_lake8x8.pkl'

    # --- 1. 地圖處理邏輯 (重要！) ---
    # 原因：因為使用隨機地圖，訓練與測試必須是「同一張地圖」。
    # 如果不存檔，訓練完後測試時會生成一張新地圖，導致原本訓練好的 Agent 撞牆。
    if is_training:
        # 訓練時：生成一張 8x8 的隨機地圖 (p=0.8 代表 80% 是冰面，20% 是洞)
        map_desc = generate_random_map(size=8, p=0.9)
        # 將地圖存檔，供測試時使用
        with open(map_filename, 'wb') as f:
            pickle.dump(map_desc, f)
        print("New random map generated and saved.")
    else:
        # 測試時：嘗試讀取之前訓練用的地圖
        try:
            with open(map_filename, 'rb') as f:
                map_desc = pickle.load(f)
            print("Loaded map from training.")
        except FileNotFoundError:
            # 防呆機制：如果找不到地圖檔，只好生成新的 (但這會導致測試結果很差)
            map_desc = generate_random_map(size=8, p=0.8)
            print("Warning: Map file not found, generated new one.")

    # --- 2. 建立環境 ---
    # is_slippery=True: 地板會滑。你選「向右」，實際上可能「向右、向上、或向下」。
    # 這增加了環境的隨機性，需要更保守的學習率。
    env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=True, render_mode='human' if render else None)

    # --- 3. 初始化 Q-table ---
    if(is_training):
        # 訓練時：建立一個全為 0 的表格 (64個狀態 x 4個動作)
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # 測試時：讀取已經訓練好的 Q-table
        f = open(q_table_filename, 'rb')
        q = pickle.load(f)
        f.close()

    # --- 4. 超參數設定 (Hyperparameters) ---
    # Learning Rate (Alpha): 設為 0.1。
    # 因為是 Slippery 環境，結果有隨機性，不能太相信單次結果 (0.9太高)，0.1 能取長期平均。
    learning_rate_a = 0.1   
    
    # Discount Factor (Gamma): 設為 0.99。
    # 因為獎勵只有在終點才有 (1)，路途中都是 0，我們需要 Agent 極度重視未來的潛在獎勵。
    discount_factor_g = 0.99 
    
    # Epsilon (探索率): 控制「隨機亂走」的機率。
    # 訓練初期設為 1 (100% 隨機)，隨時間慢慢減少。
    epsilon = 1
    
    # Decay Rate: 控制 Epsilon 降多快。
    # 設定為 1 / (總回合數 * 0.8)，確保在前 80% 的時間裡還有機會探索，最後 20% 才專注衝刺。
    epsilon_decay_rate = 1 / (episodes * 0.8) if is_training else 0 
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    # --- 5. 訓練/測試 迴圈開始 ---
    for i in range(episodes):
        # 重置環境，回到起點 (state 0)
        state = env.reset()[0]
        terminated = False      # 是否掉進洞或到達終點
        truncated = False       # 是否步數過多被強制結束

        while(not terminated and not truncated):
            # --- [關鍵邏輯] 動作選擇 (Epsilon-Greedy) ---
            
            # 情況 A: 探索 (Exploration) - 隨機選一個動作
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            
            # 情況 B: 利用 (Exploitation) - 根據 Q-table 選最好的
            else:
                # 原始寫法: action = np.argmax(q[state,:])
                # 問題：當 Q 值都是 0 (訓練初期) 或有多個相同最大值時，argmax 永遠回傳索引 0 (向左)。
                # 這會導致 Agent 一直撞左邊的牆，無法探索其他方向。
                
                # 改良寫法 (Tie-breaking)：
                max_q_value = np.max(q[state, :]) # 找出目前最大的 Q 值
                actions_with_max_q = np.where(q[state, :] == max_q_value)[0] # 找出所有擁有最大值的動作索引 
                action = rng.choice(actions_with_max_q) # 從這些最好的動作中「隨機」選一個

            # 執行動作，觀察環境回饋
            new_state, reward, terminated, truncated, _ = env.step(action)

            # --- [核心演算法] Q-Learning 更新公式 ---
            if is_training:
                # Q(s,a) = Q(s,a) + alpha * [ Reward + gamma * max(Q(s',a')) - Q(s,a) ]
                # 意義：新的Q值 = 舊Q值 + 學習率 * (目標值 - 舊Q值)
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            # 更新狀態，準備走下一步
            state = new_state

        # 每個 Episode 結束後，減少 Epsilon (減少隨機探索，增加對 Q-table 的依賴)
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # 紀錄是否成功 (FrozenLake 只有到達終點才有 Reward 1)
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    # --- 6. 繪圖與存檔 ---
    # 計算滑動平均 (Moving Average) 以平滑化曲線
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    if is_training:
        # 繪製學習曲線
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake8x8.png')
        
        # 儲存 Q-table (大腦)
        with open(q_table_filename, "wb") as f:
            pickle.dump(q, f)
        print(f"Training finished. Map saved to {map_filename}. Q-table saved to {q_table_filename}.")
    
    if not is_training:
        print_success_rate(rewards_per_episode)

if __name__ == '__main__':
    # 1. 訓練階段 (Training)
    # 跑 15000 次，不渲染畫面 (加速)，更新 Q 表
    print("--- Starting Training ---")
    run(15000, is_training=True, render=False)
    
    # 2. 測試階段 (Testing)
    # 跑 1000 次，不渲染畫面 (計算勝率用)，不更新 Q 表，epsilon=0 (純利用)
    print("\n--- Starting Testing ---")
    run(1000, is_training=False, render=False)