import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def print_success_rate(rewards_per_episode):
    """計算並印出成功率"""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False):
    # 建立 8x8 滑溜環境
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) 
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # --- 🔥 參數優化 (Optimized Params) ---
    # 1. 學習率：固定或極慢衰減。在 Stochastic 環境中，0.1 是一個經驗上很穩定的值
    learning_rate_a = 0.1
    
    # 2. Gamma：保持高數值，因為路徑很長
    discount_factor_g = 0.99
    
    # 3. Epsilon 策略：根據總回合數動態調整
    epsilon = 1         
    min_exploration_rate = 0.01
    # 讓 epsilon 在訓練進行到一半時才降到極低，確保前期充分探索
    epsilon_decay_rate = 1.0 / (episodes * 0.6) 
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                # 遇到 Q 值都一樣的情況 (例如初期)，隨機選擇以增加隨機性
                if np.all(q[state, :] == q[state, 0]):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # Q-Learning 更新公式
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        # Epsilon 衰減
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        if reward == 1:
            rewards_per_episode[i] = 1
        
        # 每 5000 回合印一次進度，方便觀察
        if (i+1) % 5000 == 0 and is_training:
            print(f"Episode {i+1}: Current Epsilon {epsilon:.4f}")

    env.close()

    # 繪圖
    sum_rewards = np.zeros(episodes)
    window_size = 500 # 增加平滑視窗大小，圖表會比較好看
    for t in range(episodes):
        start_index = max(0, t - window_size)
        sum_rewards[t] = np.sum(rewards_per_episode[start_index:(t+1)])
    
    plt.plot(sum_rewards)
    plt.title('Running Sum of Rewards')
    plt.savefig('frozen_lake8x8.png')
    
    if not is_training:
        print_success_rate(rewards_per_episode)

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        print("Training finished.")
        # 檢查最後 1000 回合的表現 (這才是真正的收斂實力)
        print("Training Last 1000 Episodes Success Rate (Reference):")
        print_success_rate(rewards_per_episode[-1000:])

if __name__ == '__main__':
    #print("--- 🚀 Start Training (15000 episodes) ---")
    #run(15000, is_training=True, render=False)
    
    print("\n--- 📊 Start Evaluation (1000 test episodes) ---")
    run(1000, is_training=False, render=False)