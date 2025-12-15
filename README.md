# OOP final project 第六組
## 1.Project Overview :
### 專案目標：訓練一個強化學習Agent，在8x8的冰湖滑溜環境中，成功避開冰洞並抵達終點。
### 挑戰：
#### 環境滑溜：移動動作有2/3機率隨機滑動，增加了狀態轉移的不確定性。
#### 稀疏獎勵：只有抵達終點才有分數(+1)，其餘皆為0。
### 核心方法:
#### 採用Q-Learning演算法。
#### 設計Epsilon在前60%回合緩慢衰減，後40%保持最低值以穩定收斂。
#### 修正np.argmax在Q值全為0時的預設偏差，增加初期隨機性。
### 結果:
#### 大概60~62%
### 未達70%目標的解決方法
#### 使用隨機生成地圖來讓坑洞減少
### 結果:
#### 大概0~99% (0%是因為可能生成無法到達終點的地圖)
## 2.Dependencies :
#### 請先下載zip file後照著以下步驟初始化： 
```bash
# 1. Install python 3.12
brew install python@3.12

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install 
#    Gymnasium v1.2.2
#    matplotlib
pip install "gymnasium[classic_control]==1.2.2" numpy matplotlib pygame
```

## 3.How to run :
### Part 1:
```bash
# Train the agent
python3.12 mountain_car.py --train --episodes 5000

# Render and visualize performance
python3.12 mountain_car.py --render --episodes 10
```
### Part 2:
#### 先cd到part2資料夾中
```bash
python3.12 frozen_lake.py
```
### Part 3:
#### 先cd到part3資料夾中
```bash
python main.py
```

## 4.Contribution list :
|Work|Name|Name|
|---|---|---|
|readme.md|童煜凱|楊紘鈞|
|part1|-|-|
|part2|童煜凱|-|
|part3|楊紘鈞|-|