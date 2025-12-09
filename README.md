# OOP final project 第六組
## 1.Project Overview :

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
#### 先確定目前位置在part2資料夾中
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