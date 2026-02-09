# train.py 逻辑梳理文档

## 概述

`train.py` 是 AlphaZero 五子棋训练的核心模块，实现了完整的强化学习训练流程。该模块通过自我对弈生成训练数据，使用策略价值网络进行学习，并定期评估模型性能。

## 核心类：TrainPipeline

### 1. 初始化参数 (`__init__`)

#### 棋盘参数
- `board_width`: 棋盘宽度，默认为 10
- `board_height`: 棋盘高度，默认为 10

#### 训练超参数
- `learn_rate`: 学习率，设置为 `5e-5`（原始值为 `2e-3`）
- `lr_multiplier`: 学习率自适应调整系数，初始为 `1.0`
- `temp`: 温度参数，softmax使用(访问->概率转化)，默认为 `1.0`
- `n_playout`: 每步棋的 MCTS 模拟次数，设置为 `1000`（原始值为 `400`），控制搜索广度
- `c_puct`: MCTS 探索系数，默认为 `5`
- `buffer_size`: 数据缓冲区大小，默认为 `10000`
- `batch_size`: 训练时的 mini-batch 大小，默认为 `512`
- `play_batch_size`: 每次收集数据的自我对弈局数，默认为 `1`
- `epochs`: 每次更新时的训练轮数，默认为 `5`
- `kl_targ`: KL 散度目标值，用于自适应调整学习率，默认为 `0.02`
- `check_freq`: 模型评估频率（每多少轮自我对弈进行一次评估），设置为 `50`
- `game_batch_num`: 总共进行的自我对弈轮数，设置为 `2000`（原始值为 `1500`）

#### 评估参数
- `best_win_ratio`: 记录最佳胜率，初始为 `0.0`
- `pure_mcts_playout_num`: 纯 MCTS 对手的模拟次数，默认为 `1000`
- `is_shown_pygame`: 是否显示/刷新 pygame 界面，默认为 `1`（显示）

#### 数据结构
- `data_buffer`: 使用 `deque` 实现的训练数据缓冲区，最大长度为 `buffer_size`

#### 模型和游戏对象
- `policy_value_net`: 策略价值网络，可以从头训练或基于已有模型继续训练
- `game`: 游戏环境对象，负责执行对弈和管理游戏状态

---

## 2. 核心方法

### 2.1 数据增强 (`get_aug_data`)

**功能**: 通过旋转和翻转对训练数据进行扩充，增加数据多样性。

**输入**:
- `play_data`: 原始对弈数据，格式为 `[(state, mcts_prob, winner_z), ...]`
  - `state`: 棋盘状态
  - `mcts_prob`: MCTS 搜索得到的动作概率分布
  - `winner`: 游戏结果（+1/-1/0）

**处理过程**:
1. 对每个状态进行 4 次旋转（90°、180°、270°、360°）
2. 对每次旋转后的状态进行水平翻转
3. 同步变换对应的 MCTS 概率分布
4. 保持游戏结果不变

**输出**:
- `extend_data`: 扩充后的数据，数量是原始数据的 8 倍

**意义**: 利用五子棋的对称性，从一局游戏生成 8 倍的训练样本，提高数据利用效率。

---

### 2.2 收集自我对弈数据 (`collect_selfplay_data`)

**功能**: 让 AI 与自己对弈，收集训练数据。

**参数**:
- `n_games`: 自我对弈的局数，默认为 `1`
- `visualize_playout`: 是否启用 playout 推演可视化（会显著降低速度），默认为 `False`
- `playout_delay`: 每次 playout 可视化后的延迟时间（秒），默认为 `0.5`

**流程**:
1. 循环进行 `n_games` 局自我对弈
2. 调用 `game.start_self_play()` 进行一局完整的自我对弈
3. 获取对弈数据（包括每步的状态、MCTS 概率分布和最终结果）
4. 记录本局的步数 `episode_len`
5. 对数据进行增强（调用 `get_aug_data`）
6. 将增强后的数据添加到 `data_buffer` 中

**数据格式**: 每局游戏生成的数据经过增强后，会产生大量的训练样本存入缓冲区。

---

### 2.3 策略更新 (`policy_update`)

**功能**: 使用收集到的数据更新策略价值网络。

**流程**:

#### 第一步：采样准备
1. 从 `data_buffer` 中随机采样 `batch_size` 个样本
2. 分离出状态批次、MCTS 概率批次和结果批次
3. 获取更新前的策略概率 `old_probs` 和价值预测 `old_v`

#### 第二步：训练迭代
1. 进行 `epochs` 轮训练（默认 5 轮）
2. 每轮调用 `policy_value_net.train_step()` 更新网络参数
3. 计算损失 `loss` 和熵 `entropy`
4. 获取更新后的策略概率 `new_probs` 和价值预测 `new_v`
5. 计算 KL 散度，监控策略变化幅度
6. **早停机制**: 如果 KL 散度 > `kl_targ * 4`，提前终止训练，防止策略变化过大

#### 第三步：学习率自适应调整
- 如果 KL 散度 > `kl_targ * 2` 且 `lr_multiplier > 0.1`，则 `lr_multiplier /= 1.5`（降低学习率）
- 如果 KL 散度 < `kl_targ / 2` 且 `lr_multiplier < 10`，则 `lr_multiplier *= 1.5`（提高学习率）

#### 第四步：性能指标计算
- `explained_var_old`: 更新前的价值网络解释方差（衡量价值预测的准确性）
- `explained_var_new`: 更新后的价值网络解释方差

**输出**:
- 打印训练指标：KL 散度、学习率倍数、损失、熵、解释方差等
- 返回 `loss` 和 `entropy`

**意义**: 通过梯度下降优化策略价值网络，使其更好地拟合 MCTS 搜索结果和实际对弈结果。

---

### 2.4 策略评估 (`policy_evaluate`)

**功能**: 通过与纯 MCTS 对手对弈，评估当前策略的强度。

**参数**:
- `n_games`: 评估对局数，默认为 `10`

**流程**:
1. 创建当前策略的 MCTS 玩家 `current_mcts_player`
2. 创建纯 MCTS 玩家 `pure_mcts_player`（无神经网络辅助）
3. 进行 `n_games` 局对弈
4. 随机选择先后手（概率各 50%）
5. 统计胜负平局数
6. 计算胜率：`win_ratio = (胜局数 + 0.5 * 平局数) / 总局数`

**输出**:
- 打印评估结果：纯 MCTS 的模拟次数、胜负平局数
- 返回胜率 `win_ratio`

**意义**: 提供一个客观的基准来衡量模型的进步，纯 MCTS 是一个稳定的参照对手。

---

### 2.5 训练主循环 (`run`)

**功能**: 执行完整的训练流程，整合数据收集、模型更新和评估。

**流程**:

#### 初始化
1. 启动训练计时器

#### 主循环（迭代 `game_batch_num` 次）
对于每一轮迭代 `i`:

**步骤 1: 数据收集**
- 调用 `collect_selfplay_data()` 进行自我对弈
- 获取可视化配置参数（`visualize_playout` 和 `playout_delay`）
- 打印当前批次和对局步数

**步骤 2: 模型更新**
- 如果 `data_buffer` 中的样本数 > `batch_size`，则调用 `policy_update()` 更新模型

**步骤 3: 定期评估和保存**
- 每 `check_freq` 轮（默认 50 轮）进行一次评估
- 调用 `policy_evaluate()` 评估当前模型
- 保存当前模型为 `current_policy_{width}_{width}_5.model`
- 如果胜率超过历史最佳 `best_win_ratio`:
  - 更新 `best_win_ratio`
  - 保存最佳模型为 `best_policy_{width}_{width}_5_realGood.model`
  - **难度递增机制**: 如果胜率达到 100% 且纯 MCTS 模拟次数 < 5000，则增加纯 MCTS 的模拟次数（+1000），并重置 `best_win_ratio` 为 0

#### 异常处理
- 捕获 `KeyboardInterrupt`，允许用户手动中断训练

---

## 3. 主程序入口 (`__main__`)

### 配置选项

#### 初始化模型
```python
# 从头开始训练
training_pipeline = TrainPipeline()

# 或者基于已有模型继续训练（注释掉的代码）
# training_pipeline = TrainPipeline(init_model='best_policy_8_8_5_realGood.model')
```

#### 可视化配置（可选）
```python
training_pipeline.visualize_playout = False  # 是否启用推演过程可视化
training_pipeline.playout_delay = 2          # 推演可视化延迟时间（秒）
```

**注意**:
- 推演可视化会显著降低训练速度
- 搜索树展示有两个粒度：
  1. 推演前后的展示
  2. 推演过程的展示（这里配置的是后者）

#### 启动训练
```python
training_pipeline.run()
```

#### 输出训练时间
```python
print("训练时间: {}s".format(training_pipeline.game.get_training_time_str()))
```

---

## 4. 训练流程图

```
开始
  ↓
初始化 TrainPipeline
  ↓
启动训练计时器
  ↓
┌──────────────────────────────────┐
│ 主循环 (game_batch_num 轮)       │
│                                  │
│  ┌────────────────────────────┐ │
│  │ 1. 自我对弈收集数据         │ │
│  │    - AI vs AI              │ │
│  │    - 数据增强 (8倍)        │ │
│  │    - 存入 data_buffer      │ │
│  └────────────────────────────┘ │
│           ↓                      │
│  ┌────────────────────────────┐ │
│  │ 2. 模型更新                 │ │
│  │    - 采样 mini-batch       │ │
│  │    - 梯度下降训练           │ │
│  │    - KL 散度监控           │ │
│  │    - 学习率自适应调整       │ │
│  └────────────────────────────┘ │
│           ↓                      │
│  ┌────────────────────────────┐ │
│  │ 3. 定期评估 (每 50 轮)     │ │
│  │    - vs 纯 MCTS            │ │
│  │    - 保存当前模型           │ │
│  │    - 保存最佳模型           │ │
│  │    - 难度递增机制           │ │
│  └────────────────────────────┘ │
│           ↓                      │
│      回到步骤 1                  │
└──────────────────────────────────┘
  ↓
输出训练时间
  ↓
结束
```

---

## 5. 关键技术点

### 5.1 自我对弈强化学习
- **思想**: AI 通过与自己对弈不断学习和进化
- **优势**: 不需要人类棋谱，能够发现新的策略
- **实现**: 使用 MCTS + 神经网络的混合方法

### 5.2 数据增强
- **方法**: 旋转和翻转
- **效果**: 将数据量扩充 8 倍，充分利用对称性
- **意义**: 提高数据效率，加速训练收敛

### 5.3 经验回放 (Experience Replay)
- **实现**: 使用 `deque` 作为循环缓冲区
- **容量**: 10000 个样本
- **采样**: 随机采样 mini-batch 进行训练
- **作用**: 打破数据时序相关性，提高训练稳定性

### 5.4 KL 散度约束
- **目的**: 防止策略更新过快导致训练不稳定
- **机制**:
  - 早停：KL > 4 * kl_targ
  - 学习率调整：根据 KL 散度动态调整
- **效果**: 保证训练平稳进行

### 5.5 课程学习 (Curriculum Learning)
- **实现**: 动态调整纯 MCTS 对手的难度
- **触发条件**: 当前模型对纯 MCTS (1000 次模拟) 胜率达到 100%
- **调整方式**: 增加纯 MCTS 模拟次数 (+1000)，最高到 5000
- **意义**: 逐步提高对手难度，避免过拟合

### 5.6 自适应学习率
- **基础学习率**: 5e-5
- **调整倍数**: lr_multiplier (0.1 ~ 10)
- **调整依据**: KL 散度
  - 策略变化太大 → 降低学习率
  - 策略变化太小 → 提高学习率
- **效果**: 平衡训练速度和稳定性

---

## 6. 训练监控指标

### 训练过程指标
- **KL 散度**: 衡量策略更新幅度
- **Loss**: 总体损失函数值
- **Entropy**: 策略熵，衡量探索程度
- **Explained Variance**: 价值网络的拟合质量
- **Learning Rate Multiplier**: 当前学习率倍数
- **Episode Length**: 每局对弈的步数

### 评估指标
- **Win Ratio**: 对纯 MCTS 的胜率
- **Best Win Ratio**: 历史最佳胜率
- **Pure MCTS Playout Num**: 纯 MCTS 对手的模拟次数（难度）

---

## 7. 模型保存策略

### 两种模型文件
1. **当前模型** (`current_policy_{width}_{width}_5.model`)
   - 保存频率：每 `check_freq` 轮（50 轮）
   - 用途：记录训练进度，可用于恢复训练

2. **最佳模型** (`best_policy_{width}_{width}_5_realGood.model`)
   - 保存条件：胜率超过历史最佳
   - 用途：保存最强的模型用于实战

### 断点续训
- 通过 `init_model` 参数加载已保存的模型
- 可以在中断后继续训练

---

## 8. 性能优化建议

### 训练速度
- **关闭可视化**: `is_shown_pygame = 0`
- **禁用推演可视化**: `visualize_playout = False`
- **调整 playout 次数**: 降低 `n_playout` 可加速训练，但可能影响质量

### 训练效果
- **增加 buffer_size**: 更多样化的训练数据
- **调整 batch_size**: 更大的 batch 可能更稳定
- **增加 epochs**: 每次更新训练更充分
- **调整 n_playout**: 更多模拟次数产生更高质量的训练数据

### 超参数调整建议
```python
# 快速实验配置（低质量，快速验证）
n_playout = 400
pure_mcts_playout_num = 500
check_freq = 20

# 标准训练配置（当前配置）
n_playout = 1000
pure_mcts_playout_num = 1000
check_freq = 50

# 高质量训练配置（慢速，高质量）
n_playout = 1600
pure_mcts_playout_num = 2000
check_freq = 100
```

---

## 9. 代码依赖关系

```
train.py
  ├── env/game.py               # 游戏环境
  ├── player/MCTSPlayer.py      # MCTS + 神经网络玩家
  ├── player/mcts_pure.py       # 纯 MCTS 玩家（评估基准）
  └── model/policy_value_net_pytorch.py  # 策略价值网络（PyTorch 实现）
```

---

## 10. 使用示例

### 从头开始训练
```python
python train.py
```

### 基于已有模型继续训练
```python
# 修改 __main__ 中的代码
training_pipeline = TrainPipeline(init_model='best_policy_10_10_5_realGood.model')
training_pipeline.run()
```

### 启用推演可视化（调试用）
```python
training_pipeline = TrainPipeline()
training_pipeline.visualize_playout = True
training_pipeline.playout_delay = 2
training_pipeline.run()
```

---

## 11. 常见问题

### Q1: 训练多久能得到一个可用的模型？
**A**: 取决于硬件和配置，一般需要数百到数千轮自我对弈。建议先运行 200-500 轮观察效果。

### Q2: 如何判断训练是否收敛？
**A**: 观察以下指标：
- Loss 逐渐下降并趋于稳定
- Explained Variance 接近 1
- Win Ratio 持续提高并稳定在高位

### Q3: 为什么训练速度很慢？
**A**: 可能的原因：
- 可视化开启（关闭 `is_shown_pygame` 和 `visualize_playout`）
- `n_playout` 设置过高
- CPU 性能不足（MCTS 是 CPU 密集型）

### Q4: 如何加速训练？
**A**:
- 使用更强的 CPU（MCTS 并行计算）
- 关闭所有可视化
- 适当降低 `n_playout`（但不要低于 400）
- 使用 GPU 加速神经网络训练（需要 PyTorch GPU 支持）

### Q5: 模型过拟合怎么办？
**A**:
- 增加 `buffer_size` 保留更多历史数据
- 增加数据增强的多样性
- 降低学习率
- 增加 KL 散度约束

---

## 12. 总结

`train.py` 实现了一个完整的 AlphaZero 风格的强化学习训练流程：

1. **自我对弈** 生成训练数据
2. **数据增强** 提高数据利用率
3. **经验回放** 打破数据相关性
4. **策略梯度** 优化神经网络
5. **KL 约束** 保证训练稳定性
6. **定期评估** 监控训练进度
7. **课程学习** 逐步提高难度
8. **模型保存** 记录最佳性能

这种方法能够在无人类先验知识的情况下，通过自我对弈不断提升棋力，最终达到超人水平。

