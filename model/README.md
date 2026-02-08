# PolicyValueNet (PyTorch) 技术文档

## 1. 概述

该模块基于 PyTorch 框架实现了 AlphaZero 的核心神经网络。它是一个**双头网络 (Dual-headed Network)**，共享底层的卷积特征提取层，然后分叉为两个输出头：

1. **策略头 (Policy Head)**：输出下一步的策略分布(即概率分布)
2. **价值头 (Value Head)**：评估当前局面的胜率（-1 到 1）

这种设计允许网络同时进行两个任务，从而提高了模型的样本效率和泛化能力。

---

## 2. 网络架构详解

### 2.1 输入格式

网络输入是一个 `(batch_size, 4, width, height)` 的张量。

输入特征的 4 个通道通常代表：
1. **通道 0**：当前玩家的棋子位置
2. **通道 1**：对手玩家的棋子位置
3. **通道 2**：对手上一步落子的位置（聚集对手最新落子->1个1，63个0）
4. **通道 3**：当前玩家的标识（是否先手->全0或者全1）

### 2.2 共享卷积层 (Common Layers)

用于提取棋盘的空间特征。经过这些层后，网络学到了高度抽象的棋盘模式。

| 层 | 输入 | 输出 | Kernel | Padding | 激活 |
|---|---|---|---|---|---|
| Conv1 | 4 channels | 32 filters | 3×3 | 1 | ReLU |
| Conv2 | 32 filters | 64 filters | 3×3 | 1 | ReLU |
| Conv3 | 64 filters | 128 filters | 3×3 | 1 | ReLU |

**输出形状**：`(batch_size, 128, width, height)`

### 2.3 策略头 (Action/Policy Head)

用于预测每个合法动作的概率分布。

```
共享特征 (128, width, height)
    ↓
Conv 1×1 (4 filters) → ReLU
    ↓
Flatten → 4 × width × height
    ↓
FC Layer: (4×W×H) → (W×H)
    ↓
LogSoftmax → 动作概率
```

**关键特性**：
- 使用 `1×1 卷积` 进行降维，减少参数数量
- 最后一层使用 `LogSoftmax` 激活（与 NLLLoss 配合使用）
- 输出为每个合法位置的落子概率

### 2.4 价值头 (Value Head)

用于预测当前局面的评分（从 -1 到 1）。

```
共享特征 (128, width, height)
    ↓
Conv 1×1 (2 filters) → ReLU
    ↓
Flatten → 2 × width × height
    ↓
FC Layer: (2×W×H) → 64 → ReLU
    ↓
FC Layer: 64 → 1
    ↓
Tanh → 局面评分 [-1, 1]
```

**关键特性**：
- 使用两层全连接层进行预测
- 最后使用 `Tanh` 激活将输出压缩到 [-1, 1] 区间
  - **1**：当前玩家必胜
  - **-1**：当前玩家必负
  - **0**：局面均衡或平局

---

## 3. 封装类 `PolicyValueNet`

这个类封装了 `Net`，提供了与 MCTS 和训练管线交互的接口。

### 3.1 初始化 (`__init__`)

```python
PolicyValueNet(board_width, board_height, model_file=None)
```

**功能**：
- 初始化 PyTorch 模型 (`Net`)
- 自动检测并使用 GPU (cuda) 或 CPU
- 定义优化器：**Adam**（带有 L2 正则化 `weight_decay`）

**参数**：
- `board_width`, `board_height`: 棋盘尺寸
- `model_file`: 可选，预训练模型文件路径

### 3.2 核心方法

#### `policy_value(state_batch)`

```python
act_probs, value = policy_value(state_batch)
```

**功能**：批量推理（通常用于训练阶段）

**参数**：
- `state_batch`: 一批棋盘状态 (numpy array，形状 `(batch_size, 4, width, height)`)

**返回值**：
- `act_probs`: 动作概率，形状 `(batch_size, width*height)`
- `value`: 局面评分，形状 `(batch_size,)`

**计算过程**：
1. 将 numpy 数组转换为 PyTorch Tensor
2. 送入模型进行前向传播
3. 返回 softmax 概率和 tanh 评分

---

#### `policy_value_fn(board)` ⭐ **MCTS 专用接口**

```python
action_probs, leaf_value = policy_value_fn(board)
```

**功能**：为 MCTS 搜索提供策略指导

**参数**：
- `board`: 当前的 `Board` 对象

**逻辑流程**：
1. **获取合法动作**：从 `board.availables` 获取所有可行的落子位置
2. **状态转换**：调用 `board.current_state()` 获取当前棋盘状态
3. **网络推理**：将状态送入神经网络
4. **概率过滤**：
   - 获取所有 `width*height` 个位置的概率
   - 仅保留合法动作的概率
   - 重新归一化概率（因为有些位置已被占据）
5. **返回结果**：`(action, probability)` 的列表 和 `leaf_value`

**输出**：
- `action_probs`: `[(action_id, prob), ...]` 列表，仅包含合法动作
- `leaf_value`: 从当前玩家视角的局面评分

**示例**：
```python
# 棋盘状态：8×8 的五子棋
action_probs, value = policy_value_fn(board)
# action_probs = [(5, 0.15), (12, 0.08), (23, 0.05), ...]
# value = 0.23  # 当前玩家有 61.5% 的胜率
```

---

#### `train_step(state_batch, mcts_probs, winner_batch, lr)`

```python
loss, entropy = train_step(state_batch, mcts_probs, winner_batch, lr)
```

**功能**：执行一步训练（反向传播）

**参数**：
- `state_batch`: 训练样本状态，形状 `(batch_size, 4, width, height)`
- `mcts_probs`: MCTS 搜索得出的目标概率分布（Policy 的标签），形状 `(batch_size, width*height)`
- `winner_batch`: 最终游戏胜负结果（Value 的标签），形状 `(batch_size,)`，值为 `-1.0`, `0.0`, 或 `1.0`
- `lr`: 学习率

**损失函数**：

$$\text{Loss} = (z - v)^2 - \pi^T \log(p) + c||\theta||^2$$

其中：
- $z$：真实的游戏结果（赢为 1，输为 -1，平为 0）
- $v$：网络预测的价值
- $\pi$：MCTS 搜索的目标概率分布
- $p$：网络预测的概率分布
- $c$：L2 正则化系数（防止过拟合）

**损失组成**：
1. **Value Loss**（均方误差）：让网络预测的胜率接近实际胜负
2. **Policy Loss**（负对数似然）：让网络预测的概率分布接近 MCTS 的搜索结果
3. **L2 正则化**：防止模型过拟合

**返回值**：
- `loss`: 总损失值（浮点数，用于监控训练进度）
- `entropy`: 策略熵（用于监控探索程度，越大说明网络输出越不确定）

**示例**：
```python
loss, entropy = train_step(
    state_batch,
    mcts_probs,
    winner_batch,
    lr=0.00005
)
print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")
```

---

#### `save_model(model_file)` / `load_model(model_file)`

```python
# 保存模型
policy_value_net.save_model('best_policy_8_8_5.model')

# 加载模型（在 __init__ 中指定）
policy_value_net = PolicyValueNet(8, 8, model_file='best_policy_8_8_5.model')
```

**功能**：
- **save_model**：保存模型的状态字典到文件
- **load_model**：从文件加载预训练模型

---

## 4. 数据流向图

### 推理流程 (Inference)

```
棋盘状态 (4×8×8)
    ↓
共享卷积层
    ├─────────────────┬──────────────────┐
    ↓                 ↓                  ↓
策略头          价值头          (128 filters)
    ↓                 ↓
动作概率分布    局面评分
(8×8=64维)     (1维: -1~1)
    ↓                 ↓
返回合法动作    MCTS 评估
的概率          当前局面
```

### 训练流程 (Training)

```
训练样本 (棋盘状态 + MCTS概率 + 游戏结果)
    ↓
前向传播 (Forward Pass)
    ├─ Policy Head 输出 → Policy Loss
    └─ Value Head 输出 → Value Loss
    ↓
计算总损失 + L2正则化
    ↓
反向传播 (Backward Pass)
    ↓
优化器更新参数 (Adam)
    ↓
返回损失值 & 熵
```

---

## 5. 使用示例

### 5.1 初始化模型

```python
from model.policy_value_net_pytorch import PolicyValueNet

# 创建新模型
policy_value_net = PolicyValueNet(board_width=8, board_height=8)

# 或加载预训练模型
policy_value_net = PolicyValueNet(
    board_width=8,
    board_height=8,
    model_file='best_policy_8_8_5.model'
)
```

### 5.2 与 MCTS 结合

```python
from player.MCTSPlayer import MCTSPlayer

# 创建使用神经网络的 MCTS 玩家
mcts_player = MCTSPlayer(
    policy_value_fn=policy_value_net.policy_value_fn,
    c_puct=5,
    n_playout=1000
)

# MCTS 会自动调用 policy_value_fn 进行搜索
action = mcts_player.get_action(board)
```

### 5.3 训练流程

```python
import random

# 假设有训练数据
train_data = load_training_data()  # [(state, mcts_probs, winner), ...]

# 随机采样批次
batch = random.sample(train_data, batch_size=512)
state_batch = [data[0] for data in batch]
mcts_probs_batch = [data[1] for data in batch]
winner_batch = [data[2] for data in batch]

# 训练一步
loss, entropy = policy_value_net.train_step(
    state_batch,
    mcts_probs_batch,
    winner_batch,
    lr=0.00005
)

print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")

# 定期保存模型
if iteration % 100 == 0:
    policy_value_net.save_model(f'checkpoint_{iteration}.model')
```

---

## 6. 性能优化建议

1. **批量处理**：使用较大的 `batch_size`（如 512）以充分利用 GPU
2. **学习率衰减**：可根据训练进度动态调整学习率
3. **早停 (Early Stopping)**：监控验证集损失，防止过拟合
4. **数据增强**：利用棋盘的旋转和翻转对称性增加训练数据
5. **模型检查点**：定期保存模型，以便恢复训练

---

## 7. 常见问题

**Q: 为什么使用 LogSoftmax + NLLLoss 而不是 Softmax + CrossEntropyLoss？**
- 答：在 PyTorch 中，这两种方式在数值上等价，但 LogSoftmax 更高效（避免了冗余计算）

**Q: Value Head 的输出范围为什么是 [-1, 1]？**
- 答：这样可以直接表示胜率：1 表示必胜，-1 表示必负，0 表示平手或均衡局面

**Q: 如何在 CPU 上训练？**
- 答：模型会自动检测 GPU 可用性。如果不可用，会自动使用 CPU（但速度会明显降低）

**Q: 如何加载旧版本的模型文件？**
- 答：确保模型结构与旧版本兼容，然后直接调用 `load_model()`

---

## 8. 参考资源

- AlphaGo Zero: https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge
- PyTorch 官方文档: https://pytorch.org/docs/
- 卷积神经网络 (CNN) 基础: https://cs231n.github.io/

---

**最后更新**：2026年2月
**作者**：Han Feng Yuan Mai

