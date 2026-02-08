# 论文解读：Relational Recurrent Neural Networks (RMC)

## 1. 一句话概述

这篇论文提出了一种名为**关系记忆核心（Relational Memory Core, RMC）的新型递归神经网络架构，通过在记忆槽（Memory Slots）之间引入多头自注意力机制（Multi-Head Self-Attention）**，解决了传统 LSTM 在处理时序数据时难以进行显式关系推理（Relational Reasoning）的缺陷。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**核心问题**：
传统的基于记忆的神经网络（如 LSTM 或记忆增强网络 MANNs）在处理时序数据时，往往难以显式地对“记忆中存储的信息”进行复杂的**关系推理**（Relational Reasoning）。

* 
**LSTM** 倾向于将所有历史信息压缩到一个高维向量中，导致信息纠缠，难以区分实体间的关系 。


* 
**MANNs** 虽然有独立的记忆槽，但记忆之间缺乏交互机制，通常是独立的读写操作 。



**主要贡献**：

1. 
**架构创新**：提出了 Relational Memory Core (RMC)。这是一个替代传统 RNN 单元的模块，它拥有一个固定的记忆矩阵，并在每个时间步利用**多头点积注意力（MHDPA）**让记忆槽之间相互“对话”和交互 。


2. 
**理论验证**：设计了一个名为 " Farthest" 的玩具任务，证明了标准 LSTM 在需要对跨时间步的实体进行比较和排序时完全失效，而 RMC 能轻松解决 。


3. 
**SOTA 性能**：在强化学习（Mini PacMan）、程序执行（Program Evaluation）和语言建模（WikiText-103）任务上取得了显著优于 LSTM 和 DNC 的结果 。



## 3. Introduction: 论文的动机是什么？

论文的动机源于对人类认知和现有神经网络缺陷的深刻观察。

**1. 关系推理的重要性**
人类的记忆不仅仅是存储（Storage）和检索（Retrieval），更重要的是**交互（Interaction）** 。例如，要在脑海中比较两个很久前见过的人的身高，我们不仅要回忆起他们，还要在工作记忆中对这两个实体进行“关系比较”。

**2. 现有模型的局限**

* 
**LSTM 的局限**：LSTM 依靠门控机制更新细胞状态 。虽然有效，但它本质上是在做一个“加权平均”或“信息覆盖”，将所有时序信息压缩进一个向量。这使得模型很难在长时间跨度后，区分出“实体 A”和“实体 B”并计算它们的关系 。


* 
**Attention 的潜力**：Transformer 证明了 Self-Attention 是捕捉长距离依赖的神器。但在 2018 年（论文发表时），Attention 多用于非递归结构。作者思考：如果将 Attention 放入 RNN 的**每一步递归**中，让记忆在时间步内部进行自我整理，是否能赋予 RNN 逻辑推理能力？。



**3. RMC 的设计哲学**
作者提出了一种归纳偏置（Inductive Bias）：**记忆应该是矩阵形式（分槽存储实体），且记忆之间必须通过注意力机制显式交互** 。通过这种方式，模型不仅记住了信息，还能随着时间推移不断更新对这些信息之间关系的理解。

## 4. Method: 解决方案是什么？

RMC 的核心在于将传统的 hidden state  替换为一个**记忆矩阵 **（ 个记忆槽，每个大小为 ），并在递归的每一步引入 Attention。

### 4.1 核心组件：多头点积注意力 (MHDPA)

在每个时间步 `t`，模型拥有上一步记忆矩阵 `M_{t-1}` 和当前输入 `x_t`。
为了让新信息进入记忆并让记忆槽之间交互，作者将输入 `x_t` 拼接到记忆矩阵中（或作为 query/value 的一部分）。
注意力更新公式如下：

其中，Query (`Q`)、Key (`K`)、Value (`V`) 均由上一时刻记忆 `M_{t-1}` 线性投影得到。

> "Using MHDPA, each memory will attend over all of the other memories, and will update its content based on the attended information." 
> 这意味着，记忆槽 `m_i` 可以“查询”记忆槽 `m_j` 的内容，并将相关信息融合到自身更新中。
> 
> 

### 4.2 记忆更新与门控 (Gating)

得到注意力交互后的记忆  后，并不是直接覆盖原记忆，而是通过一个类似于 LSTM 的门控机制进行更新。这保证了长期记忆的稳定性。
更新方程（类似 LSTM，但应用于矩阵的每一行 ）：

这里， 分别是遗忘门、输入门和输出门 。
**关键点**：这些门控参数是**跨记忆槽共享**的（Row-wise shared weights），这意味着无论记忆矩阵多大，参数量不会爆炸 。

### 4.3 整体流程图 (Mermaid)

```mermaid
graph TD
    subgraph "Relational Memory Core (at time t)"
    Input[Input x_t] --> Concat
    PrevMem[Memory M_{t-1}] --> Concat
    Concat[Concat: [M; x]] --> MHA
    
    subgraph "Interaction Phase"
    MHA[Multi-Head Attention] --> |Queries entities| Residual
    Concat --> Residual[Residual Add]
    Residual --> MLP[Row-wise MLP]
    end
    
    subgraph "Update Phase"
    MLP --> Gates[LSTM-style Gates (Input/Forget/Output)]
    PrevMem --> Gates
    Gates --> NewMem[New Memory M_t]
    end
    end
    
    NewMem --> Output[Output h_t]
    NewMem --> NextStep[To Time t+1]

```

## 5. Experiment: 实验与结果

### 5.1  Farthest Task (Toy Task)

这是一个专门设计的压力测试。

* **任务**：输入一系列向量，最后询问：“距离向量  第  远的向量是哪个？”
* **挑战**：模型不仅要记住所有向量，还要在内存中计算它们与目标向量  的欧氏距离，并进行排序。这纯粹是**关系推理**。
* **结果**：
* LSTM / DNC 准确率：< 30%（彻底失败）。
* RMC 准确率：**91%** 。


* 分析：Attention map 显示，RMC 在看到查询向量  后，其记忆槽的注意力分布发生了显著变化，显式地去关注相关的存储实体 。





### 5.2 程序执行 (Program Evaluation)

* **任务**：预测随机生成的类 Python 代码片段的输出（包含加法、控制流、变量赋值）。
* **结果**：RMC 在"Control"（控制流）和"Program"（完整程序）任务上均优于 LSTM 和 DNC。
* Control 任务：RMC 99.6% vs DNC 83.8% 。


* 这表明 RMC 能更好地理解代码符号之间的逻辑关系。



### 5.3 语言建模 (Language Modeling)

* **数据集**：WikiText-103（大规模长文本）。
* 
**结果**：RMC 达到了当时的 SOTA 困惑度（Perplexity），比最佳 LSTM 基线降低了 1.4-5.4 。


* 
**发现**：RMC 特别擅长处理**高频词**的关系建模，且在较短的上下文窗口下就能取得很好的效果，说明其捕捉局部关系的能力很强 。



### 5.4 强化学习 (Mini PacMan)

* **任务**：部分可观测的吃豆人（带有 Viewport，不能看全图）。
* **意义**：Agent 必须在记忆中构建地图并规划路线。
* 
**结果**：RMC (677分) 显著优于 LSTM (550分) 。



## 6. Numpy 与 Torch 对照实现

### 代码说明

这份对比代码聚焦于 RMC 的核心模型定义（Model Definition）。

* **对应论文部分**：Section 3 (Model), specifically 3.1 (Attention), 3.2 (Encoding), 3.3 (Recurrence)。
* **Numpy 版本**：来源于提供的输入代码（Sections 1, 2, 3），这部分代码实现了前向传播的逻辑。为保持篇幅和阅读体验，我提取了核心架构部分（Section 1-3），去掉了纯用于教学的手写反向传播（Section 11），因为 PyTorch 会自动处理反向传播。
* **Torch 版本**：基于 Numpy 逻辑的高效实现。
* **Shape 假设**：Input `(Batch, Input_Size)`, Memory `(Batch, Slots, Head_Size * Heads)`。
* **优化点**：Numpy 代码使用了 Python `for` 循环来处理多头注意力和行级 MLP（Row-wise MLP）。在 Torch 版本中，我将其全部**向量化（Vectorized）**，利用矩阵运算同时处理所有 Head 和 Memory Slots，极大提升 GPU 效率。



::: code-group

```python [Numpy]
# ================================================================
# Section 1: Multi-Head Attention
# ================================================================
def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    Multi-head attention mechanism
    
    Args:
        X : (N, d_model) – input matrix (memory slots + current input)
        W_q, W_k, W_v: Query, Key, Value projection weights for each head
        W_o: Output projection weight
        num_heads: Number of attention heads
        mask: Optional attention mask
    
    Returns:
        output: (N, d_model) - attended output
        attn_weights: attention weights (for visualization)
    """
    N, d_model = X.shape
    d_k = d_model // num_heads
    
    heads = []
    for h in range(num_heads):
        Q = X @ W_q[h]              # (N, d_k)
        K = X @ W_k[h]              # (N, d_k)
        V = X @ W_v[h]              # (N, d_k)
        
        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(d_k)   # (N, N)
        if mask is not None:
            scores = scores + mask
        attn_weights = softmax(scores, axis=-1)
        head = attn_weights @ V           # (N, d_k)
        heads.append(head)
    
    # Concatenate all heads and project
    concatenated = np.concatenate(heads, axis=-1)   # (N, num_heads * d_k)
    output = concatenated @ W_o                     # (N, d_model)
    return output, attn_weights if num_heads == 1 else None

# ================================================================
# Section 2: Relational Memory Core
# ================================================================
class RelationalMemory:
    """
    Relational Memory Core using multi-head self-attention
    
    The memory consists of multiple slots that interact via attention,
    enabling relational reasoning between stored representations.
    """
    
    def __init__(self, mem_slots, head_size, num_heads=4, gate_style='memory'):
        assert head_size * num_heads % 1 == 0
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.d_model = head_size * num_heads
        self.gate_style = gate_style
        
        # Attention weights (one set per head)
        self.W_q = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_k = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_v = [np.random.randn(self.d_model, head_size) * 0.1 for _ in range(num_heads)]
        self.W_o = np.random.randn(self.d_model, self.d_model) * 0.1
        
        # MLP for processing attended values
        self.W_mlp1 = np.random.randn(self.d_model, self.d_model*2) * 0.1
        self.W_mlp2 = np.random.randn(self.d_model*2, self.d_model) * 0.1
        
        # LSTM-style gating per memory slot
        self.W_gate_i = np.random.randn(self.d_model, self.d_model) * 0.1  # input gate
        self.W_gate_f = np.random.randn(self.d_model, self.d_model) * 0.1  # forget gate
        self.W_gate_o = np.random.randn(self.d_model, self.d_model) * 0.1  # output gate
        
        # Initialize memory slots
        self.memory = np.random.randn(mem_slots, self.d_model) * 0.01
    
    def reset_state(self):
        """Reset memory slots to random initialization"""
        self.memory = np.random.randn(self.mem_slots, self.d_model) * 0.01
    
    def step(self, input_vec):
        """
        Update memory with new input via self-attention
        
        Args:
            input_vec: (d_model,) - new input to incorporate
        
        Returns:
            output: (d_model,) - output representation
        """
        # Append input to memory for attention
        M_tilde = np.concatenate([self.memory, input_vec[None]], axis=0)  # (mem_slots+1, d_model)
        
        # Multi-head self-attention across all slots
        attended, _ = multi_head_attention(
            M_tilde, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads)
        
        # Residual connection
        gated = attended + M_tilde
        
        # Row-wise MLP
        hidden = np.maximum(0, gated @ self.W_mlp1)  # ReLU activation
        mlp_out = hidden @ self.W_mlp2
        
        # Memory gating (LSTM-style gates for each slot)
        new_memory = []
        for i in range(self.mem_slots):
            m = mlp_out[i]
            
            # Compute gates
            i_gate = 1 / (1 + np.exp(-(m @ self.W_gate_i)))  # input gate
            f_gate = 1 / (1 + np.exp(-(m @ self.W_gate_f)))  # forget gate
            o_gate = 1 / (1 + np.exp(-(m @ self.W_gate_o)))  # output gate
            
            # Update memory slot
            candidate = np.tanh(m)
            new_slot = f_gate * self.memory[i] + i_gate * candidate
            new_memory.append(o_gate * np.tanh(new_slot))
        
        self.memory = np.array(new_memory)
        
        # Output is the last row (corresponding to input)
        output = mlp_out[-1]
        return output

# ================================================================
# Section 3: Relational RNN Cell
# ================================================================
class RelationalRNNCell:
    """
    Complete Relational RNN Cell combining LSTM and Relational Memory
    
    Architecture:
    1. LSTM processes input and produces proposal hidden state
    2. Relational memory updates based on LSTM output
    3. Combine LSTM and memory outputs
    """
    
    def __init__(self, input_size, hidden_size, mem_slots=4, num_heads=4):
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # Standard LSTM for proposal hidden state
        # Gates: input, forget, output, cell candidate
        self.lstm = np.random.randn(input_size + hidden_size, 4*hidden_size) * 0.1
        self.lstm_bias = np.zeros(4*hidden_size)
        
        # Relational memory
        self.rm = RelationalMemory(
            mem_slots=mem_slots,
            head_size=hidden_size//num_heads,
            num_heads=num_heads
        )
        
        # Combination layer (LSTM hidden + memory output)
        self.W_combine = np.random.randn(2*hidden_size, hidden_size) * 0.1
        self.b_combine = np.zeros(hidden_size)
        
        # Initialize hidden and cell states
        self.h = np.zeros(hidden_size)
        self.c = np.zeros(hidden_size)
    
    def reset_state(self):
        """Reset hidden state, cell state, and relational memory"""
        self.h = np.zeros(self.hidden_size)
        self.c = np.zeros(self.hidden_size)
        self.rm.reset_state()
    
    def forward(self, x):
        """
        Forward pass through Relational RNN cell
        
        Args:
            x: (input_size,) - input vector
        
        Returns:
            h: (hidden_size,) - output hidden state
        """
        # 1. LSTM proposal
        concat = np.concatenate([x, self.h])
        gates = concat @ self.lstm + self.lstm_bias
        i, f, o, g = np.split(gates, 4)
        
        # Apply activations
        i = 1 / (1 + np.exp(-i))  # input gate
        f = 1 / (1 + np.exp(-f))  # forget gate
        o = 1 / (1 + np.exp(-o))  # output gate
        g = np.tanh(g)            # cell candidate
        
        # Update cell and hidden states
        self.c = f * self.c + i * g
        h_proposal = o * np.tanh(self.c)
        
        # 2. Relational memory step
        rm_output = self.rm.step(h_proposal)
        
        # 3. Combine LSTM and memory outputs
        combined = np.concatenate([h_proposal, rm_output])
        self.h = np.tanh(combined @ self.W_combine + self.b_combine)
        
        return self.h

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelationalMemoryCore(nn.Module):
    """
    Efficient PyTorch Implementation of Section 2: Relational Memory Core.
    Vectorized for batch processing and GPU acceleration.
    """
    def __init__(self, mem_slots, head_size, num_heads=4, input_size=None):
        super().__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.d_model = head_size * num_heads
        
        # Adjust input size if provided, otherwise assume input matches d_model
        # (Numpy code assumes input matches d_model, but practically we might project)
        self.input_size = input_size if input_size is not None else self.d_model

        # 1. Attention Projections (Vectorized Heads)
        # Instead of list of arrays (Numpy), use one big Linear layer
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model) # Output projection
        
        # 2. Row-wise MLP (Applied to all slots simultaneously)
        # Corresponds to Numpy Section 2: MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # 3. Gating Mechanisms (Shared across slots)
        # Corresponds to Numpy Section 2: LSTM-style gating
        self.W_gate_i = nn.Linear(self.d_model, self.d_model)
        self.W_gate_f = nn.Linear(self.d_model, self.d_model)
        self.W_gate_o = nn.Linear(self.d_model, self.d_model)
        
        # Memory State Buffer (registered as buffer to save state but not as param)
        self.register_buffer('memory', torch.zeros(1, mem_slots, self.d_model))
        
    def reset_state(self, batch_size, device):
        # Corresponds to Numpy reset_state
        stdev = 0.01
        self.memory = torch.randn(batch_size, self.mem_slots, self.d_model, device=device) * stdev

    def forward(self, input_vec):
        """
        Args:
            input_vec: (Batch, input_size)
        Returns:
            output: (Batch, d_model)
        """
        batch_size = input_vec.size(0)
        
        # Ensure memory exists and matches batch size
        if self.memory.size(0) != batch_size:
            self.reset_state(batch_size, input_vec.device)
            
        # 1. Prepare Memory Augmentation [M; x]
        # input_vec: (B, D) -> (B, 1, D)
        input_expanded = input_vec.unsqueeze(1)
        # memory: (B, Slots, D)
        # M_tilde: (B, Slots+1, D) -> Corresponds to Numpy 'np.concatenate'
        M_tilde = torch.cat([self.memory, input_expanded], dim=1)
        
        # 2. Multi-Head Attention (Vectorized)
        # Q, K, V: (B, Slots+1, D)
        Q = self.W_q(M_tilde)
        K = self.W_k(M_tilde)
        V = self.W_v(M_tilde)
        
        # Reshape for heads: (B, Slots+1, Heads, Head_Size)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled Dot-Product: (B, Heads, Slots+1, Slots+1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted Sum: (B, Heads, Slots+1, Head_Size)
        attended_heads = torch.matmul(attn_weights, V)
        
        # Concat Heads: (B, Slots+1, D)
        attended = attended_heads.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attended = self.W_o(attended)
        
        # 3. Residual & MLP
        gated = attended + M_tilde
        mlp_out = self.mlp(gated) # (B, Slots+1, D)
        
        # 4. Memory Gating (Vectorized - No for-loop!)
        # Extract slots (exclude input part for memory update)
        # memory_update: (B, Slots, D)
        memory_update_candidate = mlp_out[:, :-1, :]
        
        # Compute gates efficiently for all slots at once
        i_gate = torch.sigmoid(self.W_gate_i(memory_update_candidate))
        f_gate = torch.sigmoid(self.W_gate_f(memory_update_candidate))
        o_gate = torch.sigmoid(self.W_gate_o(memory_update_candidate))
        
        candidate = torch.tanh(memory_update_candidate)
        
        # Update Memory
        # old_memory: (B, Slots, D)
        new_memory = f_gate * self.memory + i_gate * candidate
        self.memory = o_gate * torch.tanh(new_memory) # Update internal state
        
        # Output is the processed input (the last slot in the sequence)
        output = mlp_out[:, -1, :]
        return output

class RelationalRNNCell(nn.Module):
    """
    Efficient PyTorch Implementation of Section 3: Relational RNN Cell.
    Combines standard LSTM (proposal) with RMC.
    """
    def __init__(self, input_size, hidden_size, mem_slots=4, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Proposal LSTM
        # Using nn.LSTMCell for efficiency (Corresponds to Numpy self.lstm)
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        
        # 2. Relational Memory
        self.rmc = RelationalMemoryCore(mem_slots, hidden_size // num_heads, num_heads, input_size=hidden_size)
        
        # 3. Combination Layer
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # State tracking (h, c)
        self.h = None
        self.c = None

    def reset_state(self, batch_size, device):
        self.h = torch.zeros(batch_size, self.hidden_size, device=device)
        self.c = torch.zeros(batch_size, self.hidden_size, device=device)
        self.rmc.reset_state(batch_size, device)

    def forward(self, x):
        """
        Args:
            x: (Batch, input_size)
        """
        batch_size = x.size(0)
        if self.h is None:
            self.reset_state(batch_size, x.device)
            
        # 1. Standard LSTM Step
        self.h, self.c = self.lstm(x, (self.h, self.c))
        
        # 2. RMC Step (using LSTM output as input)
        rm_out = self.rmc(self.h)
        
        # 3. Combine
        combined = torch.cat([self.h, rm_out], dim=1)
        final_h = torch.tanh(self.combine(combined))
        
        # Update h for next step (Architecture choice: RMC output affects next LSTM state?
        # The Numpy code updates self.h with the combined output, so we do too)
        self.h = final_h
        
        return final_h

```

:::

### 对照讲解：Numpy vs Torch 实现差异

1. **向量化（Vectorization）vs 循环**：
* **Numpy**: 在 `RelationalMemory.step` 中使用了 `for i in range(self.mem_slots)` 循环来逐个更新记忆槽的门控。这对性能是致命的，尤其是当 Slot 数较多时。
* **Torch**: 我使用了全矩阵运算。`self.W_gate_i(memory_update_candidate)` 直接作用于 `(Batch, Slots, Hidden)` 张量，PyTorch 会自动将线性层广播到最后两维。这使得门控计算瞬间完成，完全并行。


2. **多头注意力（Multi-Head Attention）的实现**：
* **Numpy**: 显式维护了 `W_q`, `W_k`, `W_v` 的列表 `[]`，并循环计算每个头。
* **Torch**: 标准的高效做法是使用一个大的 `nn.Linear(d_model, d_model)`，然后通过 `view` 和 `transpose` 操作将 `(Batch, Seq, D_model)` 重塑为 `(Batch, Heads, Seq, Head_Dim)`。这样只需一次矩阵乘法即可计算所有头的 Q/K/V。


3. **状态管理（State Management）**：
* **Numpy**: 类内部维护 `self.memory` 和 `self.h` 状态。由于 Numpy 无法自动感知 Batch 维度变化，通常只能处理固定 Batch 或单样本。
* **Torch**: 使用 `register_buffer` 管理 `memory`，这不仅能让记忆随模型保存（`state_dict`），还能自动处理设备（CPU/GPU）迁移。在 `forward` 中动态检查 Batch Size 并重置状态，增强了鲁棒性。


4. **容易写错的细节**：
* **拼接维度**: 在 RMC 中，输入是作为“新的一行”拼接到记忆矩阵中的（Dim=1），而不是扩展特征维度（Dim=2）。Torch 代码中的 `torch.cat(..., dim=1)` 对应了 Numpy 的 `axis=0`（因为 Numpy 代码假设输入是单样本 `(d_model,)`，而 Torch 是批次 `(Batch, d_model)`）。
* **Softmax 维度**: 注意力权重的 Softmax 必须作用在最后一个维度（Key 的维度）。如果维度搞错，注意力机制就会失效。Numpy 代码中 `axis=-1` 和 Torch `dim=-1` 是一致的。

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`18.pdf`

![Relational RNN 图 1](/paper-figures/18/img-006.png)
*图 1：建议结合本节 `递归中的关系记忆` 一起阅读。*

![Relational RNN 图 2](/paper-figures/18/img-004.png)
*图 2：建议结合本节 `递归中的关系记忆` 一起阅读。*

![Relational RNN 图 3](/paper-figures/18/img-005.png)
*图 3：建议结合本节 `递归中的关系记忆` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**Relational RNN**（围绕 `递归中的关系记忆`）

### 一、选择题（10题）

1. 在 Relational RNN 中，最关键的建模目标是什么？
   - A. 递归中的关系记忆
   - B. memory slots
   - C. MHDPA
   - D. gate
   - **答案：A**

2. 下列哪一项最直接对应 Relational RNN 的核心机制？
   - A. memory slots
   - B. MHDPA
   - C. gate
   - D. RMC
   - **答案：B**

3. 在复现 Relational RNN 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 Relational RNN，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 Relational RNN 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. Relational RNN 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 Relational RNN 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 Relational RNN 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，Relational RNN 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，Relational RNN 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 Relational RNN 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 Relational RNN 的核心前向步骤（简化版），并返回中间张量。
   - 参考答案：
     ```python
     import numpy as np
     
     def forward_core(x, w, b):
         z = x @ w + b
         h = np.tanh(z)
         return h, {"z": z, "h": h}
     ```

3. 写一个训练 step：前向、loss、反向、更新。
   - 参考答案：
     ```python
     def train_step(model, optimizer, criterion, xb, yb):
         optimizer.zero_grad()
         pred = model(xb)
         loss = criterion(pred, yb)
         loss.backward()
         optimizer.step()
         return float(loss.item())
     ```

4. 实现一个评估函数，返回主指标与一个辅助指标。
   - 参考答案：
     ```python
     import numpy as np
     
     def evaluate(y_true, y_pred):
         acc = (y_true == y_pred).mean()
         err = 1.0 - acc
         return {"acc": float(acc), "err": float(err)}
     ```

5. 实现梯度裁剪与学习率调度的训练循环（简化版）。
   - 参考答案：
     ```python
     import torch
     
     def train_loop(model, loader, optimizer, criterion, scheduler=None, clip=1.0):
         model.train()
         for xb, yb in loader:
             optimizer.zero_grad()
             loss = criterion(model(xb), yb)
             loss.backward()
             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
             optimizer.step()
             if scheduler is not None:
                 scheduler.step()
     ```

6. 实现 ablation 开关：可切换是否启用 `memory slots`。
   - 参考答案：
     ```python
     def forward_with_ablation(x, module, use_feature=True):
         if use_feature:
             return module(x)
         return x
     ```

7. 实现一个鲁棒的数值稳定 softmax / logsumexp 工具函数。
   - 参考答案：
     ```python
     import numpy as np
     
     def stable_softmax(x, axis=-1):
         x = x - np.max(x, axis=axis, keepdims=True)
         ex = np.exp(x)
         return ex / np.sum(ex, axis=axis, keepdims=True)
     ```

8. 写一个小型单元测试，验证 `MHDPA` 相关张量形状正确。
   - 参考答案：
     ```python
     def test_shape(out, expected_last_dim):
         assert out.ndim >= 2
         assert out.shape[-1] == expected_last_dim
     ```

9. 实现模型推理包装器，支持 batch 输入并返回结构化结果。
   - 参考答案：
     ```python
     def infer(model, xb):
         logits = model(xb)
         pred = logits.argmax(dim=-1)
         return {"pred": pred, "logits": logits}
     ```

10. 实现一个实验记录器，保存超参、指标和随机种子。
   - 参考答案：
     ```python
     import json
     from pathlib import Path
     
     def save_run(path, cfg, metrics, seed):
         payload = {"cfg": cfg, "metrics": metrics, "seed": seed}
         Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
     ```


<!-- AUTO_INTERVIEW_QA_END -->

