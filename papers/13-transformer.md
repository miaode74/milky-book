# 论文深度解读：Attention Is All You Need

## 1. 一句话概述

这篇开创性论文提出了 **Transformer** 架构，首次完全抛弃了当时主导的 RNN 和 CNN 结构，仅凭 **Self-Attention（自注意力机制）** 就实现了序列转换任务（如机器翻译）的最优性能（SOTA），并开启了现代大规模预训练模型（如 GPT、BERT）的时代。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**核心问题**：
当时的序列转换模型（Sequence Transduction Models）主要依赖复杂的循环神经网络（RNN）或卷积神经网络（CNN）。这些模型通常结合 Encoder-Decoder 架构和注意力机制 ，但存在以下瓶颈：

1. 
**无法并行化**：RNN 固有的时序依赖性导致训练难以并行，尤其是在长序列上 。


2. **长距离依赖建模困难**：RNN 需要多步传播才能关联远距离的词，路径长度随序列长度增长。

**主要贡献**：

1. 
**提出 Transformer**：一种全新的、简单的网络架构，完全摒弃了循环和卷积，仅基于注意力机制 。


2. 
**更优的性能与效率**：在 WMT 2014 英德翻译任务上达到 28.4 BLEU，不仅击败了所有现有模型（包括集成模型），而且训练时间大幅缩短（在 8 张 P100 GPU 上仅需 3.5 天）。


3. 
**泛化能力**：实验证明该模型能成功应用于英语成分句法分析（Constituency Parsing），显示了其在不同任务上的通用性 。



## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑

**故事背景与痛点**：
RNN（如 LSTM、GRU）通过维护隐藏状态  来处理序列，其中  依赖于  和输入 。这种**顺序计算的约束（Sequential Computation Constraint）** 使得模型无法在训练样本内部进行并行化 。虽然注意力机制已经成为序列模型的关键部分，但此前它几乎总是与 RNN 结合使用 。

**核心动机**：
作者希望能有一种架构，既能减少顺序计算以利用并行硬件，又能缩短长距离依赖在网络中的传播路径 。

**解决方案的推演**：

1. **摒弃 RNN**：为了并行化，必须打破时间步的顺序依赖。
2. 
**超越 CNN**：虽然 ByteNet 或 ConvS2S 使用卷积实现了并行，但它们关联两个位置所需的操作次数随距离增加（线性或对数级）。这使得学习长距离依赖变得困难 。


3. 
**拥抱 Attention**：Transformer 将这种操作次数降低到了常数级别  。通过 Self-Attention（自注意力），模型可以直接计算序列中任意两个位置的关联，而不受距离限制 。



## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

Transformer 依然沿用了经典的 **Encoder-Decoder** 架构 ，但其内部组件完全被替换。

### 4.1 整体架构

* 
**Encoder**：由  层堆叠而成。每层包含两个子层：多头自注意力（Multi-Head Self-Attention）和点式前馈网络（Position-wise Feed-Forward Network）。


* 
**Decoder**：同样由  层堆叠。除了 Encoder 中的两个子层外，还插入了第三个子层，用于对 Encoder 的输出执行多头注意力（Cross-Attention）。


* 
**残差与归一化**：每个子层后都接残差连接和层归一化（Layer Normalization），即  。



### 4.2 核心组件：Scaled Dot-Product Attention

这是 Transformer 的计算引擎。输入由 Query ()、Key () 和 Value () 组成。输出是 Value 的加权和。

**公式**：




* **为什么要除以 （Scaled）？**
当  较大时，点积结果的量级会变得很大，导致 Softmax 进入梯度极小的饱和区 。缩放因子抵消了这种影响。



### 4.3 Multi-Head Attention (多头注意力)

作者发现，与其使用单个注意力头，不如将  线性投影  次（本文 ），映射到不同的子空间 。

**公式**：


其中每个头  。
这允许模型同时关注来自不同位置的不同表示子空间的信息 。

### 4.4 Position-wise Feed-Forward Networks (FFN)

在注意力层之后，对每个位置独立且相同地应用一个全连接网络：


这相当于两个线性变换中间夹一个 ReLU 激活函数 。尽管不同位置的变换参数相同，但不同层的参数是不同的 。

### 4.5 Positional Encoding (位置编码)

由于模型不包含循环或卷积，必须显式注入位置信息 。作者使用了正弦和余弦函数：


这种选择允许模型学习相对位置，因为  可以表示为  的线性函数 。

```mermaid
graph TD
    subgraph Encoder
    I[Input Embedding] --> P[Positional Encoding]
    P --> L1_MHA[Multi-Head Self-Attention]
    L1_MHA --> L1_AddNorm1[Add & Norm]
    L1_AddNorm1 --> L1_FFN[Feed Forward]
    L1_FFN --> L1_AddNorm2[Add & Norm]
    L1_AddNorm2 --Stack Nx--> EncOut[Encoder Output]
    end

    subgraph Decoder
    O[Output Embedding] --> DP[Positional Encoding]
    DP --> L1_MMHA[Masked Multi-Head Attention]
    L1_MMHA --> L1_DAddNorm1[Add & Norm]
    L1_DAddNorm1 --> L1_CrossAtt[Multi-Head Attention\n(Q=Decoder, K,V=Encoder)]
    EncOut --> L1_CrossAtt
    L1_CrossAtt --> L1_DAddNorm2[Add & Norm]
    L1_DAddNorm2 --> L1_DFFN[Feed Forward]
    L1_DFFN --> L1_DAddNorm3[Add & Norm]
    L1_DAddNorm3 --Stack Nx--> Linear[Linear]
    Linear --> Softmax[Softmax]
    end

```

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 机器翻译主实验

* 
**数据集**：WMT 2014 英德（4.5M 句对）和英法（36M 句对）。


* 
**硬件**：8 张 P100 GPU。Base 模型训练 12 小时，Big 模型训练 3.5 天 。


* **结果 (Table 2)**：
* 
**英德 (EN-DE)**：Transformer (big) 达到 **28.4 BLEU**，比之前的 SOTA（包括集成模型）高出 2.0 BLEU 。


* 
**英法 (EN-FR)**：达到 **41.8 BLEU**，刷新了单模型 SOTA，且训练成本仅为之前最佳模型的 1/4 。





### 5.2 模型变体与消融实验 (Table 3)

作者通过调整 Base 模型参数验证了各组件的重要性：

1. 
**注意力头数 ()**：头数太少（）效果差（丢 0.9 BLEU），但头数太多（）效果也会下降 。


2. 
**Key 的维度 ()**：降低  会损害模型质量，表明点积兼容性的匹配并不容易 。


3. 
**模型规模**：更大的模型（更多层、更大 ）效果更好 。


4. 
**位置编码**：使用学习到的位置 Embedding 与正弦函数编码效果几乎相同 。



### 5.3 泛化性：英语成分句法分析

为了验证通用性，作者在 Penn Treebank 数据集上进行了训练。尽管没有针对该任务进行特定的调优，Transformer 依然取得了 **91.3 F1**（WSJ only setting），优于大多数之前的模型，证明其不仅适用于翻译，也能捕捉复杂的语法结构 。

## 6. Numpy 与 Torch 对照实现（含 code-group）

### 代码对应说明

提供的 Numpy 代码完整复现了 Transformer 的核心组件。以下是各部分的详细对应：

* **Scaled Dot-Product Attention**: 对应论文公式 (1)，包含 `scores` 计算、缩放、Masking 和 Softmax。
* **MultiHeadAttention**: 对应论文 3.2.2 节。其 `split_heads` 逻辑将 `(seq, d_model)` 变换为 `(heads, seq, dk)`，这与论文中将不同子空间投影并行计算的思想一致。
* **Positional Encoding**: 对应论文 3.5 节公式，使用 log 空间计算频率 `div_term`。
* **FeedForward**: 对应论文 3.3 节公式 (2)，包含两个线性层和 ReLU。
* **TransformerBlock**: 对应论文 Figure 1 中的单个 Encoder 层结构。

**数据维度假设**：

* Numpy 代码中的输入多为 `(seq_len, d_model)`，即未包含 batch 维度（或 batch=1）。
* Torch 实现将扩展为支持 `(batch_size, seq_len, d_model)`，这是深度学习框架的标准做法。
* Numpy 中使用了手动初始化的权重（如 `self.W_q`），Torch 中我们将使用 `nn.Linear` 替代，但逻辑等价。

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
      
# Scaled Dot-Product Attention
# The fundamental building block:

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
     
    Q: Queries (seq_len_q, d_k)
    K: Keys (seq_len_k, d_k)
    V: Values (seq_len_v, d_v)
    mask: Optional mask (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
     
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
     
    # Apply mask if provided (for causality or padding)
    if mask is not None:
        scores = scores + (mask * -1e9)
     
    # Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
     
    # Weighted sum of values
    output = np.dot(attention_weights, V)
     
    return output, attention_weights

# Test scaled dot-product attention
seq_len = 5
d_model = 8

Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"Attention weights sum (should be 1): {attn_weights.sum(axis=1)}")

# Visualize attention pattern
plt.figure(figsize=(8, 6))
plt.imshow(attn_weights, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Attention Weights Matrix')
plt.show()
      
# Multi-Head Attention
# Multiple attention "heads" attend to different aspects of the input:

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V for all heads (parallelized)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def split_heads(self, x):
        """Split into multiple heads: (seq_len, d_model) -> (num_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)
    
    def combine_heads(self, x):
        """Combine heads: (num_heads, seq_len, d_k) -> (seq_len, d_model)"""
        seq_len = x.shape[1]
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Multi-head attention forward pass
        
        Q, K, V: (seq_len, d_model)
        """
        # Linear projections
        Q = np.dot(Q, self.W_q.T)
        K = np.dot(K, self.W_k.T)
        V = np.dot(V, self.W_v.T)
        
        # Split into multiple heads
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention to each head
        head_outputs = []
        self.attention_weights = []
        
        for i in range(self.num_heads):
            head_out, head_attn = scaled_dot_product_attention(
                Q[i], K[i], V[i], mask
            )
            head_outputs.append(head_out)
            self.attention_weights.append(head_attn)
        
        # Stack heads
        heads = np.stack(head_outputs, axis=0)  # (num_heads, seq_len, d_k)
        
        # Combine heads
        combined = self.combine_heads(heads)  # (seq_len, d_model)
        
        # Final linear projection
        output = np.dot(combined, self.W_o.T)
        
        return output

# Test multi-head attention
d_model = 64
num_heads = 8
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads)

X = np.random.randn(seq_len, d_model)
output = mha.forward(X, X, X)  # Self-attention

print(f"\nMulti-Head Attention:")
print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Number of heads: {num_heads}")
print(f"Dimension per head: {mha.d_k}")
      
# Positional Encoding
# Since Transformers have no recurrence, we add position information:

def positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encoding
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Generate positional encodings
seq_len = 50
d_model = 64
pe = positional_encoding(seq_len, d_model)

# Visualize positional encodings
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.imshow(pe.T, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding (All Dimensions)')

plt.subplot(2, 1, 2)
# Plot first few dimensions
for i in [0, 1, 2, 3, 10, 20]:
    plt.plot(pe[:, i], label=f'Dim {i}')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
plt.title('Positional Encoding (Selected Dimensions)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Positional encoding shape: {pe.shape}")
print(f"Different frequencies encode position at different scales")
      
# Feed-Forward Network
# Applied to each position independently:

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # First layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        
        # Second layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# Test feed-forward
d_model = 64
d_ff = 256  # Usually 4x larger

ff = FeedForward(d_model, d_ff)
x = np.random.randn(10, d_model)
output = ff.forward(x)

print(f"\nFeed-Forward Network:")
print(f"Input: {x.shape}")
print(f"Hidden: ({x.shape[0]}, {d_ff})")
print(f"Output: {output.shape}")
      
# Layer Normalization
# Normalize across features (not batch like BatchNorm)

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        
        return output

ln = LayerNorm(d_model)
x = np.random.randn(10, d_model) * 3 + 5  # Unnormalized
normalized = ln.forward(x)

print(f"\nLayer Normalization:")
print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Output mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
      
# Complete Transformer Block

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x

# Test transformer block
block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)
x = np.random.randn(10, 64)
output = block.forward(x)

print(f"\nTransformer Block:")
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"\nBlock contains:")
print(f"  1. Multi-Head Self-Attention")
print(f"  2. Layer Normalization")
print(f"  3. Feed-Forward Network")
print(f"  4. Residual Connections")
      
# Visualize Multi-Head Attention Patterns

# Create attention with interpretable input
seq_len = 8
d_model = 64
num_heads = 4

mha = MultiHeadAttention(d_model, num_heads)
X = np.random.randn(seq_len, d_model)
output = mha.forward(X, X, X)

# Plot attention patterns for each head
fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))

for i, ax in enumerate(axes):
    attn = mha.attention_weights[i]
    im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Head {i+1}')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    
plt.colorbar(im, ax=axes, label='Attention Weight', fraction=0.046, pad=0.04)
plt.suptitle('Multi-Head Attention Patterns', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

print("\nEach head learns to attend to different patterns!")
print("Different heads capture different relationships in the data.")
      
# Causal (Masked) Self-Attention for Autoregressive Models

def create_causal_mask(seq_len):
    """Create mask to prevent attending to future positions"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask

# Test causal attention
seq_len = 8
causal_mask = create_causal_mask(seq_len)

Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

# Without mask (bidirectional)
output_bi, attn_bi = scaled_dot_product_attention(Q, K, V)

# With causal mask (unidirectional)
output_causal, attn_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

# Visualize difference
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Causal mask
ax1.imshow(causal_mask, cmap='Reds', aspect='auto')
ax1.set_title('Causal Mask\n(1 = masked/not allowed)')
ax1.set_xlabel('Key Position')
ax1.set_ylabel('Query Position')

# Bidirectional attention
im2 = ax2.imshow(attn_bi, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax2.set_title('Bidirectional Attention\n(can see future)')
ax2.set_xlabel('Key Position')
ax2.set_ylabel('Query Position')

# Causal attention
im3 = ax3.imshow(attn_causal, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax3.set_title('Causal Attention\n(cannot see future)')
ax3.set_xlabel('Key Position')
ax3.set_ylabel('Query Position')

plt.colorbar(im3, ax=[ax2, ax3], label='Attention Weight')
plt.tight_layout()
plt.show()

print("\nCausal masking is crucial for:")
print("  - Autoregressive generation (GPT, language models)")
print("  - Prevents information leakage from future tokens")
print("  - Each position can only attend to itself and previous positions")

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 对应 Numpy: self.W_q, self.W_k, self.W_v
        # 使用 nn.Linear 替代手写的矩阵乘法，合并 weights 和 biases
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 对应 Numpy: self.W_o
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, d_model)
        # 这里增加了 batch 维度，比 Numpy 版本更通用
        batch_size = query.size(0)
        
        # 1. Linear projections
        # 对应 Numpy: np.dot(Q, self.W_q.T)
        Q = self.w_q(query) 
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. Split heads
        # 对应 Numpy: split_heads 函数
        # 变换: (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        # 对应 Numpy: scaled_dot_product_attention 函数
        # 计算 scores: (batch, heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        # 对应 Numpy: scores + (mask * -1e9)
        if mask is not None:
            # mask shape 需广播至 (batch, 1, seq_q, seq_k)
            scores = scores.masked_fill(mask == 1, -1e9)
        
        # Softmax
        # 对应 Numpy: softmax(scores, axis=-1)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum
        # 对应 Numpy: np.dot(attention_weights, V)
        # 输出: (batch, heads, seq_q, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # 4. Combine heads
        # 对应 Numpy: combine_heads 函数
        # 变换: (batch, heads, seq, d_k) -> (batch, seq, heads, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. Final projection
        # 对应 Numpy: np.dot(combined, self.W_o.T)
        output = self.w_o(attn_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 对应 Numpy: positional_encoding 函数
        # 创建一个 buffer，不作为模型参数更新
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度: (1, max_len, d_model) 以便广播
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # 截取对应长度的 PE
        return x + self.pe[:, :x.size(1), :]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # 对应 Numpy: W1, b1, W2, b2
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() # 对应 Numpy: np.maximum(0, ...)

    def forward(self, x):
        # 对应 Numpy: max(0, xW1+b1)W2+b2
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        # 对应 Numpy: norm1, norm2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 对应 Numpy: Residual + Norm 结构
        # 注意: 论文原本是 Norm(x + Sublayer(x))，但也常见 x + Sublayer(Norm(x)) (Pre-LN)
        # 这里遵循论文原意 (Post-LN) 与 Numpy 实现
        
        # Sublayer 1: Self-Attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Sublayer 2: Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

def create_causal_mask(seq_len):
    """
    对应 Numpy: create_causal_mask
    返回形状 (seq_len, seq_len) 的 mask，上三角为 1 (masked)
    """
    # triu: 上三角矩阵，k=1 表示对角线上一位开始为 1
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask

# --- 测试代码 ---
d_model = 64
num_heads = 8
d_ff = 256
batch_size = 2
seq_len = 10

# 初始化模型
block = TransformerBlock(d_model, num_heads, d_ff)
pe_layer = PositionalEncoding(d_model)

# 模拟输入 (Batch, Seq, Feature)
x = torch.randn(batch_size, seq_len, d_model)
x = pe_layer(x) # 加上位置编码

# 创建 mask (Causal Mask)
mask = create_causal_mask(seq_len) # (seq, seq)

# 前向传播
output = block(x, mask)

print(f"Torch Block Output Shape: {output.shape}")

```

:::

### 对照讲解与差异分析

1. **Batch 维度的处理**：
* **Numpy**: 示例代码主要处理 2D 张量 `(seq_len, d_model)`，相当于 batch_size=1。
* **Torch**: 所有 `nn.Module` 都原生支持 3D 张量 `(batch_size, seq_len, d_model)`。在 `MultiHeadAttention` 中，必须使用 `.view` 和 `.transpose` 显式维护 batch 维度，否则会把不同样本的数据混淆。


2. **维度变换 (Transpose vs Permute/View)**：
* **Numpy**: 使用 `transpose(1, 0, 2)` 交换前两个维度。
* **Torch**: 使用 `.transpose(1, 2)` 交换头和序列长度维度。**关键点**：在 Torch 中，`transpose` 后内存可能不连续，必须调用 `.contiguous()` 才能进行 `.view` 操作（对应 `combine_heads` 步骤），这是初学者最容易报错的地方（RuntimeError: view size is not compatible with input tensor's size and stride）。


3. **Mask 的实现方式**：
* **Numpy**: 使用 `scores + (mask * -1e9)`，利用广播机制。
* **Torch**: 推荐使用 `.masked_fill(mask == 1, -1e9)`。这种方式语义更清晰，且在 GPU 上通常有优化内核。注意 Torch 的 mask 经常需要 `unsqueeze` 来适配 batch 和 heads 维度。


4. **参数初始化**：
* **Numpy**: 代码中显式地用 `np.random.randn * 0.1` 初始化权重。
* **Torch**: `nn.Linear` 默认使用 Kaiming 或 Xavier 初始化（取决于版本），通常比简单的正态分布初始化收敛更稳。这里为了效率直接使用了 `nn.Linear`，它同时也自动管理了 Bias。


5. **数值稳定性**：
* **Softmax**: Numpy 版本手动实现了 `exp(x - max)` 以防溢出。Torch 的 `F.softmax` 内部已经高度优化并处理了数值稳定性问题，直接调用即可。