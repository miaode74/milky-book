# 论文解读：Recurrent Neural Network Regularization

## 1. 一句话概述

这篇论文提出了一种在循环神经网络（RNN/LSTM）中应用 **Dropout** 的正确方法：**仅在非循环连接（层与层之间、输入/输出方向）上应用 Dropout，而严格保留循环连接（时间步之间）的完整性**，从而在解决过拟合的同时保留了捕捉长期依赖的能力。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**试图解决的问题**：
尽管 RNN（特别是 LSTM）在序列建模任务中表现出色，但它们往往面临严重的**过拟合（Overfitting）**问题 。在当时，Dropout 是前馈神经网络中最成功的正则化手段，但直接将其应用于 RNN（即在循环连接上加噪声）效果很差，因为它会破坏模型捕捉长期依赖（Long-term dependencies）的能力 。

**主要贡献**：

1. 
**提出改进的 Dropout 策略**：作者展示了如何“正确地”将 Dropout 应用于 LSTM。核心原则是**仅对非循环连接（Non-recurrent connections）**应用 Dropout operator，而不干扰循环状态的传递 。


2. 
**验证了有效性**：该方法在语言建模（PTB数据集）、语音识别和机器翻译等多个任务上显著减少了过拟合，并提升了模型性能 。


3. 
**理论直觉**：论文解释了这种方法为何有效：它避免了噪声在时间维度上的指数级放大，使信息流只被干扰有限次（约 `L` 次，其中 `L` 为层数），而不是随着序列长度 `T` 累积。



## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑

**动机（Motivation）**：
RNN 的强大之处在于利用循环连接存储历史信息。然而，训练大型 RNN 极其困难，因为它们很容易过拟合 。虽然 Dropout 在前馈网络中非常有效，但早期的尝试发现它在 RNN 上不仅没有帮助，反而有害 。Bayer et al. (2013) 指出，这是因为在循环连接上使用 Dropout 会导致噪声被反复放大，从而损害学习过程 。

**故事逻辑**：

1. 
**痛点**：由于缺乏有效的正则化手段，实际应用中的 RNN 往往被迫设计得很小，无法发挥大模型的潜力 。


2. 
**分析**：标准的 Dropout 会随机将单元置零。如果在  的循环路径上应用 Dropout，由于 RNN 的递归性质，这种随机扰动会随着时间步  复合，这就好比在记忆存储过程中不断擦除信息，导致 LSTM 无法“记住”长距离之外的事件 。


3. 
**解决方案**：作者提出，Dropout 应该只作用于“信息处理”的单元变换（即输入到隐藏层、隐藏层到输出），而不能作用于“记忆保持”的通道（即循环连接）。


4. 
**预期结果**：通过这种方式，模型既享受了 Dropout 带来的集成学习效应（Ensemble effect）和特征鲁棒性，又保留了 LSTM 宝贵的记忆能力 。



## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

### 核心策略

Zaremba et al. 的核心思想极其简洁：**Dropout 仅应用于非循环连接**（图中的虚线箭头），即  和 ，而循环连接  保持确定的、无噪声的状态 。

```mermaid
graph LR
    subgraph Time_t-1
    x_tm1(x_{t-1}) -->|Dropout| h_tm1(h_{t-1})
    h_tm1 -->|Clean Recurrence| h_t
    end
    
    subgraph Time_t
    x_t(x_t) -->|Dropout| h_t(h_t)
    h_t -->|Dropout| y_t(y_t)
    end
    
    subgraph Time_t+1
    x_tp1(x_{t+1}) -->|Dropout| h_tp1(h_{t+1})
    h_t -->|Clean Recurrence| h_tp1
    end
    
    style h_tm1 fill:#f9f,stroke:#333,stroke-width:2px
    style h_t fill:#f9f,stroke:#333,stroke-width:2px
    style h_tp1 fill:#f9f,stroke:#333,stroke-width:2px

```

### 关键公式与算子

论文定义了一个 Dropout 算子 ，它随机将参数子集置零。
对于一个标准的 LSTM 计算过程，我们将 Dropout 应用于线性变换之前的输入 （上一层输出）或 （输入），但不应用于循环输入 （同一层的上一时刻状态）。

具体公式如下（以 LSTM 的四个门计算为例）：
$$[i_t,f_t,o_t,g_t] = [\sigma,\sigma,\sigma,\tanh]\big(W_x D(x_t) + W_h h_{t-1} + b\big)$$
其中，Dropout 只作用在前馈输入 `x_t`（记作 `D(x_t)`），循环项 `h_{t-1}` 不做 Dropout。
**引用说明**：对应论文对“non-recurrent connections”的约束。


* **符号含义**：
* `D(x_t)`：来自上一层（或输入层）的激活值，经过 **Dropout** 处理。
* `h_{t-1}`：来自同一层上一时刻的隐藏状态，**未经 Dropout** 处理（直接连接）。
* `W_x, W_h, b`：仿射变换参数（矩阵乘法+偏置）。



### 为什么这样做有效？

论文指出，在这种设置下，信息从 `t` 时刻流向 `t+1` 时刻时，不会在循环边上反复穿过 Dropout。沿深度方向大约只受 `L` 次扰动（`L` 为层数），而不是沿时间方向受 `T` 次扰动（`T` 为序列长度）。这使梯度和记忆更容易穿越长序列，同时保留 Dropout 的正则化收益。

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 语言建模（Language Modeling）

* 
**数据集**：Penn Tree Bank (PTB) 。


* 
**设置**：训练了中型（Medium）和大型（Large）的正则化 LSTM，并与非正则化的基线模型对比。使用了 50% (Medium) 和 65% (Large) 的 Dropout 率 。


* **结果**：
* 
**非正则化 LSTM**：Test Perplexity 为 **114.5**，明显过拟合（训练集 perplexity 更低）。


* 
**正则化 Large LSTM**：Test Perplexity 降至 **78.4**，显著优于基线 。


* 
**模型集成**：将 38 个正则化模型集成后，Perplexity 进一步降至 **68.7**，达到了当时的 State-of-the-art 。





### 5.2 语音识别与机器翻译

* 
**语音识别**：在内部的冰岛语语音数据集上，Dropout 提高了帧级别的准确率（从 68.9% 提升至 70.5%）。


* 
**机器翻译**：在 WMT'14 英语到法语任务上，正则化 LSTM 的 BLEU 得分从 25.9 提升到了 **29.03** 。



### 5.3 分析实验

论文通过可视化证明了策略的必要性。

> "Standard dropout perturbs the recurrent connections, which makes it difficult for the LSTM to learn to store information for long periods of time." 
> 
> 

如果不保护循环连接，模型将无法学习长距离依赖。实验结果不仅证明了性能提升，也验证了该方法使大模型（Large LSTM）变得可用，此前由于过拟合，大模型往往不如小模型 。

## 6. Numpy 与 Torch 对照实现（含 code-group）

### 代码说明

这份 Numpy 代码实现了一个**简化的 Vanilla RNN**（注意：并非论文中的 LSTM，但原理通用），展示了三种情况：

1. **Standard Dropout (Zaremba et al.)**：代码中的 `RNNWithDropout` 类。
* 它在 `forward` 循环中，对每一步的前馈输入 `x_t` 以及送入输出层前的隐藏状态 `h_t` 调用 `dropout()`。
* **关键点**：由于 `dropout()` 函数内部每次调用 `np.random.rand`，这意味着**每个时间步生成的 Mask 都是不同的**。这符合 Zaremba 论文的原意（非循环连接上的噪声是独立同分布的）。
* **形状假设**：输入 `inputs` 是一个列表，每个元素形状 `(input_size, 1)`（Batch size=1）。


2. **Variational Dropout**：代码中的 `RNNWithVariationalDropout` 类。
* 它在循环开始前生成一次 Mask (`input_mask`, `hidden_mask`)，并在整个序列中复用。这其实更接近 Gal & Ghahramani (2016) 的 Variational RNN，但常被混淆。
* **对比**：Zaremba 方法每步 Mask 独立；Variational 方法全序列共用 Mask。



下面的 PyTorch 实现将严格复现这两种逻辑，并尽量利用向量化操作（虽然 Zaremba 风格的“每步独立 Mask”在 Torch 中通常直接用 `nn.Dropout` 即可实现，但我会显式写出逻辑以供对照）。

::: code-group

```python [Numpy]
# Paper 4: Recurrent Neural Network Regularization
# Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals (2014)
# Dropout for RNNs
# Key insight: Apply dropout to non-recurrent connections only, not recurrent connections.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
      
# Standard Dropout

def dropout(x, dropout_rate=0.5, training=True):
    """
    Standard dropout
    During training: randomly zero elements with probability dropout_rate
    During testing: scale by (1 - dropout_rate)
    """
    if not training or dropout_rate == 0:
        return x
    
    # Inverted dropout (scale during training)
    mask = (np.random.rand(*x.shape) > dropout_rate).astype(float)
    return x * mask / (1 - dropout_rate)

# Test dropout
x = np.ones((5, 1))
print("Original:", x.T)
print("With dropout (p=0.5):", dropout(x, 0.5).T)
print("With dropout (p=0.5):", dropout(x, 0.5).T)
print("Test mode:", dropout(x, 0.5, training=False).T)
      
# RNN with Proper Dropout
# Key: Dropout on inputs and outputs, NOT on recurrent connections!

class RNNWithDropout:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, dropout_rate=0.0, training=True):
        """
        Forward pass with dropout
        
        Dropout applied to:
        1. Input connections (x -> h)
        2. Output connections (h -> y)
        
        NOT applied to:
        - Recurrent connections (h -> h)
        """
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        hidden_states = []
        
        for x in inputs:
            # Apply dropout to INPUT
            x_dropped = dropout(x, dropout_rate, training)
            
            # RNN update (NO dropout on recurrent connection)
            h = np.tanh(
                np.dot(self.W_xh, x_dropped) +  # Dropout HERE
                np.dot(self.W_hh, h) +           # NO dropout HERE
                self.bh
            )
            
            # Apply dropout to HIDDEN state before output
            h_dropped = dropout(h, dropout_rate, training)
            
            # Output
            y = np.dot(self.W_hy, h_dropped) + self.by  # Dropout HERE
            
            outputs.append(y)
            hidden_states.append(h)
        
        return outputs, hidden_states

# Test
rnn = RNNWithDropout(input_size=10, hidden_size=20, output_size=10)
test_inputs = [np.random.randn(10, 1) for _ in range(5)]

outputs_train, _ = rnn.forward(test_inputs, dropout_rate=0.5, training=True)
outputs_test, _ = rnn.forward(test_inputs, dropout_rate=0.5, training=False)

print(f"Training output[0] mean: {outputs_train[0].mean():.4f}")
print(f"Test output[0] mean: {outputs_test[0].mean():.4f}")
      
# Variational Dropout
# Key innovation: Use same dropout mask across all timesteps!

class RNNWithVariationalDropout:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights (same as before)
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs, dropout_rate=0.0, training=True):
        """
        Variational dropout: SAME mask for all timesteps
        """
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        hidden_states = []
        
        # Generate masks ONCE for entire sequence
        if training and dropout_rate > 0:
            input_mask = (np.random.rand(self.input_size, 1) > dropout_rate).astype(float) / (1 - dropout_rate)
            hidden_mask = (np.random.rand(self.hidden_size, 1) > dropout_rate).astype(float) / (1 - dropout_rate)
        else:
            input_mask = np.ones((self.input_size, 1))
            hidden_mask = np.ones((self.hidden_size, 1))
        
        for x in inputs:
            # Apply SAME mask to each input
            x_dropped = x * input_mask
            
            # RNN update
            h = np.tanh(
                np.dot(self.W_xh, x_dropped) +
                np.dot(self.W_hh, h) +
                self.bh
            )
            
            # Apply SAME mask to each hidden state
            h_dropped = h * hidden_mask
            
            # Output
            y = np.dot(self.W_hy, h_dropped) + self.by
            
            outputs.append(y)
            hidden_states.append(h)
        
        return outputs, hidden_states

# Test variational dropout
var_rnn = RNNWithVariationalDropout(input_size=10, hidden_size=20, output_size=10)
outputs_var, _ = var_rnn.forward(test_inputs, dropout_rate=0.5, training=True)

print("Variational dropout uses consistent masks across timesteps")

```

```python [Torch]
import torch
import torch.nn as nn

class RNNWithDropoutTorch(nn.Module):
    """
    对应 Numpy 的 RNNWithDropout (Zaremba et al. Style)
    每次 forward 调用都会为每个时间步生成独立的 dropout mask
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 对应 Numpy: self.W_xh, self.W_hh, self.bh
        # 注意：Torch Linear 包含 bias，所以这里不用单独定义 bh
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False) # bias 合并在 i2h 中
        self.h2y = nn.Linear(hidden_size, output_size)
        
        # 初始化权重以匹配 Numpy (scale 0.01)
        nn.init.normal_(self.i2h.weight, std=0.01)
        nn.init.normal_(self.h2h.weight, std=0.01)
        nn.init.normal_(self.h2y.weight, std=0.01)
        
        # 对应 Numpy: dropout(x)
        self.dropout = nn.Dropout(p=0.0) # p 将在 forward 中动态设置

    def forward(self, inputs, dropout_rate=0.0, training=True):
        # 强制设置 dropout 状态和概率
        self.train(training)
        self.dropout.p = dropout_rate
        
        # 假设 inputs 是 list of tensors (T steps), each shape (B, input_size)
        # 对应 Numpy: h = np.zeros((self.hidden_size, 1))
        # 这里假设 batch_size = inputs[0].shape[0]
        batch_size = inputs[0].size(0)
        device = inputs[0].device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        hidden_states = []
        
        for x in inputs:
            # 1. Apply dropout to INPUT (每次调用生成新 mask)
            # 对应 Numpy: x_dropped = dropout(x, ...)
            x_dropped = self.dropout(x)
            
            # 2. RNN update
            # 对应 Numpy: h = np.tanh(np.dot(self.W_xh, x_dropped) + ...)
            # 循环连接 h 没有加 dropout
            h_pre = self.i2h(x_dropped) + self.h2h(h)
            h = torch.tanh(h_pre)
            
            # 3. Apply dropout to HIDDEN state before output
            # 对应 Numpy: h_dropped = dropout(h, ...)
            h_dropped = self.dropout(h)
            
            # 4. Output
            # 对应 Numpy: y = np.dot(self.W_hy, h_dropped) + ...
            y = self.h2y(h_dropped)
            
            outputs.append(y)
            hidden_states.append(h)
            
        return outputs, hidden_states

class RNNWithVariationalDropoutTorch(nn.Module):
    """
    对应 Numpy 的 RNNWithVariationalDropout
    关键点：整个序列使用同一个 mask
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h2y = nn.Linear(hidden_size, output_size)
        
        nn.init.normal_(self.i2h.weight, std=0.01)
        nn.init.normal_(self.h2h.weight, std=0.01)
        nn.init.normal_(self.h2y.weight, std=0.01)

    def forward(self, inputs, dropout_rate=0.0, training=True):
        batch_size = inputs[0].size(0)
        device = inputs[0].device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        hidden_states = []
        
        # 对应 Numpy: if training... Generate masks ONCE
        input_mask = None
        hidden_mask = None
        
        if training and dropout_rate > 0:
            # 手动生成 Bernoulli mask 并进行 scaling (Inverted Dropout)
            # 保持 mask 在 device 上
            input_mask = (torch.rand(batch_size, self.input_size, device=device) > dropout_rate).float()
            input_mask /= (1 - dropout_rate)
            
            hidden_mask = (torch.rand(batch_size, self.hidden_size, device=device) > dropout_rate).float()
            hidden_mask /= (1 - dropout_rate)
            
        for x in inputs:
            # 1. Apply SAME mask to each input
            if input_mask is not None:
                x_dropped = x * input_mask
            else:
                x_dropped = x
            
            # 2. RNN update
            h_pre = self.i2h(x_dropped) + self.h2h(h)
            h = torch.tanh(h_pre)
            
            # 3. Apply SAME mask to hidden
            if hidden_mask is not None:
                h_dropped = h * hidden_mask
            else:
                h_dropped = h
                
            y = self.h2y(h_dropped)
            
            outputs.append(y)
            hidden_states.append(h)
            
        return outputs, hidden_states

# Test Equivalence Helper
def test_torch_implementation():
    # Setup similar to Numpy test
    torch.manual_seed(42)
    rnn = RNNWithDropoutTorch(input_size=10, hidden_size=20, output_size=10)
    
    # Random inputs: List of (Batch=1, Input=10) tensors
    inputs = [torch.randn(1, 10) for _ in range(5)]
    
    out_train, _ = rnn(inputs, dropout_rate=0.5, training=True)
    out_test, _ = rnn(inputs, dropout_rate=0.5, training=False)
    
    print(f"Torch Train Output[0] mean: {out_train[0].mean().item():.4f}")
    print(f"Torch Test Output[0] mean: {out_test[0].mean().item():.4f}")

if __name__ == "__main__":
    test_torch_implementation()

```

:::

### 对照讲解与差异分析

1. **Zaremba vs Variational 的核心区别**：
* **Numpy `RNNWithDropout` (Zaremba)**：在 `for` 循环内部调用 `dropout(x)`。由于随机数生成器在每次调用时状态会改变，因此**时间步 T1 和 T2 的 Mask 是不一样的**。这虽然增加了噪声，但严格遵循了 Zaremba 论文中“非循环连接应当被正则化”的定义。
* **Numpy `RNNWithVariationalDropout`**：在 `for` 循环**外部**预先生成 Mask。这意味着**时间步 T1 和 T2 使用完全相同的 Mask**（即总是丢弃第 k 个单元）。这能保持特征连续性，在理论上更优（贝叶斯解释），但实现复杂度稍高。


2. **Torch 实现细节**：
* **Inverted Dropout**：Numpy 代码手动实现了 `mask / (1-p)`。Torch 的 `nn.Dropout` 内部自动处理了 Inverted Dropout（训练时缩放，推理时保持原样），因此在 `RNNWithDropoutTorch` 中我们直接调用 `self.dropout(x)` 即可。
* **手动掩码 (Variational)**：在 `RNNWithVariationalDropoutTorch` 中，不能直接用 `nn.Dropout`，因为我们需要**固定 Mask**。因此我使用了 `torch.rand(...) > dropout_rate` 手动生成 0/1 矩阵，并手动执行缩放 ` /= (1 - dropout_rate)`，完全对齐 Numpy 的逻辑。


3. **易错点提示**：
* **训练/测试模式**：Numpy 代码通过 `training` 参数控制。Torch 通过 `model.train()` 和 `model.eval()` 控制 `nn.Dropout` 的行为。为了强行对齐 Numpy 的函数签名，我在 `forward` 中显式设置了 `self.train(training)`。
* **循环连接的纯净性**：请注意在两个实现中，`h`（上一时刻状态）进入 `self.h2h(h)` 时是**没有**经过 Dropout 的。这是本论文最关键的工程细节。只有输出到下一层（或输出层）的 `h_dropped` 才被 Mask 干扰。

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`04.pdf`

![RNN Regularization 图 1](/paper-figures/04/img-000.png)
*图 1：建议结合本节 `循环网络正则化` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**RNN Regularization**（围绕 `循环网络正则化`）

### 一、选择题（10题）

1. 在 RNN Regularization 中，最关键的建模目标是什么？
   - A. 循环网络正则化
   - B. Dropout
   - C. non-recurrent
   - D. 泛化
   - **答案：A**

2. 下列哪一项最直接对应 RNN Regularization 的核心机制？
   - A. Dropout
   - B. non-recurrent
   - C. 泛化
   - D. 过拟合
   - **答案：B**

3. 在复现 RNN Regularization 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 RNN Regularization，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 RNN Regularization 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. RNN Regularization 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 RNN Regularization 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 RNN Regularization 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，RNN Regularization 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，RNN Regularization 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 RNN Regularization 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 RNN Regularization 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `Dropout`。
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

8. 写一个小型单元测试，验证 `non-recurrent` 相关张量形状正确。
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

