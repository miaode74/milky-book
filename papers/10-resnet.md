# 论文深度解读：Deep Residual Learning for Image Recognition (ResNet)

## 1. 一句话概述
这篇开创性的论文提出了**残差学习（Residual Learning）**框架，通过引入跨层恒等映射（Identity Shortcut Connection），解决了深度神经网络因梯度消失/爆炸和退化问题而无法训练的难题，成功将网络深度从十几层推向了上百层（如 ResNet-152），并横扫了 ILSVRC 2015 的各项冠军。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**核心问题：**
更深的神经网络通常被认为应该具有更强的表达能力，但在实际训练中，随着网络深度增加，训练变得更加困难。论文指出了一个反直觉的现象：深层网络的训练误差反而高于浅层网络，这被称为**退化问题（degradation problem）**，而非过拟合 。

**主要贡献：**
1.  **残差学习框架：** 提出让堆叠的层拟合残差映射 $\mathcal{F}(x) := \mathcal{H}(x) - x$，而不是直接拟合底层映射 $\mathcal{H}(x)$ 。
2.  **易于优化：** 证明了残差网络（ResNets）比普通网络（Plain Nets）更容易优化，并且随着深度增加，准确率能持续提升 。
3.  **极深网络实践：** 在 ImageNet 上训练了深达 152 层的网络（比 VGG 深 8 倍），且复杂度更低 。
4.  **SOTA 结果：** 单个模型在 ImageNet 验证集上的 top-5 错误率降至 4.49%，集成模型达到 3.57%，获得了 ILSVRC 2015 分类任务第一名 。

## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑

**深度学习的困境：**
深度卷积神经网络（CNN）通过端到端的多层特征提取带来了图像分类的突破。普遍认为“深度”是极其重要的 。然而，简单地堆叠更多层并不能直接带来更好的网络。

**障碍 1：梯度消失/爆炸（Vanishing/Exploding Gradients）**
这是阻碍深层网络收敛的传统拦路虎。虽然通过归一化初始化（Normalized Initialization）和中间归一化层（如 Batch Normalization）在很大程度上解决了这个问题，使得数十层的网络可以开始收敛 。

**障碍 2：退化问题（The Degradation Problem）**
当网络能够收敛时，出现了一个令人意外的问题：随着深度增加，准确率达到饱和后迅速下降。论文通过实验有力地证明了**这并非过拟合**，因为深层网络在**训练集**上的误差也比浅层网络高 。

> "The deeper network has higher training error, and thus test error." 

**逻辑推演与解决方案：**
作者提出了一个**构造性解（solution by construction）**的思考实验：
如果我们在一个较浅的已训练模型上增加层，且这些增加的层仅执行**恒等映射（Identity Mapping）**，那么深层模型的训练误差理应不高于浅层模型 。然而，现有的求解器（solvers）很难用非线性层去逼近恒等映射。

基于此，作者提出了**深度残差学习**：
如果恒等映射是那个“最优解”，那么将原本的学习目标转化为“逼近零残差”要比“逼近恒等映射”容易得多 。通过引入“捷径连接（Shortcut Connections）”，网络只需要学习输入与输出之间的残差部分。

## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

### 4.1 残差学习 (Residual Learning)
假设 $\mathcal{H}(x)$ 是我们要拟合的基础映射。如果假设多个非线性层可以逼近复杂函数，那么它们等价于可以逼近残差函数 $\mathcal{H}(x) - x$ 。
作者明确让堆叠层拟合残差映射：
$$\mathcal{F}(x) := \mathcal{H}(x) - x$$
因此原始映射变为：
$$\mathcal{H}(x) = \mathcal{F}(x) + x$$
**核心假设：** 优化残差映射 $\mathcal{F}(x)$ 比优化原始映射 $\mathcal{H}(x)$ 更容易。如果最优函数接近恒等映射，那么将权重推向零（使 $\mathcal{F}(x) \to 0$）比从头学习恒等映射要容易得多 。

### 4.2 捷径连接 (Shortcut Connections)
实现上，论文通过“捷径连接”来实现 $\mathcal{F}(x)+x$。
* **结构：** 如图 2 所示，捷径连接跳过了一层或多层。
* **计算：** 执行恒等映射，其输出直接与堆叠层的输出相加。
* **公式：**
    $$y = \mathcal{F}(x, \{W_i\}) + x$$
    其中 $\mathcal{F}(x, \{W_i\})$ 代表要学习的残差映射（例如两层卷积 $W_2\sigma(W_1x)$）。最后通过 ReLU 激活函数 $\sigma(y)$ 。
* **优势：** 不引入额外的参数，也不增加计算复杂度 。

### 4.3 网络架构细节
论文设计了类似 VGG 的网络结构进行对比：
1.  **Plain Network（普通网络）：** 受 VGG 启发，主要使用 $3\times3$ 卷积。遵循两个设计原则：(i) 输出特征图尺寸相同则滤波器数量相同；(ii) 特征图尺寸减半则滤波器数量加倍（保持时间复杂度一致）。
2.  **Residual Network（残差网络）：** 在普通网络的基础上插入捷径连接。
    * 当维度匹配时，直接使用恒等捷径（实线）。
    * 当维度增加时（虚线），探讨了两种策略：(A) 零填充；(B) $1\times1$ 卷积投影匹配维度 。

### 4.4 白板讲解逻辑图

```mermaid
graph TD
    subgraph "Problem Identification"
    A[Start: Train Deep Net] --> B{Does Depth Increase?}
    B -- Yes --> C[Observation: Training Error Increases]
    C --> D[Conclusion: Not Overfitting, but Degradation]
    end

    subgraph "Residual Solution"
    E[Goal: Fit Mapping H(x)] --> F[Reformulation: Let Layers Fit F(x) = H(x) - x]
    F --> G[New Mapping: H(x) = F(x) + x]
    G --> H[Implementation: Shortcut Connection]
    end

    subgraph "Forward Pass Flow"
    I[Input x] --> J[Weight Layer 1]
    J --> K[ReLU]
    K --> L[Weight Layer 2]
    L --> M[Result F(x)]
    I --> N[Identity Shortcut x]
    M --> O((Element-wise Add))
    N --> O
    O --> P[ReLU] --> Q[Output y]
    end

    D -.-> E
    H -.-> I

```

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 ImageNet 分类主实验

作者在 ImageNet 2012 数据集上对比了 18 层和 34 层的 Plain Net 与 ResNet。

* 
**Plain Net 现象：** 34 层网络的训练误差和验证误差均高于 18 层网络，证实了退化问题 。


* 
**ResNet 现象：** 34 层 ResNet 优于 18 层 ResNet，且大幅优于 34 层 Plain Net 。


> "The 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data." 
> 
> 


* **结论：** 残差学习成功解决了退化问题，使网络能从增加的深度中获益。

### 5.2 捷径连接类型的消融实验 (Table 3)

对比了三种处理维度增加的捷径方案 ：

* **A (Zero-padding):** 无额外参数。
* **B (Projection):** 仅在维度增加时用  卷积投影。
* **C (All Projection):** 所有捷径都用  卷积。


**结果：** C > B > A，但差异很小。作者认为投影并不是解决退化问题的关键，考虑到内存和速度，后续主要使用 B 或 A 。



### 5.3 更深的网络与瓶颈结构 (Bottleneck Architectures)

为了训练 50、101 和 152 层网络，作者引入了**瓶颈设计（Bottleneck Design）** 。

* 
**结构：** 将 2 层 `3x3` 卷积替换为 3 层堆叠：`1x1`（降维）、`3x3`（特征提取）、`1x1`（升维/恢复维度）。


* **目的：** 两侧的 `1x1` 层负责减少和恢复维度，使中间 `3x3` 层在较小通道宽度上计算，从而控制计算量。
* 
**结果：** ResNet-152 虽然极深，但 FLOPs (11.3 billion) 仍低于 VGG-19 (19.6 billion) ，且取得了单模型 top-5 错误率 4.49% 的最佳成绩 。



### 5.4 CIFAR-10 分析实验

在 CIFAR-10 上探索了 1202 层的超深网络。虽然可以训练且没有优化困难（训练误差 < 0.1%），但测试结果略差于 110 层网络，作者认为是由于数据集太小导致的过拟合 。同时，层响应的标准差分析表明，ResNet 的层响应通常比 Plain Net 更小，支持了“残差函数更接近于零”的假设 。

## 6. Numpy 与 Torch 对照实现

### 6.1 代码说明

提供的 Numpy 代码对应论文中 **Residual Block 的核心机制** 以及 **梯度流（Gradient Flow）的验证**。

* **对应模块：**
* `PlainLayer`: 模拟论文中的权重层（这里简化为全连接层，论文主干是卷积层，但数学原理一致）。
* `ResidualBlock`: 对应论文 Figure 2，实现了 。
* `measure_gradient_flow`: 验证论文 Section 1 & 4 中提到的“梯度传播”问题。


* **张量形状 (Shape)：**
* Numpy 代码假设输入 `x` 为列向量 `(input_size, 1)`。
* 权重 `W` 形状为 `(output_size, input_size)`。
* 这是典型的数学符号表示法，不同于深度学习框架常用的 `(Batch, Channel/Dim)` 形式。


* **假设：** 代码为了演示原理，手动实现了反向传播（Backward Pass），模拟了 SGD 的一步。输入 Batch Size 为 1（由 `x.shape` 推断）。

### 6.2 代码对照 (Code Group)

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
      
# The Problem: Degradation in Deep Networks
# Before ResNet, adding more layers actually made networks worse (not due to overfitting, but optimization difficulty).

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class PlainLayer:
    """Standard neural network layer"""
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((output_size, 1))
    
    def forward(self, x):
        self.x = x
        self.z = np.dot(self.W, x) + self.b
        self.a = relu(self.z)
        return self.a
    
    def backward(self, dout):
        da = dout * relu_derivative(self.z)
        self.dW = np.dot(da, self.x.T)
        self.db = np.sum(da, axis=1, keepdims=True)
        dx = np.dot(self.W.T, da)
        return dx

class ResidualBlock:
    """Residual block with skip connection: y = F(x) + x"""
    def __init__(self, size):
        self.layer1 = PlainLayer(size, size)
        self.layer2 = PlainLayer(size, size)
    
    def forward(self, x):
        self.x = x
        
        # Residual path F(x)
        out = self.layer1.forward(x)
        out = self.layer2.forward(out)
        
        # Skip connection: F(x) + x
        self.out = out + x
        return self.out
    
    def backward(self, dout):
        # Gradient flows through both paths
        # Skip connection provides direct path
        dx_residual = self.layer2.backward(dout)
        dx_residual = self.layer1.backward(dx_residual)
        
        # Total gradient: residual path + skip connection
        dx = dx_residual + dout  # This is the key!
        return dx

print("ResNet components initialized")
      
# Build Plain Network vs ResNet

class PlainNetwork:
    """Plain deep network without skip connections"""
    def __init__(self, input_size, hidden_size, num_layers):
        self.layers = []
        
        # First layer
        self.layers.append(PlainLayer(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(PlainLayer(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(PlainLayer(hidden_size, input_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class ResidualNetwork:
    """Deep network with residual connections"""
    def __init__(self, input_size, hidden_size, num_blocks):
        # Project to hidden size
        self.input_proj = PlainLayer(input_size, hidden_size)
        
        # Residual blocks
        self.blocks = [ResidualBlock(hidden_size) for _ in range(num_blocks)]
        
        # Project back to output
        self.output_proj = PlainLayer(hidden_size, input_size)
    
    def forward(self, x):
        x = self.input_proj.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.output_proj.forward(x)
        return x
    
    def backward(self, dout):
        dout = self.output_proj.backward(dout)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        dout = self.input_proj.backward(dout)
        return dout

# Create networks
input_size = 16
hidden_size = 16
depth = 10

plain_net = PlainNetwork(input_size, hidden_size, depth)
resnet = ResidualNetwork(input_size, hidden_size, depth)

print(f"Created Plain Network with {depth} layers")
print(f"Created ResNet with {depth} residual blocks")
      
# Demonstrate Gradient Flow
# The key advantage: gradients flow more easily through skip connections

def measure_gradient_flow(network, name):
    """Measure gradient magnitude at different depths"""
    # Random input
    x = np.random.randn(input_size, 1)
    
    # Forward pass
    output = network.forward(x)
    
    # Create gradient signal
    dout = np.ones_like(output)
    
    # Backward pass
    network.backward(dout)
    
    # Collect gradient magnitudes
    grad_norms = []
    
    if isinstance(network, PlainNetwork):
        for layer in network.layers:
            grad_norm = np.linalg.norm(layer.dW)
            grad_norms.append(grad_norm)
    else:  # ResNet
        grad_norms.append(np.linalg.norm(network.input_proj.dW))
        for block in network.blocks:
            grad_norm1 = np.linalg.norm(block.layer1.dW)
            grad_norm2 = np.linalg.norm(block.layer2.dW)
            grad_norms.append(np.mean([grad_norm1, grad_norm2]))
        grad_norms.append(np.linalg.norm(network.output_proj.dW))
    
    return grad_norms

# Measure gradient flow in both networks
plain_grads = measure_gradient_flow(plain_net, "Plain Network")
resnet_grads = measure_gradient_flow(resnet, "ResNet")

# Plot comparison
plt.figure(figsize=(12, 5))
plt.plot(range(len(plain_grads)), plain_grads, 'o-', label='Plain Network', linewidth=2)
plt.plot(range(len(resnet_grads)), resnet_grads, 's-', label='ResNet', linewidth=2)
plt.xlabel('Layer Depth (deeper →)')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow: ResNet vs Plain Network')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

print(f"\nPlain Network - First layer gradient: {plain_grads[0]:.6f}")
print(f"Plain Network - Last layer gradient: {plain_grads[-1]:.6f}")
print(f"Gradient ratio (first/last): {plain_grads[0]/plain_grads[-1]:.2f}x\n")

print(f"ResNet - First layer gradient: {resnet_grads[0]:.6f}")
print(f"ResNet - Last layer gradient: {resnet_grads[-1]:.6f}")
print(f"Gradient ratio (first/last): {resnet_grads[0]/resnet_grads[-1]:.2f}x")

print(f"\nResNet maintains gradient flow {(plain_grads[0]/plain_grads[-1]) / (resnet_grads[0]/resnet_grads[-1]):.1f}x better!")

```

```python [Torch]
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set deterministic behavior for comparison
torch.manual_seed(42)

class PlainLayer(nn.Module):
    """
    Standard neural network layer: Linear + ReLU.
    Equivalent to Numpy PlainLayer but uses Autograd.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        # PyTorch Linear uses (Batch, In) -> (Batch, Out) convention
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        
        # Initialize weights to match Numpy: He initialization
        # Numpy: randn * sqrt(2/n)
        nn.init.normal_(self.linear.weight, mean=0.0, std=np.sqrt(2.0 / input_size))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x shape in Numpy was (Input, 1), here we expect (Batch, Input)
        return self.relu(self.linear(x))

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: y = F(x) + x.
    Corresponds to Numpy ResidualBlock.
    """
    def __init__(self, size):
        super().__init__()
        # F(x) involves two layers
        self.layer1 = PlainLayer(size, size)
        self.layer2 = PlainLayer(size, size)

    def forward(self, x):
        # Residual path F(x)
        out = self.layer1(x)
        out = self.layer2(out)
        
        # Skip connection: F(x) + x
        # Note: PyTorch handles the gradient branching (addition) automatically
        return out + x

class PlainNetwork(nn.Module):
    """Plain deep network without skip connections"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        layers = []
        # First layer
        layers.append(PlainLayer(input_size, hidden_size))
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(PlainLayer(hidden_size, hidden_size))
        # Output layer
        layers.append(PlainLayer(hidden_size, input_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResidualNetwork(nn.Module):
    """Deep network with residual connections"""
    def __init__(self, input_size, hidden_size, num_blocks):
        super().__init__()
        self.input_proj = PlainLayer(input_size, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.output_proj = PlainLayer(hidden_size, input_size)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x

def measure_gradient_flow(network, name, input_size):
    """
    Measure gradient magnitude using Autograd.
    Equivalent to Numpy measure_gradient_flow logic.
    """
    # Create input with Batch Size = 1 to match Numpy (16, 1) vector logic
    # Torch expects (Batch, Input_Size) -> (1, 16)
    x = torch.randn(1, input_size, requires_grad=True)
    
    # Forward pass
    output = network(x)
    
    # Backward pass
    network.zero_grad()
    # Gradient signal of 1s, matching Numpy 'dout = np.ones_like(output)'
    output.backward(torch.ones_like(output))
    
    # Collect gradient magnitudes
    grad_norms = []
    
    if isinstance(network, PlainNetwork):
        # Iterate through Sequential layers
        for layer in network.net:
            # Access the underlying Linear layer's weight gradient
            grad_norm = layer.linear.weight.grad.norm().item()
            grad_norms.append(grad_norm)
    else:  # ResNet
        # Input projection
        grad_norms.append(network.input_proj.linear.weight.grad.norm().item())
        # Blocks
        for block in network.blocks:
            g1 = block.layer1.linear.weight.grad.norm().item()
            g2 = block.layer2.linear.weight.grad.norm().item()
            grad_norms.append(np.mean([g1, g2]))
        # Output projection
        grad_norms.append(network.output_proj.linear.weight.grad.norm().item())
        
    return grad_norms

# --- Configuration matching Numpy ---
input_size = 16
hidden_size = 16
depth = 10

plain_net = PlainNetwork(input_size, hidden_size, depth)
resnet = ResidualNetwork(input_size, hidden_size, depth)

# --- Execution ---
plain_grads = measure_gradient_flow(plain_net, "Plain Network", input_size)
resnet_grads = measure_gradient_flow(resnet, "ResNet", input_size)

# --- Visualization (Reusing Matplotlib) ---
plt.figure(figsize=(12, 5))
plt.plot(range(len(plain_grads)), plain_grads, 'o-', label='Plain Network (Torch)', linewidth=2)
plt.plot(range(len(resnet_grads)), resnet_grads, 's-', label='ResNet (Torch)', linewidth=2)
plt.xlabel('Layer Depth (deeper →)')
plt.ylabel('Gradient Magnitude')
plt.title('Gradient Flow (Torch Autograd): ResNet vs Plain Network')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

# --- Analysis Output ---
print(f"Plain (Torch) - First/Last Ratio: {plain_grads[0]/(plain_grads[-1]+1e-9):.2f}x")
print(f"ResNet (Torch) - First/Last Ratio: {resnet_grads[0]/(resnet_grads[-1]+1e-9):.2f}x")

```

:::

### 6.3 关键对照讲解

1. **自动微分 vs 手动反向传播：**
* **Numpy:** 必须手动编写 `backward()` 方法，应用链式法则（Chain Rule）。例如在 `ResidualBlock` 中，需要显式计算 `dx = dx_residual + dout`。这是理解 ResNet 为什么能改善梯度的关键物理视角：梯度 `dout` 毫无损耗地通过 `+` 号直接传给了输入。
* **Torch:** 利用 `autograd` 引擎。我们在 `forward` 中写 `out + x` 时，PyTorch 自动构建计算图。在反向传播时，加法节点的梯度分发特性（梯度会等值复制到两个分支）自动实现了 Numpy 代码中的 `dx_residual + dout`。


2. **数据形状与维度 (Shape Semantics)：**
* **Numpy:** 采用了数学上常见的列向量表示 。输入形状为 `(N, 1)`，权重为 `(Out, In)`。
* **Torch:** 采用深度学习框架通用的 Batch First 约定。输入形状为 `(Batch, In)`，线性层内部计算实质是 。因此代码中输入张量形状为 `(1, 16)` 而非 `(16, 1)`。


3. **权重初始化 (Initialization)：**
* **Numpy:** 手动实现了 He Initialization (`randn * sqrt(2/n)`)。
* **Torch:** 使用 `nn.init.normal_` 配合计算出的标准差来实现同等效果。如果直接使用默认的 `nn.Linear` 初始化（通常是 Uniform），梯度流的数值表现会与 Numpy 版本不一致，从而影响对照结论。


4. **梯度提取 (Gradient Access)：**
* **Numpy:** 梯度直接存储在 `self.dW` 属性中，因为这是我们自己定义的类。
* **Torch:** 必须先执行 `.backward()`，梯度才会由 Autograd 计算并累积到 `tensor.grad` 属性中。在 `measure_gradient_flow` 函数中，我们通过访问 `layer.linear.weight.grad` 来获取与 Numpy `dW` 物理意义一致的量。



```

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`10.pdf`

![ResNet 图 1](/paper-figures/10/page-1.png)
*图 1：建议结合本节 `残差学习` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**ResNet**（围绕 `残差学习`）

### 一、选择题（10题）

1. 在 ResNet 中，最关键的建模目标是什么？
   - A. 残差学习
   - B. shortcut
   - C. 退化问题
   - D. 瓶颈层
   - **答案：A**

2. 下列哪一项最直接对应 ResNet 的核心机制？
   - A. shortcut
   - B. 退化问题
   - C. 瓶颈层
   - D. 梯度传播
   - **答案：B**

3. 在复现 ResNet 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 ResNet，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 ResNet 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. ResNet 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 ResNet 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 ResNet 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，ResNet 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，ResNet 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 ResNet 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 ResNet 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `shortcut`。
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

8. 写一个小型单元测试，验证 `退化问题` 相关张量形状正确。
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

