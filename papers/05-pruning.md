# 论文解读：Keeping Neural Networks Simple by Minimizing the Description Length of the Weights

## 1. 一句话概述

Geoffrey Hinton 等人（1993）提出基于 **最小描述长度（MDL）原则**，通过在训练过程中向权重注入高斯噪声并优化“噪声权重的描述成本（KL散度）”与“数据误差”的权衡，从而在极少训练数据下实现神经网络的强泛化能力。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**核心问题**：在训练数据非常稀缺（scarce training data）的情况下，复杂的神经网络极其容易过拟合（overfit），导致对新数据的泛化能力极差。

**主要贡献**：
1.  **理论框架**：引入 MDL 原则，将学习过程视为“发送者”向“接收者”传送模型和数据误差的过程。目标是最小化总编码长度（Code Length）。
2.  **方法创新**：提出 **“噪声权重”（Noisy Weights）** 机制。权重不是固定值，而是一个高斯分布。训练不仅更新均值，还更新方差。
3.  **Bits Back 参数**：提出了著名的 **"Bits Back"** 论点，证明了使用后验分布对权重进行有损编码时，部分编码成本可以被“赎回”，使得权重的描述成本等于先验分布与后验分布之间的 KL 散度（Kullback-Leibler divergence）。
4.  **自适应先验**：使用高斯混合模型（Mixture of Gaussians）作为权重的先验分布，允许网络自动学习权重的聚类（soft weight-sharing）。

> "Supervised neural networks generalize well if there is much less information in the weights than there is in the output vectors of the training cases." 
> （如果权重中包含的信息远少于训练样本输出向量中的信息，监督神经网络就能很好地泛化。）

## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑

**过拟合的本质**：
当模型过于复杂而数据过少时，模型会记住训练数据中的噪声。为了避免这种情况，必须限制模型权重中包含的“信息量”。

**已有的限制方法**：
* **限制连接数**：直接减少参数数量。
* **权重共享（Weight Sharing）**：强制一组权重相等（如 CNN）。
* **权重截断/量化**：但这会导致优化困难，因为离散值的导数不平滑 。

**MDL 的引入（Sender-Receiver 模型）**：
论文建立了一个通信模型：发送者（Sender）知道输入和正确的输出，接收者（Receiver）只知道输入。
* 发送者首先发送神经网络的 **权重（Weights）**。
* 然后发送网络预测值与真实值之间的 **残差（Misfits）**。
* **总成本 = 权重描述成本 + 数据误差描述成本**。
最小化这个总成本，就等于找到了一个既简单（权重便宜）又准确（误差小）的模型。

> "The Minimum Description Length Principle (Rissanen, 1986) asserts that the best model of some data is the one that minimizes the combined cost of describing the model and describing the misfit..." 
> （最小描述长度原则断言：数据的最佳模型是那个能最小化“描述模型成本”与“描述误差成本”之和的模型。）

**本文的独特视角（Noisy Weights）**：
传统的权重衰减（Weight Decay）可以被视为假设权重服从固定高斯先验的 MDL 特例 。但本文更进一步，允许权重的“精度”（Precision/Variance）也是可学习的。
* 如果一个权重对误差影响很大，它需要高精度（低方差），编码成本高。
* 如果一个权重对误差影响很小，它可以有很高的噪声（高方差），编码成本低。
* 这相当于自动剪枝或降低不重要权重的精度。

## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

该论文的方法不仅仅是简单的正则化，而是一套完整的**变分推理（Variational Inference）**流程的早期形式。

### 4.1 核心逻辑框图 (Mermaid)

```mermaid
graph TD
    subgraph Training Loop
    A[初始化权重分布 Q(w|θ)] -->|参数: mean μ, std σ| B(前向传播)
    B --> C{添加高斯噪声}
    C -->|采样| D[采样权重 w = μ + σ*ε]
    D --> E[计算网络输出 y]
    E --> F[计算数据误差 E_data]
    D --> G[计算 KL 散度 KL(Q||P)]
    F & G --> H[总损失 Loss = E_data + KL]
    H -->|Backprop| I[更新 μ 和 log(σ)]
    end
    subgraph Inference
    J[使用训练好的 μ 作为权重] --> K[预测]
    end

```

### 4.2 关键公式与技术细节

#### 1. 权重的描述长度（Bits Back Argument）

这是论文最难懂也是最精彩的部分。为了通过噪声信道传送权重，我们从后验分布 `q(w)` 中采样一个具体值 `w` 发送。虽然发送高精度的 `w` 很贵，但接收者在恢复出 `w` 后，可以推算出当初采样时对应的随机信息，这部分随机比特可被“赎回”（bits back）。
最终，权重的编码成本可写成后验 `q(w)` 与先验 `p(w)` 的 KL 散度：
$$L_C \approx \mathrm{KL}(q(w)\|p(w))$$




对于高斯后验/先验（对角近似），这可以写成逐参数的解析形式：
$$\mathrm{KL}\big(\mathcal{N}(\mu,\sigma^2)\,\|\,\mathcal{N}(0,\tau^2)\big)=\log\frac{\tau}{\sigma}+\frac{\sigma^2+\mu^2}{2\tau^2}-\frac12$$




* **含义**：如果后验方差 `\sigma^2` 较大（该权重不重要），编码该权重的净成本会下降，因此“噪声大”的权重更便宜、也更容易被剪枝。

#### 2. 数据误差的期望（Expected Squared Error）

由于权重是带噪声的随机变量，网络的输出也是随机的。论文针对单隐层网络推导了输出均值和方差的精确计算方法 ，从而可以直接优化**期望误差**，而不需要蒙特卡洛采样（虽然现代实现通常直接用 Reparameterization Trick 采样）。

#### 3. 自适应先验（Mixture of Gaussians Prior）

为了进一步压缩，论文不假设所有权重都来自同一个零均值高斯（即标准 L2 正则），而是假设权重来自一个**高斯混合模型（GMM）**：




* 网络在训练权重的 *同时*，也在优化这个 GMM 先验的参数（均值、方差、混合比例）。
* 这导致了 **Soft Weight Sharing**：权重会自动聚类到几个中心附近（例如 0, +1, -1），从而大幅降低描述长度。

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 实验设置

* **任务**：预测多肽分子（peptide molecules）的生物活性。
* **数据**：
* **输入**：128维特征（归一化）。
* **训练集**：仅 105 个样本（极小，典型的 "High Dimension, Low Sample Size" 问题）。
* **测试集**：420 个样本。


* **模型**：4 个隐层单元的神经网络（参数量 521 个）。注意参数量 (521) 远大于训练样本数 (105)，极易过拟合。

### 5.2 实验结果

论文对比了三种方法的 **相对误差（Relative Error）**（值越低越好，1.0 代表瞎猜均值）：

1. 
**无正则化网络**：Relative Error = **0.967**。完全过拟合，基本没有泛化能力 。


2. 
**权重衰减（Weight Decay）**：Relative Error = **0.291**。需要仔细调节衰减系数 。


3. 
**本文方法（MDL + Noisy Weights + GMM Prior）**：Relative Error = **0.286** 。



### 5.3 结果分析

* 
**权重分布的可视化**：论文展示了训练后的权重分布（Figure 3），发现权重自动分成了三个尖锐的簇（clusters），且大部分权重被压缩到了 0 附近或特定的值。这验证了 GMM 先验起到了自动发现结构的作用 。


* 
**主要结论**：在数据极少的情况下，通过显式地最小化权重的“信息量”（即 MDL），可以训练出比标准正则化泛化能力更强的复杂非线性模型 。



---

## 6. Numpy 与 Torch 对照实现

### 6.1 代码对应关系与假设说明

**特别说明**：
用户提供的 Numpy 代码实现的是 **"Magnitude-Based Pruning"（基于幅度的剪枝）** 和 **"Iterative Pruning"（迭代剪枝）**。

* **关联性**：这是 Hinton 论文思想的**现代工程化特例**。Hinton 论文通过 KL 散度“软性”地压缩权重信息（让不重要的权重噪声变大/趋向于0），而现代剪枝直接“硬性”地将小权重置为 0。两者目标一致：**最小化模型描述长度（MDL）**（代码中的 `compute_mdl` 函数也体现了这一点）。
* **Numpy 代码逻辑**：
1. `SimpleNN`: 一个双层全连接网络，手动实现 ReLU 和 Softmax。
2. `mask`: 引入掩码矩阵，前向传播时 `w * mask`。
3. `prune_by_magnitude`: 计算权重绝对值阈值，更新 mask。
4. `train_network`: 手写反向传播（Backprop），并在梯度更新时应用 mask。



**数据与张量形状假设**：

* **Input**: `X`，其中第一维是 batch size `B`。
* **Layer 1**: 权重形状按 Numpy 习惯写作 `(in_dim, out_dim)`；而 PyTorch `nn.Linear` 内部通常存储为 `(out_dim, in_dim)` 并执行 `y = xW^T + b`。在 Torch 实现中使用标准 `nn.Linear`，但逻辑保持等价。
* **Output**: `logits`，用于分类任务。

### 6.2 代码对照 (VitePress Code Group)

::: code-group

```python [Numpy]
# Paper 5: Keeping Neural Networks Simple by Minimizing the Description Length
# Hinton & Van Camp (1993) + Modern Pruning Techniques
# Network Pruning & Compression
# Key insight: Remove unnecessary weights to get simpler, more generalizable networks. Smaller = better!

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
      
# Simple Neural Network for Classification

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleNN:
    """Simple 2-layer neural network"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        
        # Keep track of masks for pruning
        self.mask1 = np.ones_like(self.W1)
        self.mask2 = np.ones_like(self.W2)
    
    def forward(self, X):
        """Forward pass"""
        # Apply masks (for pruned weights)
        W1_masked = self.W1 * self.mask1
        W2_masked = self.W2 * self.mask2
        
        # Hidden layer
        self.h = relu(np.dot(X, W1_masked) + self.b1)
        
        # Output layer
        logits = np.dot(self.h, W2_masked) + self.b2
        probs = softmax(logits)
        
        return probs
    
    def predict(self, X):
        """Predict class labels"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def accuracy(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def count_parameters(self):
        """Count total and active (non-pruned) parameters"""
        total = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        active = int(np.sum(self.mask1) + self.b1.size + np.sum(self.mask2) + self.b2.size)
        return total, active

# Test network
nn = SimpleNN(input_dim=10, hidden_dim=20, output_dim=3)
X_test = np.random.randn(5, 10)
y_test = nn.forward(X_test)
print(f"Network output shape: {y_test.shape}")
total, active = nn.count_parameters()
print(f"Parameters: {total} total, {active} active")
      
# Generate Synthetic Dataset

def generate_classification_data(n_samples=1000, n_features=20, n_classes=3):
    """
    Generate synthetic classification dataset
    Each class is a Gaussian blob
    """
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for c in range(n_classes):
        # Random center for this class
        center = np.random.randn(n_features) * 3
        
        # Generate samples around center
        X_class = np.random.randn(samples_per_class, n_features) + center
        y_class = np.full(samples_per_class, c)
        
        X.append(X_class)
        y.append(y_class)
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

# Generate data
X_train, y_train = generate_classification_data(n_samples=1000, n_features=20, n_classes=3)
X_test, y_test = generate_classification_data(n_samples=300, n_features=20, n_classes=3)

print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")
      
# Train Baseline Network

def train_network(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    """
    Simple training loop
    """
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Forward pass
        probs = model.forward(X_train)
        
        # Cross-entropy loss
        y_one_hot = np.zeros((len(y_train), model.output_dim))
        y_one_hot[np.arange(len(y_train)), y_train] = 1
        loss = -np.mean(np.sum(y_one_hot * np.log(probs + 1e-8), axis=1))
        
        # Backward pass (simplified)
        batch_size = len(X_train)
        dL_dlogits = (probs - y_one_hot) / batch_size
        
        # Gradients for W2, b2
        dL_dW2 = np.dot(model.h.T, dL_dlogits)
        dL_db2 = np.sum(dL_dlogits, axis=0)
        
        # Gradients for W1, b1
        dL_dh = np.dot(dL_dlogits, (model.W2 * model.mask2).T)
        dL_dh[model.h <= 0] = 0  # ReLU derivative
        dL_dW1 = np.dot(X_train.T, dL_dh)
        dL_db1 = np.sum(dL_dh, axis=0)
        
        # Update weights (only where mask is active)
        model.W1 -= lr * dL_dW1 * model.mask1
        model.b1 -= lr * dL_db1
        model.W2 -= lr * dL_dW2 * model.mask2
        model.b2 -= lr * dL_db2
        
        # Track metrics
        train_losses.append(loss)
        test_acc = model.accuracy(X_test, y_test)
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Test Acc: {test_acc:.2%}")
    
    return train_losses, test_accuracies

# Train baseline model
print("Training baseline network...\n")
baseline_model = SimpleNN(input_dim=20, hidden_dim=50, output_dim=3)
train_losses, test_accs = train_network(baseline_model, X_train, y_train, X_test, y_test, epochs=100)

baseline_acc = baseline_model.accuracy(X_test, y_test)
total_params, active_params = baseline_model.count_parameters()
print(f"\nBaseline: {baseline_acc:.2%} accuracy, {active_params} parameters")
      
# Magnitude-Based Pruning
# Remove weights with smallest absolute values

def prune_by_magnitude(model, pruning_rate):
    """
    Prune weights with smallest magnitudes
    
    pruning_rate: fraction of weights to remove (0-1)
    """
    # Collect all weights
    all_weights = np.concatenate([model.W1.flatten(), model.W2.flatten()])
    all_magnitudes = np.abs(all_weights)
    
    # Find threshold
    threshold = np.percentile(all_magnitudes, pruning_rate * 100)
    
    # Create new masks
    model.mask1 = (np.abs(model.W1) > threshold).astype(float)
    model.mask2 = (np.abs(model.W2) > threshold).astype(float)
    
    print(f"Pruning threshold: {threshold:.6f}")
    print(f"Pruned {pruning_rate:.1%} of weights")
    
    total, active = model.count_parameters()
    print(f"Remaining parameters: {active}/{total} ({active/total:.1%})")

# Test pruning
import copy
pruned_model = copy.deepcopy(baseline_model)

print("Before pruning:")
acc_before = pruned_model.accuracy(X_test, y_test)
print(f"Accuracy: {acc_before:.2%}\n")

print("Pruning 50% of weights...")
prune_by_magnitude(pruned_model, pruning_rate=0.5)

print("\nAfter pruning (before retraining):")
acc_after = pruned_model.accuracy(X_test, y_test)
print(f"Accuracy: {acc_after:.2%}")
print(f"Accuracy drop: {(acc_before - acc_after):.2%}")
      
# Fine-tuning After Pruning
# Retrain remaining weights to recover accuracy

print("Fine-tuning pruned network...\n")
finetune_losses, finetune_accs = train_network(
    pruned_model, X_train, y_train, X_test, y_test, epochs=50, lr=0.005
)

acc_finetuned = pruned_model.accuracy(X_test, y_test)
total, active = pruned_model.count_parameters()

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"Baseline:      {baseline_acc:.2%} accuracy, {total_params} params")
print(f"Pruned 50%:    {acc_finetuned:.2%} accuracy, {active} params")
print(f"Compression:  {total_params/active:.1f}x smaller")
print(f"Acc. change:  {(acc_finetuned - baseline_acc):+.2%}")
print(f"{'='*60}")
      
# Iterative Pruning
# Gradually increase pruning rate

def iterative_pruning(model, X_train, y_train, X_test, y_test, 
                      target_sparsity=0.9, num_iterations=5):
    """
    Iteratively prune and finetune
    """
    results = []
    
    # Initial state
    total, active = model.count_parameters()
    acc = model.accuracy(X_test, y_test)
    results.append({
        'iteration': 0,
        'sparsity': 0.0,
        'active_params': active,
        'accuracy': acc
    })
    
    # Gradually increase sparsity
    for i in range(num_iterations):
        # Sparsity for this iteration
        current_sparsity = target_sparsity * (i + 1) / num_iterations
        
        print(f"\nIteration {i+1}/{num_iterations}: Target sparsity {current_sparsity:.1%}")
        
        # Prune
        prune_by_magnitude(model, pruning_rate=current_sparsity)
        
        # Finetune
        train_network(model, X_train, y_train, X_test, y_test, epochs=30, lr=0.005)
        
        # Record results
        total, active = model.count_parameters()
        acc = model.accuracy(X_test, y_test)
        results.append({
            'iteration': i + 1,
            'sparsity': current_sparsity,
            'active_params': active,
            'accuracy': acc
        })
    
    return results

# Run iterative pruning
iterative_model = copy.deepcopy(baseline_model)
results = iterative_pruning(iterative_model, X_train, y_train, X_test, y_test, 
                            target_sparsity=0.95, num_iterations=5)
      
# Visualize Pruning Results

# Extract data
sparsities = [r['sparsity'] for r in results]
accuracies = [r['accuracy'] for r in results]
active_params = [r['active_params'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs Sparsity
ax1.plot(sparsities, accuracies, 'o-', linewidth=2, markersize=10, color='steelblue')
ax1.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label='Baseline')
ax1.set_xlabel('Sparsity (Fraction Pruned)', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('Accuracy vs Sparsity', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim([0, 1])

# Parameters vs Accuracy
ax2.plot(active_params, accuracies, 's-', linewidth=2, markersize=10, color='darkgreen')
ax2.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label='Baseline')
ax2.set_xlabel('Active Parameters', fontsize=12)
ax2.set_ylabel('Test Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_ylim([0, 1])
ax2.invert_xaxis()  # Fewer params on right

plt.tight_layout()
plt.show()

print("\nKey observation: Can remove 90%+ of weights with minimal accuracy loss!")
      
# Visualize Weight Distributions

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Baseline weights
axes[0, 0].hist(baseline_model.W1.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Baseline W1 Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Weight Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(baseline_model.W2.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Baseline W2 Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Weight Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Pruned weights (only active)
pruned_W1 = iterative_model.W1[iterative_model.mask1 > 0]
pruned_W2 = iterative_model.W2[iterative_model.mask2 > 0]

axes[1, 0].hist(pruned_W1.flatten(), bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Pruned W1 Distribution (Active Weights Only)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Weight Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(pruned_W2.flatten(), bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Pruned W2 Distribution (Active Weights Only)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Weight Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Pruned weights have larger magnitudes (small weights removed)")
      
# Visualize Sparsity Patterns

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# W1 sparsity pattern
im1 = ax1.imshow(iterative_model.mask1.T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
ax1.set_xlabel('Input Dimension', fontsize=12)
ax1.set_ylabel('Hidden Dimension', fontsize=12)
ax1.set_title('W1 Sparsity Pattern (Green=Active, Red=Pruned)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1)

# W2 sparsity pattern
im2 = ax2.imshow(iterative_model.mask2.T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
ax2.set_xlabel('Hidden Dimension', fontsize=12)
ax2.set_ylabel('Output Dimension', fontsize=12)
ax2.set_title('W2 Sparsity Pattern (Green=Active, Red=Pruned)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()

total, active = iterative_model.count_parameters()
print(f"\nFinal sparsity: {(total - active) / total:.1%}")
print(f"Compression ratio: {total / active:.1f}x")
      
# MDL Principle
# Minimum Description Length: Simpler models generalize better

def compute_mdl(model, X_train, y_train):
    """
    Simplified MDL computation
    
    MDL = Model Cost + Data Cost
    - Model Cost: Bits to encode weights
    - Data Cost: Bits to encode errors
    """
    # Model cost: number of parameters (simplified)
    total, active = model.count_parameters()
    model_cost = active  # Each param = 1 "bit" (simplified)
    
    # Data cost: cross-entropy loss
    probs = model.forward(X_train)
    y_one_hot = np.zeros((len(y_train), model.output_dim))
    y_one_hot[np.arange(len(y_train)), y_train] = 1
    data_cost = -np.sum(y_one_hot * np.log(probs + 1e-8))
    
    total_cost = model_cost + data_cost
    
    return {
        'model_cost': model_cost,
        'data_cost': data_cost,
        'total_cost': total_cost
    }

# Compare MDL for different models
baseline_mdl = compute_mdl(baseline_model, X_train, y_train)
pruned_mdl = compute_mdl(iterative_model, X_train, y_train)

print("MDL Comparison:")
print(f"{'='*60}")
print(f"{'Model':<20} {'Model Cost':<15} {'Data Cost':<15} {'Total'}")
print(f"{'-'*60}")
print(f"{'Baseline':<20} {baseline_mdl['model_cost']:<15.0f} {baseline_mdl['data_cost']:<15.2f} {baseline_mdl['total_cost']:.2f}")
print(f"{'Pruned (95%)':<20} {pruned_mdl['model_cost']:<15.0f} {pruned_mdl['data_cost']:<15.2f} {pruned_mdl['total_cost']:.2f}")
print(f"{'='*60}")
print(f"\nPruned model has LOWER total cost → Better generalization!")
      
# Key Takeaways
# Neural Network Pruning:
# Core Idea: Remove unnecessary weights to create simpler, smaller networks
# Magnitude-Based Pruning:
# 1.	Train network normally
# 2.	Identify low-magnitude weights: 
# 3.	Remove these weights (set to 0, mask out)
# 4.	Fine-tune remaining weights
# Iterative Pruning:
# Better than one-shot:
# for iteration in 1..N:
# 	prune small fraction (e.g., 20%)
# 	finetune
# Allows network to adapt gradually.
# Results (Typical):
# •	50% sparsity: Usually no accuracy loss
# •	90% sparsity: Slight accuracy loss (<2%)
# •	95%+ sparsity: Noticeable degradation
# Modern networks (ResNets, Transformers) can often be pruned to 90-95% sparsity with minimal impact!
# MDL Principle:
# Occam's Razor: Simplest explanation (smallest network) that fits data is best.
# Benefits of Pruning:
# 1.	Smaller models: Less memory, faster inference
# 2.	Better generalization: Removing overfitting parameters
# 3.	Energy efficiency: Fewer operations
# 4.	Interpretability: Simpler structure
# Types of Pruning:
# Type	What's Removed	Speedup
# Unstructured	Individual weights	Low (sparse ops)
# Structured	Entire neurons/filters	High (dense ops)
# Channel	Entire channels	High
# Layer	Entire layers	Very High
# Modern Techniques:
# 1.	Lottery Ticket Hypothesis:
# •	Pruned networks can be retrained from initialization
# •	"Winning tickets" exist in random init
# 2.	Dynamic Sparse Training:
# •	Prune during training (not after)
# •	Regrow connections
# 3.	Magnitude + Gradient:
# •	Use gradient info, not just magnitude
# •	Remove weights with small magnitude AND small gradient
# 4.	Learnable Sparsity:
# •	L0/L1 regularization
# •	Automatic sparsity discovery
# Practical Tips:
# 1.	Start high, prune gradually: Don't prune 90% immediately
# 2.	Fine-tune after pruning: Critical for recovery
# 3.	Layer-wise pruning rates: Different layers have different redundancy
# 4.	Structured pruning for speed: Unstructured needs special hardware
# When to Prune:
# ✅ Good for:
# •	Deployment (edge devices, mobile)
# •	Reducing inference cost
# •	Model compression
# ❌ Not ideal for:
# •	Very small models (already efficient)
# •	Training speedup (structured pruning only)
# Compression Rates in Practice:
# •	AlexNet: 9x compression (no accuracy loss)
# •	VGG-16: 13x compression
# •	ResNet-50: 5-7x compression
# •	BERT: 10-40x compression (with quantization)
# Key Insight:
# Neural networks are massively over-parameterized!
# Most weights contribute little to final performance. Pruning reveals the "core" network that does the real work.
# "The best model is the simplest one that fits the data" - MDL Principle

```

```python [Torch]
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration (support CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------
# 1. Prunable Network Implementation (Equivalent to SimpleNN)
# ----------------------------------------------------------------
class PrunableNN(nn.Module):
    """
    Torch equivalent of SimpleNN.
    Uses nn.Linear but maintains binary masks to simulate pruning.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PrunableNN, self).__init__()
        # Standard Linear layers
        # Note: torch.nn.Linear stores weights as (out_features, in_features)
        # This is the transpose of Numpy's (in_features, out_features)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize small weights to match Numpy code's behavior (* 0.1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc2.bias)

        # Register masks as buffers (saved in state_dict but not optimized)
        self.register_buffer('mask1', torch.ones_like(self.fc1.weight))
        self.register_buffer('mask2', torch.ones_like(self.fc2.weight))

    def forward(self, x):
        # Apply masks during forward pass (Soft Pruning for Training)
        # We mask weights *before* computation
        masked_w1 = self.fc1.weight * self.mask1
        masked_w2 = self.fc2.weight * self.mask2
        
        # Functional linear call to use masked weights
        # Note: F.linear computes x @ w.T + b
        h = torch.relu(torch.nn.functional.linear(x, masked_w1, self.fc1.bias))
        logits = torch.nn.functional.linear(h, masked_w2, self.fc2.bias)
        return logits # Return logits for stability (CrossEntropyLoss includes softmax)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def count_parameters(self):
        """Count total and active (non-zero mask) parameters"""
        total = self.fc1.weight.numel() + self.fc1.bias.numel() + \
                self.fc2.weight.numel() + self.fc2.bias.numel()
        
        # Active weights (based on mask) + all biases (biases not pruned in this version)
        active = self.mask1.sum().item() + self.fc1.bias.numel() + \
                 self.mask2.sum().item() + self.fc2.bias.numel()
        return int(total), int(active)

# ----------------------------------------------------------------
# 2. Data Generation (Equivalent to generate_classification_data)
# ----------------------------------------------------------------
def generate_data_torch(n_samples=1000, n_features=20, n_classes=3):
    # Reuse Numpy logic for generation to ensure identical data distribution
    # Then convert to Tensor
    samples_per_class = n_samples // n_classes
    X_list, y_list = [], []
    
    for c in range(n_classes):
        center = np.random.randn(n_features) * 3
        X_c = np.random.randn(samples_per_class, n_features) + center
        y_c = np.full(samples_per_class, c)
        X_list.append(X_c)
        y_list.append(y_c)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Convert to Tensor
    return (torch.tensor(X, dtype=torch.float32).to(device), 
            torch.tensor(y, dtype=torch.long).to(device))

# ----------------------------------------------------------------
# 3. Training Loop (Equivalent to train_network)
# ----------------------------------------------------------------
def train_model_torch(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    # Optimizer (SGD matches Numpy implementation)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    test_accs = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        loss.backward()
        
        # CRITICAL: Zero out gradients for pruned weights
        # This prevents pruned weights from updating (becoming non-zero)
        with torch.no_grad():
            model.fc1.weight.grad *= model.mask1
            model.fc2.weight.grad *= model.mask2
        
        optimizer.step()
        
        # Track metrics
        train_losses.append(loss.item())
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model.predict(X_test)
            acc = (preds == y_test).float().mean().item()
            test_accs.append(acc)
        model.train()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Acc: {acc:.2%}")
            
    return train_losses, test_accs

# ----------------------------------------------------------------
# 4. Pruning Logic (Equivalent to prune_by_magnitude)
# ----------------------------------------------------------------
def prune_weights_torch(model, pruning_rate):
    """Prune by magnitude globally across both layers"""
    # 1. Gather all weights to find global threshold
    w1_flat = model.fc1.weight.data.abs().flatten()
    w2_flat = model.fc2.weight.data.abs().flatten()
    all_weights = torch.cat([w1_flat, w2_flat])
    
    # 2. Compute quantile/percentile for threshold
    threshold = torch.quantile(all_weights, pruning_rate)
    
    # 3. Update masks
    # Note: We use .gt() (greater than) to keep large weights
    model.mask1 = model.fc1.weight.data.abs().gt(threshold).float()
    model.mask2 = model.fc2.weight.data.abs().gt(threshold).float()
    
    # 4. Enforce pruning immediately on weights (optional but cleaner)
    with torch.no_grad():
        model.fc1.weight.data *= model.mask1
        model.fc2.weight.data *= model.mask2

    total, active = model.count_parameters()
    print(f"Pruning threshold: {threshold:.6f}")
    print(f"Remaining parameters: {active}/{total} ({active/total:.1%})")

# ----------------------------------------------------------------
# Main Execution Block (Matching Numpy workflow)
# ----------------------------------------------------------------
# Data
X_train, y_train = generate_data_torch(1000)
X_test, y_test = generate_data_torch(300)

# Baseline
print("Training Baseline (Torch)...")
model = PrunableNN(20, 50, 3).to(device)
train_model_torch(model, X_train, y_train, X_test, y_test, epochs=100)
base_total, base_active = model.count_parameters()

# Iterative Pruning
print("\nIterative Pruning (Torch)...")
pruned_model = copy.deepcopy(model)
target_sparsity = 0.95
num_iterations = 5

for i in range(num_iterations):
    current_sparsity = target_sparsity * (i + 1) / num_iterations
    print(f"\nIteration {i+1}/{num_iterations}: Target {current_sparsity:.1%}")
    
    prune_weights_torch(pruned_model, current_sparsity)
    
    # Finetune (Learning rate matches Numpy)
    train_model_torch(pruned_model, X_train, y_train, X_test, y_test, epochs=30, lr=0.005)

# Final stats
final_total, final_active = pruned_model.count_parameters()
print(f"\nFinal Compression: {final_total/final_active:.1f}x")

```

:::

### 6.3 对照讲解与差异分析

1. **Tensor 形状与转置 (Shape & Transpose)**
* **Numpy**: 使用 `np.dot(X, W)`，权重形状为 `(input_dim, output_dim)`。
* **Torch**: `nn.Linear` 默认存储权重为 `(output_dim, input_dim)`，计算时执行 。
* **影响**: 这是一个常见的“坑”。如果直接将 Numpy 权重赋值给 Torch 模型，必须记得转置 `.T`。我在 Torch 实现中保持了 `nn.Linear` 的惯用写法，但逻辑上等价。


2. **梯度掩码 (Gradient Masking)**
* **Numpy**: 在更新步骤显式写出 `model.W1 -= lr * dL_dW1 * model.mask1`。
* **Torch**: 优化器（`optim.SGD`）通常不知道 mask 的存在。为了实现等价效果，必须在 `loss.backward()` 之后、`optimizer.step()` 之前，手动将 mask 乘到梯度上：`weight.grad *= mask`。这确保了被剪枝的权重梯度为 0，不会复活。


3. **损失函数数值稳定性**
* **Numpy**: 手动实现了 Softmax 和 CrossEntropy (`-np.mean(np.sum(...))`)，并加了 `1e-8` 防止 Log(0)。
* **Torch**: 使用 `nn.CrossEntropyLoss()`，它在内部结合了 `LogSoftmax` 和 `NLLLoss`，使用了 `log-sum-exp`技巧，数值上比 Numpy 版本更稳定，尤其是在半精度（FP16）下。


4. **剪枝阈值计算**
* **Numpy**: 使用 `np.percentile`。
* **Torch**: 使用 `torch.quantile`。两者功能一致，但 `quantile` 接受 0-1 之间的小数，`percentile` 接受 0-100 的整数，需要注意参数单位。


5. **MDL 的定义差异**
* **代码实现**：Numpy 代码中使用 `model_cost = active_params`，这其实是 **AIC/BIC** (赤池/贝叶斯信息准则) 的简化版，假设每个参数的编码长度固定。
* **论文理论**：Hinton 论文使用的是 **KL 散度**。Numpy 代码是对 MDL 原则的“工程近似”（参数越少 = 描述越短），而非论文公式的直接复现。



```

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`05 keep nn simple.pdf`

![Pruning & Compression 图 1](/paper-figures/05/img-000.png)
*图 1：建议结合本节 `可压缩神经网络` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**Pruning & Compression**（围绕 `可压缩神经网络`）

### 一、选择题（10题）

1. 在 Pruning & Compression 中，最关键的建模目标是什么？
   - A. 可压缩神经网络
   - B. 剪枝
   - C. 量化
   - D. Bits Back
   - **答案：A**

2. 下列哪一项最直接对应 Pruning & Compression 的核心机制？
   - A. 剪枝
   - B. 量化
   - C. Bits Back
   - D. KL
   - **答案：B**

3. 在复现 Pruning & Compression 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 Pruning & Compression，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 Pruning & Compression 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. Pruning & Compression 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 Pruning & Compression 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 Pruning & Compression 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，Pruning & Compression 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，Pruning & Compression 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 Pruning & Compression 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 Pruning & Compression 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `剪枝`。
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

8. 写一个小型单元测试，验证 `量化` 相关张量形状正确。
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

