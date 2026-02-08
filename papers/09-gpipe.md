# 论文解读：GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

## 1. 一句话概述
GPipe 是谷歌提出的一种**流水线并行（Pipeline Parallelism）**库，通过将神经网络层切分为多个分区（Partitions）并引入**微批处理（Micro-batching）**和**重计算（Re-materialization）**技术，使得在多加速器上训练超大规模模型（如 AmoebaNet 和 Transformer）成为可能，且实现了近乎线性的扩展效率。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？
**试图解决的问题**：
在深度学习中，扩大模型容量（Capacity）是提升质量的有效手段 。然而，单个加速器（GPU/TPU）的显存限制了模型的大小 。传统的模型并行（Model Parallelism）方案通常难以设计，且针对特定架构（architecture-specific），缺乏通用性和灵活性 。

**主要贡献**：
1.  **GPipe 库**：提出了一种通用的流水线并行库，可以将任何由层序列组成的网络扩展到巨大规模 。
2.  **核心算法**：利用微批处理（batch-splitting）和重计算技术，大幅减少气泡（Idle time）和显存占用，实现了随设备数量增加的近乎线性加速 。
3.  **实验验证**：
    * **图像分类**：训练了 5.57 亿参数的 **AmoebaNet**，在 ImageNet-2012 上达到了 84.4% 的 top-1 准确率 。
    * **机器翻译**：训练了 60 亿参数的 **Transformer**（128 层），在多语言翻译任务上超越了所有双语模型 。

## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑
**动机（Motivation）**：
深度学习的进步在很大程度上归功于模型容量的扩大，这在图像分类（ImageNet）和 NLP 领域都得到了验证（如 Figure 1 所示）。然而，硬件限制（内存和通信带宽）迫使研究者必须将模型分割到不同设备上 。现有的模型并行算法极其复杂，需要在容量、灵活性和效率之间做艰难权衡，且往往不可移植 。

**故事逻辑**：
1.  **需求**：随着应用场景增加，迫切需要一种可靠、灵活的基础设施来轻松扩展各种神经网络 。
2.  **GPipe 方案**：
    * 将模型视为层的序列（sequence of layers），将其划分为多个单元（cells）放置在不同加速器上 。
    * **流水线化（Pipelining）**：将一个 mini-batch 切分为更小的 micro-batches，使不同设备能同时处理不同的 micro-batch，从而并行工作 。
    * **同步更新**：采用同步梯度下降，保证了无论分区数量如何，梯度的更新逻辑在数学上是一致的 。
3.  **验证**：通过 AmoebaNet 和 Transformer 两个截然不同的架构，证明了 GPipe 的灵活性和高效性 。

## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

### 4.1 核心架构与接口
GPipe 将任何深层网络定义为 $L$ 层序列。用户只需指定分区数量 $K$ 和微批次数量 $M$。
* **分区（Partitioning）**：网络被划分为 $K$ 个复合层（cells），第 $k$ 个 cell 放置在第 $k$ 个加速器上 。
* **成本估算**：算法尝试最小化各分区计算成本的方差，以平衡负载 。

### 4.2 流水线并行与微批处理 (Micro-batching)
为了解决朴素模型并行中的设备空闲问题（Figure 2b），GPipe 将大小为 $N$ 的 mini-batch 切分为 $M$ 个微批次（micro-batches）。
* **执行流**：微批次依次流经 $K$ 个加速器。
* **气泡开销（Bubble Overhead）**：流水线的启动和结束阶段会有设备空闲。气泡时间分数为：
    $$O\left(\frac{K-1}{M+K-1}\right)$$
    
    当 $M \ge 4K$ 时，气泡开销几乎可以忽略不计 。

### 4.3 激活重计算 (Re-materialization)
为了突破显存限制，GPipe 结合了梯度检查点技术（Gradient Checkpointing）。
* **前向传播**：每个加速器只存储分区边界处的激活值（activation tensors at partition boundaries），丢弃中间层的激活值 。
* **反向传播**：在计算梯度时，第 $k$ 个加速器利用边界输入重新计算前向函数 $F_k$ 。
* **内存优化**：峰值激活内存需求从 $O(N \times L)$ 降低到：
    $$O\left(N + \frac{L}{K} \times \frac{N}{M}\right)$$
    
    这里 $L/K$ 是单分区的层数，$N/M$ 是微批次大小。这使得单卡可以承载更大的模型。

### 4.4 逻辑框图 (Mermaid)

```mermaid
graph TD
    subgraph User Input
    A[定义模型: L 层序列] --> B[配置: K 个分区, M 个微批次]
    end

    subgraph GPipe Setup
    B --> C[自动分区: 将层分配到 Device 1...K]
    C --> D[插入通信原语]
    end

    subgraph Training Step (Mini-Batch N)
    E[输入 Batch] --> F[切分为 M 个 Micro-Batches]
    F --> G[Pipeline Schedule]
    
    subgraph Pipelining (F-then-B Strategy)
    G --> H{Device 1}
    G --> I{Device 2}
    G --> J{Device ...K}
    H -- Forward uBatch 1..M --> I
    I -- Forward uBatch 1..M --> J
    J -- Backward uBatch M..1 --> I
    I -- Backward uBatch M..1 --> H
    end
    
    style H fill:#f9f,stroke:#333
    style I fill:#ccf,stroke:#333
    
    H -.-> |Re-computation| H
    I -.-> |Re-computation| I
    end

    subgraph Update
    J --> K_grad[累积所有 uBatch 梯度]
    I --> K_grad
    H --> K_grad
    K_grad --> L_opt[同步应用梯度更新]
    end

```

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 图像分类 (Image Classification)

* 
**设置**：使用 AmoebaNet 模型，输入尺寸 ，在 ImageNet-2012 上训练 。


* **结果**：将模型参数扩展到 **5.57 亿（557M）**（AmoebaNet-B(18, 512)），分为 4 个分区。
* 
**指标**：实现了 **84.4%** 的 top-1 验证集准确率 。这是当时的 state-of-the-art 结果（图 1a 中的红点）。


* 
**迁移学习**：该大模型在 CIFAR-10 等数据集上的迁移学习效果也极其出色（CIFAR-10 错误率降至 1%）。



### 5.2 多语言机器翻译 (Multilingual Machine Translation)

* 
**设置**：使用 Transformer 模型，在 103 种语言（102 种语言对英语）的语料库上训练，共 250 亿个训练样本 。


* 
**结果**：训练了一个 **128 层、60 亿参数（6B）** 的 Transformer 模型（分 16 个分区）。


* 
**指标**：该单一模型在所有 100 个语言对上的表现都优于单独训练的双语模型（Bilingual Baselines）。


* 
**深度 vs 宽度**：实验发现，在同等参数量下，**更深**的模型（Deep）在低资源语言上的泛化能力优于更宽的模型（Wide）。



### 5.3 效率分析 (Performance Analysis)

* 
**加速比**：对于 Transformer 模型，当微批次数量 `m` 足够大时，吞吐量随加速器数量几乎线性增加（例如 4 倍设备带来约 3.5 倍加速）。


* 
**内存优化**：通过重计算，单卡可训练的模型大小提升了 2.7 倍 。结合流水线，Transformer 可扩展至 839 亿参数 。



## 6. Numpy 与 Torch 对照实现（含 code-group）

### 代码说明

这份 Numpy 代码完整实现了 **GPipe 的核心逻辑仿真**，主要包含：

1. **`Layer` 与 `Partition` 类**：模拟神经网络层和设备分区。
2. **`GPipePipeline` 类**：实现了关键的 **F-then-B 调度（Forward-then-Backward）**。这对应论文 Section 2.2 和 Figure 2c，即先把所有 `m` 个微批次前向推完，再反向。
3. **`GPipePipelineWithRemat` 类**：实现了**重计算（Re-materialization）**逻辑（论文 Section 2.3），仅存储边界激活值，反向时重算内部激活。
4. **`accumulate_gradients`**：实现了梯度的累积与平均（对应论文中的 micro-batch 梯度求和后再按 `m` 归一化）。

**数据维度假设**：

* 代码中 `X` shape 为 `(batch_size, input_dim)`。
* `Partition` 处理的是 `(micro_batch_size, input_dim)` 的切片。
* 该实现为“单线程模拟多设备”，并未真正跨 GPU 通信，但逻辑流程与真实 GPipe 一致。

### 对照实现

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import time
from collections import defaultdict

np.random.seed(42)

print("Libraries imported successfully!")
print("NumPy version:", np.__version__)

@dataclass
class Layer:
    """A single neural network layer."""
    W: np.ndarray  # Weight matrix
    b: np.ndarray  # Bias vector
    activation: str = 'relu'  # 'relu', 'tanh', or 'linear'
    
    def forward(self, x: np.ndarray, store_activation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass: z = W @ x + b, a = activation(z)"""
        z = x @ self.W + self.b  # Linear transformation
        
        # Apply activation function
        if self.activation == 'relu':
            a = np.maximum(0, z)
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'linear':
            a = z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        return a, z if store_activation else None
    
    def backward(self, da: np.ndarray, z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass: compute gradients."""
        # Activation gradient
        if self.activation == 'relu':
            dz = da * (z > 0)
        elif self.activation == 'tanh':
            dz = da * (1 - np.tanh(z)**2)
        elif self.activation == 'linear':
            dz = da
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        # Parameter gradients
        dW = x.T @ dz
        db = np.sum(dz, axis=0)
        
        # Input gradient (for previous layer)
        dx = dz @ self.W.T
        
        return dx, dW, db


@dataclass
class Partition:
    """A partition of the model (subset of layers assigned to one device)."""
    device_id: int
    layers: List[Layer]
    
    def forward(self, x: np.ndarray, store_activations: bool = True) -> Tuple[np.ndarray, List[Tuple]]:
        """Forward pass through all layers in this partition."""
        activations = []  # Store (x, z) for each layer if needed
        
        current = x
        for layer in self.layers:
            if store_activations:
                activations.append(current)  # Store input to this layer
            
            current, z = layer.forward(current, store_activation=store_activations)
            
            if store_activations:
                activations.append(z)  # Store pre-activation
        
        return current, activations
    
    def backward(self, dout: np.ndarray, activations: List) -> Tuple[np.ndarray, List[Tuple]]:
        """Backward pass through all layers in this partition."""
        gradients = []  # Store (dW, db) for each layer
        
        da = dout
        # Go through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Get stored activations
            x = activations[2*i]      # Input to this layer
            z = activations[2*i + 1]  # Pre-activation
            
            # Compute gradients
            da, dW, db = layer.backward(da, z, x)
            gradients.insert(0, (dW, db))
        
        return da, gradients  # da is gradient w.r.t. partition input


def create_model(layer_dims: List[int], activations: List[str]) -> List[Layer]:
    """Create a multi-layer neural network.
    
    Args:
        layer_dims: [input_dim, hidden1, hidden2, ..., output_dim]
        activations: Activation for each layer
    """
    layers = []
    for i in range(len(layer_dims) - 1):
        W = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
        b = np.zeros(layer_dims[i+1])
        layers.append(Layer(W, b, activations[i]))
    return layers


def partition_model(layers: List[Layer], num_partitions: int) -> List[Partition]:
    """Partition layers uniformly across devices."""
    num_layers = len(layers)
    layers_per_partition = num_layers // num_partitions
    
    partitions = []
    for k in range(num_partitions):
        start = k * layers_per_partition
        if k == num_partitions - 1:
            # Last partition gets any remaining layers
            end = num_layers
        else:
            end = (k + 1) * layers_per_partition
        
        partition_layers = layers[start:end]
        partitions.append(Partition(device_id=k, layers=partition_layers))
    
    return partitions


# Example: Create and partition a 12-layer network
layer_dims = [128] + [256] * 10 + [10]  # Input=128, 10 hidden layers of 256, output=10
activations = ['relu'] * 10 + ['linear']  # ReLU for hidden, linear for output

model_layers = create_model(layer_dims, activations)
print(f"Created model with {len(model_layers)} layers")

# Partition across 4 "devices"
K = 4
partitions = partition_model(model_layers, K)

print(f"\nPartitioned model into {K} partitions:")
for i, partition in enumerate(partitions):
    print(f"  Device {i}: {len(partition.layers)} layers")

print("\n✓ Model partitioning complete!")

def split_into_microbatches(X: np.ndarray, y: np.ndarray, num_microbatches: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split mini-batch into micro-batches.
    
    Args:
        X: Input data (batch_size, features)
        y: Labels (batch_size, ...)
        num_microbatches: M (number of micro-batches)
    
    Returns:
        List of (X_micro, y_micro) tuples
    """
    batch_size = X.shape[0]
    microbatch_size = batch_size // num_microbatches
    
    if batch_size % num_microbatches != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by num_microbatches {num_microbatches}")
    
    microbatches = []
    for m in range(num_microbatches):
        start = m * microbatch_size
        end = (m + 1) * microbatch_size
        microbatches.append((X[start:end], y[start:end]))
    
    return microbatches


def compute_bubble_fraction(K: int, M: int) -> float:
    """Theoretical bubble fraction for GPipe.
    
    Formula: (K - 1) / (K - 1 + M)
    
    Args:
        K: Number of devices/partitions
        M: Number of micro-batches
    """
    return (K - 1) / (K - 1 + M)


# Example: Analyze bubble fraction
K_values = [2, 4, 8, 16]
M_values = [1, 2, 4, 8, 16, 32, 64]

print("Bubble Fraction Analysis:")
print("\nM (micro-batches) →")
print("K ↓\t" + "\t".join(f"{M:d}" for M in M_values))
print("-" * 80)

for K in K_values:
    row = f"{K}\t"
    for M in M_values:
        bubble = compute_bubble_fraction(K, M)
        row += f"{bubble:.3f}\t"
    print(row)

print("\nKey observations:")
print("  - More devices (K) → more bubble time (devices wait for pipeline)")
print("  - More micro-batches (M) → less bubble time (pipeline stays full)")
print("  - With K=4, M=8: bubble fraction = 27.3% (device idle 27% of time)")
print("  - With K=4, M=32: bubble fraction = 8.6% (much better!)")

# Example micro-batching
batch_size = 32
M = 8
X_batch = np.random.randn(batch_size, 128)
y_batch = np.random.randint(0, 10, batch_size)

microbatches = split_into_microbatches(X_batch, y_batch, M)
print(f"\n\nSplit batch of {batch_size} into {M} micro-batches:")
for i, (X_m, y_m) in enumerate(microbatches):
    print(f"  Micro-batch {i}: X shape {X_m.shape}, y shape {y_m.shape}")

print("\n✓ Micro-batching complete!")

@dataclass
class PipelineEvent:
    """Records when a device executes an operation."""
    time_step: int
    device_id: int
    operation: str  # 'forward' or 'backward'
    microbatch_id: int


class GPipePipeline:
    """GPipe pipeline with F-then-B schedule."""
    
    def __init__(self, partitions: List[Partition]):
        self.partitions = partitions
        self.K = len(partitions)  # Number of devices
        
        # For tracking execution timeline
        self.events = []  # List of PipelineEvent
    
    def forward_pipeline(self, microbatches: List[Tuple[np.ndarray, np.ndarray]], 
                        store_activations: bool = True) -> Tuple[List[np.ndarray], List[List]]:
        """Forward pass: process all micro-batches through pipeline.
        
        Returns:
            outputs: List of final outputs for each micro-batch
            all_activations: List of activation lists (one per micro-batch)
        """
        M = len(microbatches)
        
        # Storage for outputs and activations
        outputs = [None] * M
        all_activations = [[None] * self.K for _ in range(M)]  # [microbatch][partition]
        
        # F-then-B schedule: Forward all micro-batches
        time_step = 0
        
        for m in range(M):
            X_micro, y_micro = microbatches[m]
            current = X_micro
            
            # Forward through each partition
            for k, partition in enumerate(self.partitions):
                self.events.append(PipelineEvent(time_step, k, 'forward', m))
                
                current, activations = partition.forward(current, store_activations)
                all_activations[m][k] = activations
                
                time_step += 1
            
            outputs[m] = current
        
        return outputs, all_activations
    
    def backward_pipeline(self, outputs: List[np.ndarray], 
                         labels: List[np.ndarray],
                         all_activations: List[List]) -> List[List[List[Tuple]]]:
        """Backward pass: process all micro-batches in reverse.
        
        Returns:
            all_gradients: [microbatch][partition][(dW, db) for each layer]
        """
        M = len(outputs)
        
        # Storage for gradients
        all_gradients = [[None] * self.K for _ in range(M)]
        
        # Find current time step (after forward passes)
        time_step = max(e.time_step for e in self.events) + 1
        
        # Backward all micro-batches in reverse order
        for m in range(M - 1, -1, -1):
            # Compute loss gradient (simple MSE for demonstration)
            dout = 2 * (outputs[m] - labels[m]) / labels[m].shape[0]
            
            # Backward through each partition in reverse
            for k in range(self.K - 1, -1, -1):
                partition = self.partitions[k]
                activations = all_activations[m][k]
                
                self.events.append(PipelineEvent(time_step, k, 'backward', m))
                
                dout, gradients = partition.backward(dout, activations)
                all_gradients[m][k] = gradients
                
                time_step += 1
        
        return all_gradients
    
    def get_timeline_matrix(self) -> np.ndarray:
        """Convert events to a K×T matrix for visualization.
        
        Matrix values:
            0 = bubble (idle)
            m+1 = forward micro-batch m
            -(m+1) = backward micro-batch m
        """
        max_time = max(e.time_step for e in self.events) + 1
        timeline = np.zeros((self.K, max_time))
        
        for event in self.events:
            value = event.microbatch_id + 1
            if event.operation == 'backward':
                value = -value
            timeline[event.device_id, event.time_step] = value
        
        return timeline


# Test forward pass
print("Testing GPipe forward pass...\n")

# Create pipeline
pipeline = GPipePipeline(partitions)

# Create micro-batches
M = 4
batch_size = 16
X_batch = np.random.randn(batch_size, 128)
y_batch_onehot = np.eye(10)[np.random.randint(0, 10, batch_size)]

microbatches = split_into_microbatches(X_batch, y_batch_onehot, M)

# Forward pass
outputs, all_activations = pipeline.forward_pipeline(microbatches)

print(f"Processed {M} micro-batches through {pipeline.K} devices")
print(f"Output shapes: {[out.shape for out in outputs]}")
print(f"Total forward events: {len([e for e in pipeline.events if e.operation == 'forward'])}")

# Backward pass
labels = [mb[1] for mb in microbatches]
all_gradients = pipeline.backward_pipeline(outputs, labels, all_activations)

print(f"Total backward events: {len([e for e in pipeline.events if e.operation == 'backward'])}")
print(f"\nTotal time steps: {max(e.time_step for e in pipeline.events) + 1}")

print("\n✓ Pipeline forward and backward passes complete!")

def accumulate_gradients(all_gradients: List[List[List[Tuple]]]) -> List[List[Tuple]]:
    """Accumulate and average gradients from all micro-batches.
    
    Args:
        all_gradients: [microbatch][partition][(dW, db) per layer]
    
    Returns:
        accumulated: [partition][(dW, db) per layer] - averaged over micro-batches
    """
    M = len(all_gradients)  # Number of micro-batches
    K = len(all_gradients[0])  # Number of partitions
    
    # Initialize accumulated gradients (copy structure from first micro-batch)
    accumulated = []
    for k in range(K):
        partition_grads = []
        for layer_idx in range(len(all_gradients[0][k])):
            # Sum gradients across micro-batches
            dW_sum = sum(all_gradients[m][k][layer_idx][0] for m in range(M))
            db_sum = sum(all_gradients[m][k][layer_idx][1] for m in range(M))
            
            # Average (since micro-batches are part of same mini-batch)
            dW_avg = dW_sum / M
            db_avg = db_sum / M
            
            partition_grads.append((dW_avg, db_avg))
        
        accumulated.append(partition_grads)
    
    return accumulated


def apply_gradients(partitions: List[Partition], gradients: List[List[Tuple]], learning_rate: float):
    """Apply accumulated gradients to update parameters.
    
    Args:
        partitions: List of model partitions
        gradients: [partition][(dW, db) per layer]
        learning_rate: Learning rate for SGD
    """
    for k, partition in enumerate(partitions):
        partition_grads = gradients[k]
        
        for layer_idx, layer in enumerate(partition.layers):
            dW, db = partition_grads[layer_idx]
            
            # SGD update
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db


# Test gradient accumulation
print("Testing gradient accumulation...\n")

# We already have all_gradients from previous cell
accumulated_grads = accumulate_gradients(all_gradients)

print(f"Accumulated gradients for {len(accumulated_grads)} partitions:")
for k, partition_grads in enumerate(accumulated_grads):
    print(f"  Partition {k}: {len(partition_grads)} layers")
    for i, (dW, db) in enumerate(partition_grads[:2]):  # Show first 2 layers
        print(f"    Layer {i}: dW shape {dW.shape}, db shape {db.shape}")
        print(f"             dW norm: {np.linalg.norm(dW):.6f}, db norm: {np.linalg.norm(db):.6f}")

# Apply gradients
learning_rate = 0.01
old_W = partitions[0].layers[0].W.copy()

apply_gradients(partitions, accumulated_grads, learning_rate)

new_W = partitions[0].layers[0].W
weight_change = np.linalg.norm(new_W - old_W)

print(f"\nApplied gradients with learning rate {learning_rate}")
print(f"Weight change (first layer): {weight_change:.6f}")

print("\n✓ Gradient accumulation and application complete!")

class GPipePipelineWithRemat:
    """GPipe with re-materialization (gradient checkpointing)."""
    
    def __init__(self, partitions: List[Partition]):
        self.partitions = partitions
        self.K = len(partitions)
        self.events = []
    
    def forward_pipeline_remat(self, microbatches: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[List, List]:
        """Forward pass with re-materialization: only store partition boundary activations.
        
        Returns:
            outputs: Final outputs for each micro-batch
            boundary_inputs: Inputs to each partition (for recomputation)
        """
        M = len(microbatches)
        
        outputs = [None] * M
        # Only store inputs to each partition (boundary activations)
        boundary_inputs = [[None] * self.K for _ in range(M)]
        
        time_step = 0
        
        for m in range(M):
            X_micro, y_micro = microbatches[m]
            current = X_micro
            
            for k, partition in enumerate(self.partitions):
                # Store input to this partition (boundary)
                boundary_inputs[m][k] = current.copy()
                
                self.events.append(PipelineEvent(time_step, k, 'forward', m))
                
                # Forward pass WITHOUT storing intermediate activations
                current, _ = partition.forward(current, store_activations=False)
                
                time_step += 1
            
            outputs[m] = current
        
        return outputs, boundary_inputs
    
    def backward_pipeline_remat(self, outputs: List[np.ndarray],
                                labels: List[np.ndarray],
                                boundary_inputs: List[List]) -> List[List[List[Tuple]]]:
        """Backward pass with re-materialization: recompute activations as needed."""
        M = len(outputs)
        all_gradients = [[None] * self.K for _ in range(M)]
        
        time_step = max(e.time_step for e in self.events) + 1
        
        for m in range(M - 1, -1, -1):
            dout = 2 * (outputs[m] - labels[m]) / labels[m].shape[0]
            
            for k in range(self.K - 1, -1, -1):
                partition = self.partitions[k]
                
                self.events.append(PipelineEvent(time_step, k, 'backward', m))
                
                # RECOMPUTE activations for this partition
                partition_input = boundary_inputs[m][k]
                _, activations = partition.forward(partition_input, store_activations=True)
                
                # Now compute gradients using recomputed activations
                dout, gradients = partition.backward(dout, activations)
                all_gradients[m][k] = gradients
                
                time_step += 1
        
        return all_gradients


def estimate_memory_usage(M: int, K: int, layers_per_partition: int, 
                          activation_size_mb: float, with_remat: bool) -> float:
    """Estimate memory usage with and without re-materialization.
    
    Args:
        M: Number of micro-batches
        K: Number of partitions
        layers_per_partition: Average layers per partition
        activation_size_mb: Memory for one layer's activations (MB)
        with_remat: Use re-materialization?
    
    Returns:
        Estimated memory in MB
    """
    if with_remat:
        # Only store boundary inputs (K per micro-batch)
        return M * K * activation_size_mb
    else:
        # Store all intermediate activations
        total_layers = K * layers_per_partition
        return M * total_layers * activation_size_mb


# Test re-materialization
print("Testing re-materialization...\n")

# Create fresh pipeline with remat
pipeline_remat = GPipePipelineWithRemat(partitions)

# Forward with remat
outputs_remat, boundary_inputs = pipeline_remat.forward_pipeline_remat(microbatches)

print("Forward pass with re-materialization:")
print(f"  Stored boundary inputs: {len(boundary_inputs)} micro-batches × {len(boundary_inputs[0])} partitions")
print(f"  Boundary input shapes: {[bi[0].shape for bi in boundary_inputs]}")

# Backward with remat
gradients_remat = pipeline_remat.backward_pipeline_remat(outputs_remat, labels, boundary_inputs)

print(f"\nBackward pass with re-materialization:")
print(f"  Gradients computed: {len(gradients_remat)} micro-batches × {len(gradients_remat[0])} partitions")

# Memory analysis
print("\n" + "="*70)
print("Memory Usage Comparison")
print("="*70)

M_test = 8
K_test = 4
layers_per_partition = 3
activation_size_mb = 10  # MB per layer activation

mem_without = estimate_memory_usage(M_test, K_test, layers_per_partition, activation_size_mb, with_remat=False)
mem_with = estimate_memory_usage(M_test, K_test, layers_per_partition, activation_size_mb, with_remat=True)

print(f"\nConfiguration: M={M_test}, K={K_test}, {layers_per_partition} layers/partition")
print(f"  Without re-materialization: {mem_without:.1f} MB")
print(f"  With re-materialization:    {mem_with:.1f} MB")
print(f"  Memory savings:             {mem_without / mem_with:.1f}×")

print("\n✓ Re-materialization complete!")

```

```python [Torch]
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from torch.utils.checkpoint import checkpoint

# 对应 Numpy 代码段 1: Layer 与 Partition
# 在 Torch 中，Partition 直接映射为 nn.Sequential
# Layer 直接映射为 nn.Linear + Activation

class GPipeModel(nn.Module):
    """
    一个用于单机模拟 GPipe 逻辑的 Torch 模型封装。
    实现了:
    1. 分区 (Partitioning)
    2. 微批处理 (Micro-batching)
    3. 重计算 (Re-materialization via checkpoint)
    """
    def __init__(self, layer_dims: List[int], activations: List[str], num_partitions: int):
        super().__init__()
        self.num_partitions = num_partitions
        
        # 1. 构建完整层序列 (对应 create_model)
        full_layers = []
        for i in range(len(layer_dims) - 1):
            layer = nn.Linear(layer_dims[i], layer_dims[i+1])
            # 初始化以匹配 Numpy (高斯分布)
            nn.init.normal_(layer.weight, mean=0.0, std=(2.0/layer_dims[i])**0.5)
            nn.init.zeros_(layer.bias)
            
            full_layers.append(layer)
            
            act_name = activations[i]
            if act_name == 'relu':
                full_layers.append(nn.ReLU())
            elif act_name == 'tanh':
                full_layers.append(nn.Tanh())
            # linear 无需添加模块
            
        # 2. 将层序列划分为 Partition (对应 partition_model)
        # 注意: Torch 的 Sequential 将层组合，这里我们手动按数量切分
        self.partitions = nn.ModuleList()
        num_modules = len(full_layers)
        # 简单均分逻辑 (注意 modules 包含 Linear 和 Activation，所以除以 2 近似)
        # 这里为了演示简单，直接按 module 索引切分
        modules_per_partition = (num_modules + num_partitions - 1) // num_partitions
        
        for k in range(num_partitions):
            start = k * modules_per_partition
            end = min((k + 1) * modules_per_partition, num_modules)
            if start < end:
                # 每个 Partition 是一个 nn.Sequential
                self.partitions.append(nn.Sequential(*full_layers[start:end]))
        
        self.K = len(self.partitions)

    def forward_partition(self, x: torch.Tensor, partition_idx: int, use_checkpoint: bool):
        """运行单个 Partition 的前向传播"""
        partition = self.partitions[partition_idx]
        if use_checkpoint:
            # 对应 Numpy: GPipePipelineWithRemat 的重计算逻辑
            # checkpoint(function, input) 会在反向时不保存中间激活，而是重算
            return checkpoint(partition, x, use_reentrant=False)
        else:
            return partition(x)

    def forward(self, x: torch.Tensor, num_microbatches: int = 1, with_remat: bool = True):
        """
        对应 Numpy: GPipePipeline.forward_pipeline + backward_pipeline 逻辑
        
        Args:
            x: Input batch (N, D)
            num_microbatches: M
            with_remat: 是否使用重计算
        
        Returns:
            outputs: 拼接后的最终输出
        """
        # 1. 微批处理切分 (对应 split_into_microbatches)
        # torch.chunk 将 tensor 在 dim 0 切分为 M 块
        micro_batches = torch.chunk(x, num_microbatches, dim=0)
        
        final_outputs = []
        
        # 2. Pipeline 执行 (F-then-B Schedule)
        # Torch 的 Autograd 会自动构建图，我们只需按顺序执行前向
        # 真实的 GPipe 会在不同 Device 并行，这里在单设备上模拟逻辑顺序
        
        # 为了模拟 "F-then-B" 的 Re-materialization 效果：
        # 我们遍历所有 micro-batch，依次通过所有 partition
        for mb in micro_batches:
            current = mb
            for k in range(self.K):
                # 只有需要求导时才用 checkpoint (推理模式不需要)
                should_checkpoint = with_remat and current.requires_grad
                current = self.forward_partition(current, k, should_checkpoint)
            final_outputs.append(current)
            
        # 重新拼接输出以便计算 Loss
        return torch.cat(final_outputs, dim=0)

# --- 使用示例 ---
# 配置参数
input_dim = 128
output_dim = 10
layer_dims = [input_dim] + [256] * 10 + [output_dim]
activations = ['relu'] * 10 + ['linear']
K = 4 # 分区数
M = 8 # 微批次

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPipeModel(layer_dims, activations, num_partitions=K).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建数据
batch_size = 32
X_batch = torch.randn(batch_size, input_dim).to(device)
y_batch = torch.randint(0, 10, (batch_size,)).to(device)

# 训练一步 (One Step)
optimizer.zero_grad()

# 前向传播 (GPipe 逻辑：切分 -> Checkpoint -> 拼接)
# 对应 Numpy: pipeline_remat.forward_pipeline_remat
outputs = model(X_batch, num_microbatches=M, with_remat=True)

# 计算 Loss
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, y_batch)

# 反向传播 
# 对应 Numpy: pipeline_remat.backward_pipeline_remat + accumulate_gradients
# Torch Autograd 会自动处理：
# 1. 沿着 checkpoint 的边界反向传播
# 2. 遇到 checkpoint 区域时，重新进行前向计算 (Re-computation)
# 3. 累积梯度到 model.parameters().grad
loss.backward()

# 参数更新
# 对应 Numpy: apply_gradients
optimizer.step()

print(f"Torch implementation output shape: {outputs.shape}")
print(f"Loss: {loss.item():.4f}")
print("✓ Torch GPipe logic equivalent run complete.")

```

:::

### 对照讲解

1. **自动微分 vs 手动反向**：
* **Numpy**：必须显式编写 `backward()` 函数，并手动管理梯度列表 `gradients` 和累积逻辑 `accumulate_gradients`。
* **Torch**：利用 `Autograd` 引擎。当我们对 micro-batches 的总 Loss 调用 `.backward()` 时，Torch 会自动遍历计算图。`optimizer.step()` 会处理参数更新。


2. **重计算 (Re-materialization)**：
* **Numpy**：在 `GPipePipelineWithRemat` 中显式实现：`forward` 只存边界，`backward` 时重新调用 `partition.forward`。
* **Torch**：直接使用 `torch.utils.checkpoint.checkpoint`。它在底层做了完全相同的事：前向时不保存中间激活，反向时重新运行前向部分来恢复数据。


3. **微批处理 (Micro-batching)**：
* **Numpy**：`split_into_microbatches` 手动切片列表。
* **Torch**：`torch.chunk(x, chunks=M)` 是高效的切片操作（View 操作，不拷贝内存）。


4. **容易写错的地方**：
* **Checkpoint 的输入**：在 Torch 中，传入 `checkpoint` 的输入 Tensor 必须 `requires_grad=True`，否则梯度链会断开。
* **BatchNorm**：Numpy 代码简化了，没处理 BN。在真实的 GPipe 中，BatchNorm 需要跨 Micro-batch 同步统计量，或者冻结统计量。Torch 实现中若加入 `nn.BatchNorm`，默认行为是基于当前 micro-batch 更新 running stats，这可能与大 Batch 行为不一致。



```

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`09.pdf`

![GPipe 图 1](/paper-figures/09/img-008.png)
*图 1：建议结合本节 `流水线并行训练` 一起阅读。*

![GPipe 图 2](/paper-figures/09/img-009.png)
*图 2：建议结合本节 `流水线并行训练` 一起阅读。*

![GPipe 图 3](/paper-figures/09/img-006.png)
*图 3：建议结合本节 `流水线并行训练` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**GPipe**（围绕 `流水线并行训练`）

### 一、选择题（10题）

1. 在 GPipe 中，最关键的建模目标是什么？
   - A. 流水线并行训练
   - B. micro-batch
   - C. pipeline
   - D. 重计算
   - **答案：A**

2. 下列哪一项最直接对应 GPipe 的核心机制？
   - A. micro-batch
   - B. pipeline
   - C. 重计算
   - D. 吞吐
   - **答案：B**

3. 在复现 GPipe 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 GPipe，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 GPipe 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. GPipe 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 GPipe 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 GPipe 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，GPipe 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，GPipe 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 GPipe 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 GPipe 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `micro-batch`。
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

8. 写一个小型单元测试，验证 `pipeline` 相关张量形状正确。
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

