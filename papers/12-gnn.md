# 论文解读：Neural Message Passing for Quantum Chemistry

## 1. 一句话概述

本文提出了一种名为 **MPNN (Message Passing Neural Networks)** 的通用框架，统一了当时存在的多种图神经网络模型，并设计了一种基于“边缘网络”与“Set2Set”读出的特定变体，在量子化学分子性质预测任务（QM9）上取得了当时的最先进（SOTA）结果 。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

### 2.1 核心问题

* 
**化学预测的局限性**：在药物发现和材料科学中，利用机器学习预测分子性质潜力巨大，但当时的研究多依赖于手工特征工程（Feature Engineering），缺乏能直接从分子图结构中学习且对分子对称性（如图同构）保持不变的有效深度学习模型 。


* 
**模型碎片化**：文献中已经出现了多种针对图数据的神经网络模型（如 Graph Convolution, Gated Graph Neural Networks, Interaction Networks 等），但缺乏一个统一的框架来理解它们之间的联系 。



### 2.2 主要贡献

1. 
**提出 MPNN 框架**：将现有的至少 8 种图神经网络模型抽象为一个通用的 **消息传递神经网络 (MPNN)** 框架。该框架包含两个阶段：消息传递阶段（Message Passing Phase）和读出阶段（Readout Phase） 。


2. 
**高性能变体设计**：在 MPNN 框架下探索了多种新的变体，发现结合 **Edge Network（边缘网络）** 消息函数和 **Set2Set** 读出函数的模型效果最佳 。


3. 
**QM9 上的 SOTA**：在包含 13 种化学性质的 QM9 数据集上，该模型在所有 13 个目标上均达到了最先进水平，并在 11 个目标上达到了“化学精度”（Chemical Accuracy），证明了该方法无需复杂手工特征即可有效学习分子表征 。



## 3. Introduction: 论文的动机是什么？

### 3.1 从计算机视觉到量子化学

论文开篇类比了计算机视觉的发展历程。卷积神经网络（CNN）之所以在图像领域大获成功，是因为其架构具备了适合图像数据的归纳偏置（Inductive Bias）——即平移不变性 。而在化学领域，虽然数据量随着高通量实验和量子模拟（如 DFT）的计算而激增，但缺乏能够利用原子系统对称性（如图同构不变性）的神经网络架构 。

### 3.2 现有方法的不足

* 
**传统 DFT**：密度泛函理论（DFT）计算精确但极其耗时（处理一个分子可能需要数小时），难以扩展到大规模筛选 。


* 
**手工特征 ML**：早期的机器学习方法（如 Coulomb Matrix）依赖预先计算的物理描述符，这些描述符往往要么不具备图同构不变性，要么难以处理不同大小的分子，且并未充分利用大规模数据 。



### 3.3 统一框架的必要性

当时已有若干针对图数据的神经网络尝试（如 Duvenaud et al., Li et al. 等），但它们被视为孤立的模型。作者认为，与其继续发明新的孤立模型，不如提炼出一个通用框架（MPNN），并在一个具有实际意义的高难度基准（QM9）上探索该框架的极限 。通过这种方式，作者希望找到一种能直接从分子图中学习特征、替代传统 DFT 模拟的高效模型。

## 4. Method: 解决方案是什么？

### 4.1 MPNN 通用框架

MPNN 的前向传播分为两个主要阶段：**消息传递阶段** 和 **读出阶段** 。

#### 1. 消息传递阶段 (Message Passing Phase)

该阶段运行 `T` 个时间步。在每个时间步 `t`，图中每个节点 `v` 的隐藏状态 `h_v^t` 根据邻居节点发来的消息进行更新。包含两个核心函数：

* **消息函数 (Message Function)**：`m_v^{t+1} = \sum_{u\in\mathcal{N}(v)} M_t(h_v^t, h_u^t, e_{vu})`
* **节点更新函数 (Vertex Update Function)**：`h_v^{t+1} = U_t(h_v^t, m_v^{t+1})`

公式如下 ：


其中，`\mathcal{N}(v)` 是节点 `v` 的邻居集合，`e_{vu}` 是边特征（如键类型、距离）。这一步使得信息能够在图中传播，每个节点逐渐获得局部和全局上下文信息。

#### 2. 读出阶段 (Readout Phase)

在 `T` 步消息传递后，利用读出函数 `R` 将所有节点最终状态聚合为图级特征向量 `\hat y`：


函数 `R` 必须对节点排列保持不变（invariant to permutations），以保证图同构不变性。

### 4.2 论文提出的最佳变体 (State of the Art)

作者在框架内进行了大量消融实验，最终确定的最佳配置如下：

1. **消息函数：Edge Network**
作者并未使用简单的矩阵乘法，而是设计了一个神经网络 `A(e_{vu})`，将边特征映射为一个矩阵，再与邻居节点状态相乘：





这里 `A` 是一个将边向量映射为 `d x d` 矩阵的神经网络。这种方法能很好地处理连续边特征（如原子间距离）。
2. **更新函数：GRU**
采用门控循环单元（GRU），将上一时刻节点状态 `h_v^t` 作为记忆，新消息 `m_v^{t+1}` 作为输入。


3. **读出函数：Set2Set**
相比于简单的求和（Sum Pooling），作者引入了 Vinyals 等人提出的 Set2Set 模型。它通过类似 LSTM 的注意力机制，对节点集合进行多次处理，生成一个对顺序不敏感的固定长度向量 。这对捕获长距离依赖至关重要。



### 4.3 输入表征

* 
**节点特征**：原子类型（H, C, N, O, F）、原子杂化方式等 。


* 
**边特征**：包含键类型（单键、双键等）以及原子间的欧几里得距离 。


* 
**虚拟元素**：为了加快信息传播，作者还尝试了引入“虚拟主节点”（Master Node）连接所有节点，或者使用 Set2Set 读出来聚合全局信息 。



```mermaid
graph LR
    subgraph Input
    G[分子图 (Atoms, Bonds, Distances)]
    end

    subgraph "MPNN Layer (Repeat T times)"
    Msg[Message Function: Edge Network]
    Agg[Aggregation: Sum over neighbors]
    Upd[Update Function: GRU]
    G --> Msg
    Msg --> Agg
    Agg --> Upd
    Upd --> Msg
    end

    subgraph "Readout Phase"
    S2S[Set2Set Pooling]
    MLP[Final Prediction MLP]
    end

    Upd --> S2S
    S2S --> MLP
    MLP --> Output[预测值 (e.g. Energy)]

    style Msg fill:#f9f,stroke:#333
    style S2S fill:#ff9,stroke:#333

```

## 5. Experiment: 实验与分析

### 5.1 实验设置

* 
**数据集**：QM9，包含约 134k 个有机小分子，每个分子有 13 个通过 DFT 计算得到的物理化学性质（如 HOMO, LUMO, 内能等） 。


* 
**评价指标**：平均绝对误差（MAE），并与“化学精度”（Chemical Accuracy）进行对比。若误差比率（Error Ratio）小于 1，则说明模型达到了化学精度 。


* 
**Baselines**：不仅对比了基于手工特征（如 Coulomb Matrix, BoB）的传统方法，还对比了之前的 GNN 模型（如 GG-NN, Graph Conv） 。



### 5.2 核心结果

1. 
**全面超越 SOTA**：作者提出的 MPNN 变体（enn-s2s）在所有 13 个目标上均取得了当时的最佳成绩（Table 2） 。


2. 
**化学精度达成**：在 13 个目标中有 11 个目标的预测误差低于化学精度，意味着神经网络可以替代昂贵的 DFT 计算用于大规模筛选 。


3. **推理速度**：相比传统 DFT 计算（需 10^3 秒），MPNN 推理仅需 10^-2 秒，速度提升了约 300,000 倍。

### 5.3 消融实验与分析

* **空间信息的重要性**：如果不使用原子间的距离信息（仅使用拓扑图），模型性能会显著下降。但引入 Set2Set 或 Master Node 可以部分弥补这一损失，说明全局信息聚合对长距离相互作用建模很重要。
* 
**联合训练 vs 单独训练**：意外的是，针对每个目标单独训练模型通常比联合训练所有 13 个目标效果更好，部分目标提升幅度达 40% 。


* 
**显式氢原子**：将氢原子作为图中的独立节点（而不是仅仅作为重原子的特征）对于提高精度至关重要 。



## 6. Numpy 与 Torch 对照实现

### 6.1 代码对应关系与说明

下方的代码实现主要对应论文 **Section 2 (公式 1, 2) 和 Section 5.1** 的通用 MPNN 逻辑。

* **对应部分**：
* `message` 函数对应论文中的通用消息形式 `M_t(h_v^t,h_u^t,e_{vu})`。注意：提供的 Numpy 代码实现的是 `tanh` 激活 MLP，输入是 `[h_src, h_tgt, e_feat]` 的拼接。这接近于论文 5.1 节提到的 "Pair Message" 变体（Battaglia et al., 2016），虽然论文最终推荐的是 "Edge Network"，但 Numpy 代码展示的是更通用的拼接式消息传递。
* `aggregate` 对应公式 (1) 中的求和符号 `\sum_{u\in\mathcal{N}(v)}`。
* `update` 对应公式 (2) 中的 `U_t(\cdot)`。


* **张量形状假设**：
* `node_dim`：节点特征维度。
* `edge_dim`：边特征维度。
* `hidden_dim`：隐藏层维度。
* Numpy 代码是逐节点循环（非 batch 化），Torch 实现将基于 `edge_index` 进行全图向量化操作，假设处理单个图或已 batch 化的 `BlockDiagonal` 大图。



::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)
      
# Graph Representation

class Graph:
    """Simple graph representation"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edges = []  # List of (source, target) tuples
        self.node_features = []  # List of node feature vectors
        self.edge_features = {}  # Dict: (src, tgt) -> edge features
     
    def add_edge(self, src, tgt, features=None):
        self.edges.append((src, tgt))
        if features is not None:
            self.edge_features[(src, tgt)] = features
     
    def set_node_features(self, features):
        """features: list of feature vectors"""
        self.node_features = features
     
    def get_neighbors(self, node):
        """Get all neighbors of a node"""
        neighbors = []
        for src, tgt in self.edges:
            if src == node:
                neighbors.append(tgt)
        return neighbors
     
    def visualize(self, node_labels=None):
        """Visualize graph using networkx"""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))
        G.add_edges_from(self.edges)
        
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=800, font_size=12, arrows=True,
                arrowsize=20, edge_color='gray', width=2)
        
        if node_labels:
            nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
        
        plt.title("Graph Structure")
        plt.axis('off')
        plt.show()

# Create sample molecular graph
# H2O (water): O connected to 2 H atoms
water = Graph(num_nodes=3)
water.add_edge(0, 1)  # O -> H
water.add_edge(0, 2)  # O -> H  
water.add_edge(1, 0)  # H -> O (undirected)
water.add_edge(2, 0)  # H -> O

# Node features: [atomic_num, valence, ...]
water.set_node_features([
    np.array([8, 2]),  # Oxygen
    np.array([1, 1]),  # Hydrogen
    np.array([1, 1]),  # Hydrogen
])

labels = {0: 'O', 1: 'H', 2: 'H'}
water.visualize(labels)

print(f"Number of nodes: {water.num_nodes}")
print(f"Number of edges: {len(water.edges)}")
print(f"Neighbors of node 0 (Oxygen): {water.get_neighbors(0)}")
      
# Message Passing Framework
# Two phases:
# 1. Message Passing: Aggregate information from neighbors (T steps)
# 2. Readout: Global graph representation

class MessagePassingLayer:
    """Single message passing layer"""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message function: M(h_v, h_w, e_vw)
        self.W_msg = np.random.randn(hidden_dim, 2*node_dim + edge_dim) * 0.01
        self.b_msg = np.zeros(hidden_dim)
        
        # Update function: U(h_v, m_v)
        self.W_update = np.random.randn(node_dim, node_dim + hidden_dim) * 0.01
        self.b_update = np.zeros(node_dim)
     
    def message(self, h_source, h_target, e_features):
        """Compute message from source to target"""
        # Concatenate source, target, edge features
        if e_features is None:
            e_features = np.zeros(self.edge_dim)
        
        concat = np.concatenate([h_source, h_target, e_features])
        
        # Apply message network
        message = np.tanh(np.dot(self.W_msg, concat) + self.b_msg)
        return message
     
    def aggregate(self, messages):
        """Aggregate messages (sum)"""
        if len(messages) == 0:
            return np.zeros(self.hidden_dim)
        return np.sum(messages, axis=0)
     
    def update(self, h_node, aggregated_message):
        """Update node representation"""
        concat = np.concatenate([h_node, aggregated_message])
        h_new = np.tanh(np.dot(self.W_update, concat) + self.b_update)
        return h_new
     
    def forward(self, graph, node_states):
        """
        One message passing step
        
        graph: Graph object
        node_states: list of current node hidden states
        
        Returns: updated node states
        """
        new_states = []
        
        for v in range(graph.num_nodes):
            # Collect messages from neighbors
            messages = []
            for w in graph.get_neighbors(v):
                # Get edge features
                edge_feat = graph.edge_features.get((w, v), None)
                
                # Compute message
                msg = self.message(node_states[w], node_states[v], edge_feat)
                messages.append(msg)
            
            # Aggregate messages
            aggregated = self.aggregate(messages)
            
            # Update node state
            h_new = self.update(node_states[v], aggregated)
            new_states.append(h_new)
        
        return new_states

# Test message passing
node_dim = 4
edge_dim = 2
hidden_dim = 8

mp_layer = MessagePassingLayer(node_dim, edge_dim, hidden_dim)

# Initialize node states from features
initial_states = []
for feat in water.node_features:
    # Embed to higher dimension
    state = np.concatenate([feat, np.zeros(node_dim - len(feat))])
    initial_states.append(state)

# Run message passing
updated_states = mp_layer.forward(water, initial_states)

print(f"\nInitial state (O): {initial_states[0]}")
print(f"Updated state (O): {updated_states[0]}")
print(f"\nNode states updated via neighbor information!")
      
# Complete MPNN

class MPNN:
    """Message Passing Neural Network"""
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, output_dim):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embed_W = np.random.randn(hidden_dim, node_feat_dim) * 0.01
        
        # Message passing layers
        self.mp_layers = [
            MessagePassingLayer(hidden_dim, edge_feat_dim, hidden_dim*2)
            for _ in range(num_layers)
        ]
        
        # Readout (graph-level prediction)
        self.readout_W = np.random.randn(output_dim, hidden_dim) * 0.01
        self.readout_b = np.zeros(output_dim)
     
    def forward(self, graph):
        """
        Forward pass through MPNN
        
        Returns: graph-level prediction
        """
        # Embed node features
        node_states = []
        for feat in graph.node_features:
            embedded = np.tanh(np.dot(self.embed_W, feat))
            node_states.append(embedded)
        
        # Message passing
        states_history = [node_states]
        for layer in self.mp_layers:
            node_states = layer.forward(graph, node_states)
            states_history.append(node_states)
        
        # Readout: aggregate node states to graph representation
        graph_repr = np.sum(node_states, axis=0)  # Simple sum pooling
        
        # Final prediction
        output = np.dot(self.readout_W, graph_repr) + self.readout_b
        
        return output, states_history

# Create MPNN
mpnn = MPNN(
    node_feat_dim=2,
    edge_feat_dim=2,
    hidden_dim=8,
    num_layers=3,
    output_dim=1  # Predict single property (e.g., energy)
)

# Forward pass
prediction, history = mpnn.forward(water)

print(f"Graph-level prediction: {prediction}")
print(f"(E.g., molecular property like energy, solubility, etc.)")
      
# Visualize Message Passing

# Visualize how node representations evolve
fig, axes = plt.subplots(1, len(history), figsize=(16, 4))

for step, states in enumerate(history):
    # Stack node states for visualization
    states_matrix = np.array(states).T  # (hidden_dim, num_nodes)
    
    ax = axes[step]
    im = ax.imshow(states_matrix, cmap='RdBu', aspect='auto')
    ax.set_title(f'Step {step}')
    ax.set_xlabel('Node')
    ax.set_ylabel('Hidden Dimension')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['O', 'H', 'H'])

plt.colorbar(im, ax=axes, label='Activation')
plt.suptitle('Node Representations Through Message Passing', fontsize=14)
plt.tight_layout()
plt.show()

print("\nNodes update their representations by aggregating neighbor information")
      
# Create More Complex Graph

# Create benzene ring (C6H6)
benzene = Graph(num_nodes=12)  # 6 C + 6 H

# Carbon ring (nodes 0-5)
for i in range(6):
    next_i = (i + 1) % 6
    benzene.add_edge(i, next_i)
    benzene.add_edge(next_i, i)

# Hydrogen atoms (nodes 6-11) attached to carbons
for i in range(6):
    h_idx = 6 + i
    benzene.add_edge(i, h_idx)
    benzene.add_edge(h_idx, i)

# Node features
features = []
for i in range(6):
    features.append(np.array([6, 3]))  # Carbon
for i in range(6):
    features.append(np.array([1, 1]))  # Hydrogen
benzene.set_node_features(features)

# Visualize
labels = {i: 'C' for i in range(6)}
labels.update({i: 'H' for i in range(6, 12)})
benzene.visualize(labels)

# Run MPNN
pred_benzene, hist_benzene = mpnn.forward(benzene)
print(f"\nBenzene prediction: {pred_benzene}")
      
# Different Aggregation Functions

# Compare aggregation strategies
def sum_aggregation(messages):
    return np.sum(messages, axis=0) if len(messages) > 0 else np.zeros_like(messages[0])

def mean_aggregation(messages):
    return np.mean(messages, axis=0) if len(messages) > 0 else np.zeros_like(messages[0])

def max_aggregation(messages):
    return np.max(messages, axis=0) if len(messages) > 0 else np.zeros_like(messages[0])

# Test on random messages
test_messages = [np.random.randn(8) for _ in range(3)]

print("Aggregation Functions:")
print(f"Sum: {sum_aggregation(test_messages)[:4]}...")
print(f"Mean: {mean_aggregation(test_messages)[:4]}...")
print(f"Max: {max_aggregation(test_messages)[:4]}...")
print("\nDifferent aggregations capture different patterns!")
      
# Key Takeaways
# Message Passing Framework:
# Phase 1: Message Passing (repeat T times)
# For each node v:
#   1. Collect messages from neighbors:
#      m_v = Σ_{u∈N(v)} M_t(h_v, h_u, e_uv)
#   
#   2. Update node state:
#      h_v = U_t(h_v, m_v)
# Phase 2: Readout
# Graph representation:
#   h_G = R({h_v | v ∈ G})
# Components:
# 1. Message function M: Compute message from neighbor
# 2. Aggregation: Combine messages (sum, mean, max, attention)
# 3. Update function U: Update node representation
# 4. Readout R: Graph-level pooling
# Variants:
# • GCN: Simplified message passing with normalization
# • GraphSAGE: Sampling neighbors, inductive learning
# • GAT: Attention-based aggregation
# • GIN: Powerful aggregation (sum + MLP)
# Applications:
# • Molecular property prediction: QM9, drug discovery
# • Social networks: Node classification, link prediction
# • Knowledge graphs: Reasoning, completion
# • Recommendation: User-item graphs
# • 3D vision: Point clouds, meshes
# Advantages:
# • ✅ Handles variable-size graphs
# • ✅ Permutation invariant
# • ✅ Inductive learning (generalize to new graphs)
# • ✅ Interpretable (message passing)
# Challenges:
# • Over-smoothing (deep layers make nodes similar)
# • Expressiveness (limited by aggregation)
# • Scalability (large graphs)
# Modern Extensions:
# • Graph Transformers: Attention on full graph
# • Equivariant GNNs: Respect symmetries (E(3), SE(3))
# • Temporal GNNs: Dynamic graphs
# • Heterogeneous GNNs: Multiple node/edge types

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

# 高效 Torch 实现：使用向量化（Vectorization）代替 Numpy 中的 Python 循环
# 假设输入数据已转换为张量格式（如 PyTorch Geometric 标准）

class VectorizedMPLayer(nn.Module):
    """
    对应 Numpy 的 MessagePassingLayer
    但在 Torch 中一次性处理所有边和节点
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message function: M(h_v, h_w, e_vw) -> Tanh(W_msg @ cat[...] + b)
        # 输入维度: src_node (hidden_dim) + tgt_node (hidden_dim) + edge (edge_dim)
        # 注意: Numpy 代码中用的是 hidden_dim 作为 node_state 的维度（在 embed 后）
        # 这里统一用 hidden_dim (假设输入已经是 hidden state)
        self.msg_mlp = nn.Linear(2 * hidden_dim + edge_dim, hidden_dim)
        
        # Update function: U(h_v, m_v) -> Tanh(W_upd @ cat[...] + b)
        # 输入维度: node (hidden_dim) + aggregated_msg (hidden_dim)
        self.update_mlp = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, hidden_dim] 节点特征矩阵
        edge_index: [2, num_edges] 边索引 (src, tgt)
        edge_attr: [num_edges, edge_dim] 边特征
        """
        src, tgt = edge_index[0], edge_index[1]
        
        # 1. 向量化计算所有消息 (对应 Numpy 中的 message 函数)
        # Gather source and target node states
        h_src = x[src]  # [E, hidden_dim]
        h_tgt = x[tgt]  # [E, hidden_dim]
        
        # Concatenate: [h_src, h_tgt, e_features]
        # 对应 Numpy: concat = np.concatenate([h_source, h_target, e_features])
        raw_msg_input = torch.cat([h_src, h_tgt, edge_attr], dim=1)
        
        # Apply Message Network
        # 对应 Numpy: message = np.tanh(np.dot(self.W_msg, concat) + self.b_msg)
        messages = torch.tanh(self.msg_mlp(raw_msg_input)) # [E, hidden_dim]
        
        # 2. 聚合消息 (对应 Numpy 中的 aggregate 函数)
        # 使用 scatter_add 将消息聚合到目标节点 tgt
        # 对应 Numpy: np.sum(messages, axis=0)
        aggr_out = torch.zeros_like(x)
        # index_add_: 将 messages 加到 aggr_out 的 tgt 索引位置
        aggr_out.index_add_(0, tgt, messages) 
        
        # 3. 更新节点状态 (对应 Numpy 中的 update 函数)
        # 对应 Numpy: concat = np.concatenate([h_node, aggregated_message])
        update_input = torch.cat([x, aggr_out], dim=1)
        
        # 对应 Numpy: h_new = np.tanh(np.dot(self.W_update, concat) + self.b_update)
        h_new = torch.tanh(self.update_mlp(update_input))
        
        return h_new

class VectorizedMPNN(nn.Module):
    """
    对应 Numpy 的 MPNN 类
    """
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        # Embedding: 对应 Numpy self.embed_W
        self.embedding = nn.Linear(node_feat_dim, hidden_dim)
        
        # MP Layers
        self.layers = nn.ModuleList([
            VectorizedMPLayer(hidden_dim, edge_feat_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Readout: 对应 Numpy self.readout_W
        self.readout = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # 1. Embed node features
        # 对应 Numpy: embedded = np.tanh(np.dot(self.embed_W, feat))
        # 注意: Numpy 代码用了 tanh 作为 embedding 激活，这里保持一致
        h = torch.tanh(self.embedding(x))
        
        states_history = [h]
        
        # 2. Message Passing Phase
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            states_history.append(h)
            
        # 3. Readout Phase
        # 对应 Numpy: graph_repr = np.sum(node_states, axis=0)
        graph_repr = torch.sum(h, dim=0) # [hidden_dim]
        
        # 对应 Numpy: output = np.dot(self.readout_W, graph_repr) + self.readout_b
        output = self.readout(graph_repr)
        
        return output, states_history

# --- 使用示例 (模拟 Numpy 的数据结构) ---
if __name__ == "__main__":
    # 模拟 Water 分子数据
    # Node features: [3, 2] (3 nodes, 2 features)
    x = torch.tensor([[8., 2.], [1., 1.], [1., 1.]], dtype=torch.float32)
    
    # Edge Index: [2, E] (O->H, O->H, H->O, H->O)
    edge_index = torch.tensor([
        [0, 0, 1, 2], # src
        [1, 2, 0, 0]  # tgt
    ], dtype=torch.long)
    
    # Edge Features: [E, 2] (Dummy features)
    edge_attr = torch.zeros((4, 2), dtype=torch.float32)
    
    # Model init
    model = VectorizedMPNN(node_feat_dim=2, edge_feat_dim=2, hidden_dim=8, num_layers=3, output_dim=1)
    
    # Forward
    with torch.no_grad(): # 推理模式
        pred, hist = model(x, edge_index, edge_attr)
    
    print(f"Torch Prediction: {pred.item():.4f}")
    print(f"History Steps: {len(hist)}")

```

:::

### 6.2 对照讲解：为什么 Torch 这样写？

1. **从循环到向量化 (For-loops vs. Scatter)**
* **Numpy**: 使用了显式的 Python `for` 循环遍历节点 `v`，再遍历邻居 `w`。这种方式在图很大时极其低效。
* **Torch**: 使用了“边列表” (`edge_index`, shape `[2, E]`) 的思路。通过 `x[src]` 和 `x[tgt]` 一次性把所有边的起点和终点特征“索引”出来，组成一个 `[E, ...]` 的大矩阵。这样整个消息计算变成了一次巨大的矩阵乘法（`nn.Linear`），利用了 GPU 的并行能力。


2. **聚合操作 (Aggregation)**
* **Numpy**: `np.sum(messages)` 只能对当前节点的邻居列表求和。
* **Torch**: 使用 `index_add_` (或 `scatter_add`)。这是一个专为稀疏数据设计的操作，它根据 `tgt` 索引，将所有发往同一个目标节点的消息自动加在一起。这是 GNN 实现中最核心的算子。


3. **形状与维度 (Shape Semantics)**
* Numpy 代码中，节点状态 `h` 是列表 `List[np.array]`。
* Torch 代码中，节点状态 `h` 始终保持为 `[num_nodes, hidden_dim]` 的二维张量。


4. **数值稳定性与初始化**
* Numpy 代码使用了 `np.tanh` 作为激活函数，这也是 MPNN 论文早期常用的激活函数（尽管现在 ReLU 更流行）。Torch 版为了“完全等价复现”，也保留了 `torch.tanh`。
* Numpy 手动初始化权重 `* 0.01`。Torch 的 `nn.Linear` 默认使用 Kaiming/Xavier 初始化，若要完全对齐数值，需手动重置 Torch 参数，但在生产代码中通常使用 Torch 默认值即可。

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`12.pdf`

![MPNN / GNN 图 1](/paper-figures/12/page-1.png)
*图 1：建议结合本节 `图消息传递` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**MPNN / GNN**（围绕 `图消息传递`）

### 一、选择题（10题）

1. 在 MPNN / GNN 中，最关键的建模目标是什么？
   - A. 图消息传递
   - B. 消息函数
   - C. 聚合
   - D. 读出
   - **答案：A**

2. 下列哪一项最直接对应 MPNN / GNN 的核心机制？
   - A. 消息函数
   - B. 聚合
   - C. 读出
   - D. 节点特征
   - **答案：B**

3. 在复现 MPNN / GNN 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 MPNN / GNN，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 MPNN / GNN 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. MPNN / GNN 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 MPNN / GNN 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 MPNN / GNN 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，MPNN / GNN 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，MPNN / GNN 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 MPNN / GNN 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 MPNN / GNN 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `消息函数`。
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

8. 写一个小型单元测试，验证 `聚合` 相关张量形状正确。
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

