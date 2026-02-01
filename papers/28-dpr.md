# 论文解读与复现：Dense Passage Retrieval (DPR)

## 1. 一句话概述
DPR 是一篇开放域问答（Open-Domain QA）的里程碑式论文，它证明了无需复杂的预训练任务（如 ICT），仅通过简单的**双塔模型（Dual-Encoder）**配合**批次内负采样（In-batch Negatives）**训练策略，即可在检索性能上大幅超越传统的稀疏检索方法（如 BM25）。

---

## 2. Abstract: 论文试图解决什么问题？

[cite_start]在开放域问答任务中，系统需要从海量文档（如 Wikipedia）中检索出包含答案的段落。长期以来，基于关键词匹配的稀疏向量模型（如 TF-IDF 或 BM25）是事实上的标准方法 [cite: 7]。尽管深度学习在阅读理解（Reader）阶段取得了巨大进展，但检索（Retriever）阶段依然依赖传统算法。

[cite_start]**论文试图解决的核心矛盾是**：传统稀疏检索无法捕捉语义信息（例如同义词匹配），而之前的稠密检索（Dense Retrieval）方法通常被认为需要大量的标注数据或极复杂的预训练任务（如 Inverse Cloze Task, ICT）才能生效 [cite: 26]。

**本文的主要贡献**：
1.  [cite_start]提出了一种简单的稠密检索框架（DPR），利用 BERT 双塔结构学习问题和段落的嵌入表示 [cite: 8]。
2.  [cite_start]证明了只要训练策略得当（特别是利用 **In-batch Negatives**），仅使用有限的问题-段落对（Question-Passage pairs）就能训练出高质量的检索器 [cite: 35]。
3.  [cite_start]在多个开放域 QA 数据集（Natural Questions, TriviaQA 等）上，DPR 的 Top-20 检索准确率比强大的 Lucene-BM25 系统高出 **9%-19%** [cite: 9]。
4.  [cite_start]将 DPR 与阅读器结合，在多个 benchmark 上刷新了 End-to-End QA 的 State-of-the-Art (SOTA) [cite: 9]。

---

## 3. Introduction: 动机与背景梳理

### 3.1 开放域 QA 的两阶段范式
[cite_start]现代开放域 QA 系统通常遵循 **Retriever-Reader** 的两阶段流程 [cite: 12]：
1.  **Retriever**：从数百万个文档中筛选出几十个可能包含答案的候选段落（Subset）。
2.  **Reader**：对这几十个段落进行精细阅读，抽取具体答案。

[cite_start]虽然将问题简化为阅读理解是合理的，但如果在检索阶段就漏掉了包含答案的段落，后续的 Reader 再强大也无济于事 [cite: 13][cite_start]。例如，在 SQuAD v1.1 上，如果检索不准确，Exact Match 分数会从 >80% 掉到 <40% [cite: 17]。

### 3.2 稀疏检索 vs. 稠密检索
* **稀疏检索 (BM25)**：基于倒排索引和关键词匹配。它的优势是效率高，但无法理解语义。例如，问题 "Who is the bad guy in lord of the rings?" [cite_start]对应答案包含 "villain Sauron"。BM25 很难匹配 "bad guy" 和 "villain"，而稠密向量可以通过语义接近度解决这个问题 [cite: 21, 22]。
* [cite_start]**稠密检索 (Dense)**：将问题和段落映射到低维连续空间。此前的方法（如 ORQA）认为单纯的有监督训练不够，必须引入复杂的 ICT 预训练 [cite: 26]。

### 3.3 DPR 的破局思路
[cite_start]作者提出一个核心反问：**能否仅使用问题和段落对，不进行额外的预训练，就训练出一个更好的稠密检索模型？** [cite: 31]
[cite_start]答案是肯定的。通过利用预训练的 BERT 和一种特定的训练机制——**最大化问题与相关段落点积的 Batch 训练**，DPR 证明了额外的复杂预训练可能是不必要的 [cite: 36][cite_start]。实验表明，在 Natural Questions 数据集上，DPR 的 Top-5 准确率达到 65.2%，而 BM25 仅为 42.9% [cite: 34]。

---

## 4. Method: 解决方案与核心算法

DPR 的核心是一个基于 Transformer 的双塔模型（Dual-Encoder），配合高效的负采样训练策略。

### 4.1 模型架构 (Dual-Encoder)
[cite_start]DPR 使用两个独立的 BERT 网络（base, uncased）分别作为**问题编码器** $E_Q(\cdot)$ 和**段落编码器** $E_P(\cdot)$ [cite: 55, 69]。
* 输入：文本序列。
* [cite_start]输出：使用 `[CLS]` token 的输出作为该文本的 $d$ 维稠密向量表示（$d=768$）[cite: 69]。

相似度计算采用简单的**点积（Dot Product）**：
$$sim(q, p) = E_Q(q)^T E_P(p)$$
[cite_start][cite: 62]

> [cite_start]**解释**：虽然也可以使用余弦相似度或 L2 距离，但作者通过消融实验发现，简单的点积配合可学习的编码器效果最好且计算最高效（便于利用现有的 MIPS 索引工具如 FAISS）[cite: 68]。

### 4.2 训练目标 (Metric Learning)
[cite_start]训练的核心是一个 Metric Learning 问题：让相关（Positive）的问题-段落对在向量空间中距离更近，不相关（Negative）的更远 [cite: 74]。

假设训练集包含 $m$ 个样本，每个样本由 1 个问题 $q_i$、1 个正样本段落 $p_i^+$ 和 $n$ 个负样本段落 $p_{i,j}^-$ 组成。优化目标是最小化正样本的负对数似然（NLL）：

$$L(q_i, p_i^+, p_{i,1}^-, \dots, p_{i,n}^-) = -\log \frac{e^{sim(q_i, p_i^+)}}{e^{sim(q_i, p_i^+)} + \sum_{j=1}^n e^{sim(q_i, p_{i,j}^-)}}$$
[cite_start][cite: 80]

> **解释**：这本质上是一个 Softmax 交叉熵损失。对于每个问题 $q_i$，模型需要从 $1+n$ 个候选项中“分类”出正确的那个正样本段落。

### 4.3 负采样策略 (In-batch Negatives)
[cite_start]这是 DPR 成功的关键。在检索任务中，正样本是明确的，但负样本池（全量文档库）极其庞大。如何选择负样本至关重要 [cite: 84]。

[cite_start]DPR 采用了 **In-batch Negatives** 策略 [cite: 95]：
假设一个 Batch 有 $B$ 个问题，每个问题对应一个正样本段落。于是我们有 $B$ 个 Question 向量和 $B$ 个 Passage 向量。
* 对于问题 $q_i$，与其配对的 $p_i$ 是正样本。
* **同一个 Batch 内的其他 $B-1$ 个段落 $p_j (j \neq i)$ 自动被视为负样本。**

**优势**：
1.  [cite_start]**计算效率极高**：不需要重新编码负样本。计算一个 $B \times B$ 的相似度矩阵，就可以同时为 $B$ 个问题提供训练信号 [cite: 97]。
2.  **样本数量**：有效训练对数量变成了 $B^2$（包含负对）。
3.  [cite_start]**Gold Negatives 补充**：除了 In-batch 负样本，作者还发现为每个问题额外增加一个 **BM25 检索出的“困难负样本”（Hard Negative）**（即包含关键词但不包含答案的段落）能进一步提升效果 [cite: 92]。

```mermaid
graph TD
    subgraph Training [训练阶段: In-Batch Negatives]
    Q[问题 Batch (B个)] --> EQ[BERT Question Encoder]
    P[正样本段落 Batch (B个)] --> EP[BERT Passage Encoder]
    EQ --> QV[问题向量 Q (B x d)]
    EP --> PV[段落向量 P (B x d)]
    QV --> Dot{点积相似度矩阵 S = Q * P^T}
    PV --> Dot
    Dot --> Loss[Cross Entropy Loss]
    note[对于第i个问题, S[i,i]是正例<br>S[i,j]是负例] -.-> Loss
    end

    subgraph Inference [推理阶段]
    Doc[全量 Wikipedia (21M段落)] --> EP_Inf[BERT Passage Encoder]
    EP_Inf --> Index[FAISS 向量索引]
    Query[新问题] --> EQ_Inf[BERT Question Encoder]
    EQ_Inf --> Q_Vec
    Q_Vec --> Search{MIPS 检索}
    Index --> Search
    Search --> TopK[Top-K 候选段落]
    end

```

---

## 5. Experiment: 实验设置与结果分析

### 5.1 实验设置

* 
**数据来源**：英文 Wikipedia（2018年12月 dump），切分为 100 个词的段落，共 2100 万个段落 。


* **数据集**：Natural Questions (NQ), TriviaQA, WebQuestions (WQ), CuratedTREC (TREC), SQuAD v1.1。
* **评价指标**：Top-k Retrieval Accuracy（前 k 个检索结果中是否包含答案 span）。

### 5.2 主实验结果 (Retrieval Performance)

对比结果显示在 Table 2 中 ：

* **DPR vs BM25**：DPR 在除 SQuAD 以外的所有数据集上完胜。
* 
**Natural Questions**: Top-20 准确率从 BM25 的 **59.1%** 提升至 DPR 的 **78.4%**（提升近 20%！）。


* **TriviaQA**: 66.9% -> 79.4%。


* 
**SQuAD 的特例**：在 SQuAD 上 DPR 表现不如 BM25。作者分析是因为 SQuAD 的问题是标注者看着段落写出来的，词汇重合度极高，天然利好 BM25 。



### 5.3 消融实验：负采样策略的影响

作者在 Table 3 中详细分析了不同负采样对 NQ 数据集 Top-20 准确率的影响 ：

* **1-of-N (Random)**：随机负采样，准确率 64.3%。
* **1-of-N (BM25)**：仅使用 BM25 困难负采样，准确率 63.3%。
* **In-batch (Gold)**：仅使用 Batch 内其他正样本作负样本，准确率 **69.1%**。
* 
**In-batch + 1 BM25**：Batch 内负样本 + 1 个 BM25 困难负样本，准确率提升至 **77.3%** 。



**结论**：In-batch 策略提供了大量的负样本，而 BM25 提供了具有挑战性的负样本，二者结合效果最好。

### 5.4 样本效率 (Sample Efficiency)

DPR 是否需要海量数据？Figure 1 显示，仅使用 **1000 个** 训练样本，DPR 的性能就已经超过了 BM25 。这打破了“稠密检索需要海量数据预训练”的固有印象。

---

## 6. Numpy 与 Torch 对照实现

### 代码对应说明

提供的 Numpy 代码实现了一个**简化的双塔检索模型训练与推理流程**，对应论文的 **Section 3 (Dense Passage Retriever)** 和 **Eq. (2)**。

* **Encoder**: 代码中使用了一个基于 RNN 的 `SimpleTextEncoder` 来模拟 BERT。虽然结构不同，但输入输出接口（Token IDs -> Dense Vector）与论文一致。
* **Shape**:
* `q_emb`: `(Batch_Size, Hidden_Dim)`
* `p_emb`: `(Batch_Size, Hidden_Dim)`


* **In-batch Negatives**: 对应 Numpy 代码中的 `contrastive_loss` 函数及批量训练循环。它手动构建了正负样本的分数列表。
* **假设**: Numpy 代码中的 RNN 权重初始化非常小且随机，实际上无法收敛到有意义的语义空间，仅作为逻辑演示。Torch 实现将对其进行张量化重构，但保持逻辑等价。

### 代码对照 (Code Group)

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

np.random.seed(42)
     
# Dual Encoder Architecture
# Question → Encoder_Q → q (dense vector)
# Passage  → Encoder_P → p (dense vector)

# Similarity: sim(q, p) = q · p  (dot product)
class SimpleTextEncoder:
    """Simplified text encoder (in practice: use BERT)"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Simple RNN weights
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))
        
        # Output projection
        self.W_out = np.random.randn(hidden_dim, hidden_dim) * 0.01
    
    def encode(self, token_ids):
        """
        Encode sequence of token IDs to dense vector
        Returns: dense embedding (hidden_dim,)
        """
        h = np.zeros((self.hidden_dim, 1))
        
        # Process tokens
        for token_id in token_ids:
            # Lookup embedding
            x = self.embeddings[token_id].reshape(-1, 1)
            
            # RNN step
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
        
        # Final representation (CLS-like)
        output = np.dot(self.W_out, h).flatten()
        
        # L2 normalize for cosine similarity
        output = output / (np.linalg.norm(output) + 1e-8)
        
        return output

# ... (Tokenizer code omitted in prompt but implied for context) ...
# (Assuming synthetic data setup logic exists to feed the training loop)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

def contrastive_loss(query_emb, positive_emb, negative_embs):
    """
    Contrastive loss (InfoNCE)
    
    L = -log( exp(q·p+) / (exp(q·p+) + Σ exp(q·p-)) )
    """
    # Positive score
    pos_score = np.dot(query_emb, positive_emb)
    
    # Negative scores
    neg_scores = [np.dot(query_emb, neg_emb) for neg_emb in negative_embs]
    
    # All scores
    all_scores = np.array([pos_score] + neg_scores)
    
    # Softmax
    probs = softmax(all_scores)
    
    # Negative log likelihood (positive should be first)
    loss = -np.log(probs[0] + 1e-8)
    
    return loss

# Simulate training batch
# batch_size = 3
# batch_questions = question_embeddings[:batch_size]
# batch_passages = passage_embeddings[:batch_size]
# In-batch negatives: for each question, other passages in batch are negatives
# ... (Loop implementation shown in original numpy block) ...

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子以保证对照性
torch.manual_seed(42)

class SimpleTextEncoderTorch(nn.Module):
    """
    PyTorch 高效等价实现
    对应 Numpy: SimpleTextEncoder 类
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # 对应 Numpy: self.embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 对应 Numpy: RNN weights (W_xh, W_hh, b_h)
        # 使用原生 RNN 替代手动循环以实现 GPU 加速
        # batch_first=True 让输入变为 (Batch, Seq, Feature)
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_dim, 
                          batch_first=True)
        
        # 对应 Numpy: self.W_out
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 初始化权重以近似 Numpy 的 scale (仅为演示，实际训练不需要手动乘 0.01)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.rnn.weight_ih_l0, std=0.01)
        nn.init.normal_(self.rnn.weight_hh_l0, std=0.01)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def forward(self, token_ids):
        """
        Input: token_ids (Batch_Size, Seq_Len)
        Output: dense embedding (Batch_Size, Hidden_Dim)
        """
        # 1. Lookup embedding
        # x shape: (Batch, Seq, Emb_Dim)
        x = self.embedding(token_ids)
        
        # 2. RNN Step
        # 对应 Numpy 中的 for loop + tanh
        # output: (Batch, Seq, Hidden), h_n: (1, Batch, Hidden)
        _, h_n = self.rnn(x)
        
        # 取最后一个时间步的 hidden state，移除 layer 维度
        # h_n shape: (Batch, Hidden)
        h = h_n.squeeze(0)
        
        # 3. Final projection
        # 对应 Numpy: np.dot(self.W_out, h)
        output = self.out_proj(h)
        
        # 4. L2 Normalize
        # 对应 Numpy: output / (np.linalg.norm(output) + 1e-8)
        # p=2 表示 L2 范数, dim=1 表示沿特征维度
        output = F.normalize(output, p=2, dim=1)
        
        return output

def in_batch_contrastive_loss_torch(q_emb, p_emb):
    """
    PyTorch 向量化实现的 In-batch Contrastive Loss
    对应 Numpy: 循环调用 contrastive_loss
    
    Args:
        q_emb: (Batch_Size, Hidden_Dim)
        p_emb: (Batch_Size, Hidden_Dim) - 假设 p_emb[i] 是 q_emb[i] 的正样本
    """
    # 1. 计算相似度矩阵
    # 对应 Numpy: 每一个 q 和每一个 p 做 dot product
    # sim_matrix[i][j] = q_i · p_j
    sim_matrix = torch.matmul(q_emb, p_emb.T)  # Shape: (Batch, Batch)
    
    # 2. 构建标签
    # 在 In-batch 设置中，对角线元素 (i, i) 是正样本，同一行其他元素是负样本
    batch_size = q_emb.size(0)
    targets = torch.arange(batch_size, device=q_emb.device) # [0, 1, 2, ...]
    
    # 3. 计算 Cross Entropy Loss
    # 这相当于 Numpy 中的: softmax + negative log likelihood
    # PyTorch 的 CrossEntropyLoss 内部包含了 LogSoftmax，数值更稳定
    loss = F.cross_entropy(sim_matrix, targets)
    
    return loss

# --- 使用示例 ---
# 假设参数
V, D, H = 1000, 64, 128
B, Seq = 3, 10

encoder = SimpleTextEncoderTorch(V, D, H)

# 模拟输入 (Batch, Seq)
dummy_q = torch.randint(0, V, (B, Seq))
dummy_p = torch.randint(0, V, (B, Seq))

# 前向传播
q_vectors = encoder(dummy_q)
p_vectors = encoder(dummy_p)

# 计算 Loss
loss = in_batch_contrastive_loss_torch(q_vectors, p_vectors)

print(f"Batch Loss: {loss.item():.4f}")
# 这里的 loss 计算一次即可替代 Numpy 中的 for i in range(batch_size) 循环

```

:::

### 对照讲解与差异分析

1. **向量化操作 (Vectorization)**：
* **Numpy**: 使用了 Python `for` 循环遍历 Batch 中的每个问题，单独计算 `contrastive_loss`。在计算单个 Loss 时，又显式拼接了 `pos_score` 和 `neg_scores`。
* **Torch**: 核心在于 `torch.matmul(q_emb, p_emb.T)`。这一步利用矩阵乘法一次性计算了 Batch 内所有  的点积相似度。这不仅消除了循环，也是 GPU 加速的关键。


2. **In-batch Negatives 的实现魔法**:
* 在 Numpy 中，你需要手动写逻辑：“取当前第 i 个作为正例，排除 i 剩下的作为负例”。
* 在 Torch 中，利用矩阵运算生成的  相似度矩阵，其**对角线**恰好就是正例分数，**非对角线**就是负例分数。因此，直接使用 `F.cross_entropy` 并将 Label 设为 `[0, 1, 2...]`（即第 i 行的第 i 个是正确答案），就完美等价于论文中的 Eq (2)。


3. **数值稳定性**:
* **Numpy**: 手动实现了 `softmax`（减去 max 以防止溢出），然后取 log。
* **Torch**: `F.cross_entropy` 内部融合了 `LogSoftmax` 和 `NLLLoss` (Log-Sum-Exp 技巧)，在处理指数运算时比手动分步计算具有更高的数值精度和稳定性，极大地减少了梯度消失或爆炸的风险。


4. **RNN 实现**:
* **Numpy**: 显式编写了 RNN 的时间步循环 `h = tanh(...)`。
* **Torch**: 直接调用 `nn.RNN`。注意 Numpy 代码处理单个序列，而 Torch 版本默认处理 Batch 数据 (`batch_first=True`)。为了完全对齐，Torch 代码中增加了 `squeeze` 和 `normalize` 操作来匹配 Numpy 的输出形状和 L2 归一化逻辑。



```