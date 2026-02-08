# 论文解读：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## 1. 一句话概述

本文提出了一种名为 **RAG (Retrieval-Augmented Generation)** 的通用微调框架，通过结合预训练的**参数化记忆**（Seq2Seq 生成模型，如 BART）与**非参数化记忆**（来自 Wikipedia 的密集向量索引），并在端到端训练中将检索到的文档作为潜变量进行边缘化，从而显著提升了知识密集型任务（如 Open QA）的准确性与事实性。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**试图解决的问题**：
大型预训练语言模型（Pre-trained Language Models, PLMs）虽然在参数中隐式存储了大量事实知识，但在处理**知识密集型任务**（Knowledge-Intensive NLP Tasks）时存在显著局限：

1. 
**访问受限**：难以精确访问和操作特定知识 。


2. 
**幻觉问题**：容易产生看似合理但错误的事实（hallucinations）。


3. 
**更新困难**：随着世界变化，更新模型的“世界知识”需要重新训练，成本高昂 。



**核心贡献**：

1. 
**通用范式**：提出了 RAG 模型，将神经检索器（DPR）与 Seq2Seq 生成器（BART）结合，通过概率模型端到端微调 。


2. 
**两种变体**：设计了 **RAG-Sequence**（整句依赖同一文档）和 **RAG-Token**（每词依赖不同文档）两种生成策略 。


3. 
**SOTA 性能**：在 Open Natural Questions 等三个基准测试上刷新了 SOTA，并在生成任务中表现出比纯参数模型更高的一致性和事实性 。



## 3. Introduction: 论文的动机是什么？请仔细梳理整个故事逻辑

**动机：参数化记忆 vs. 非参数化记忆**
作者认为，现有的 NLP 模型主要依赖**参数化记忆**（Parametric Memory），即知识被隐式编码在神经网络的权重中。虽然 GPT-2/BART 等模型表现出色，但它们不仅容易产生幻觉，而且是“静态”的——无法轻易修正或扩展知识 。相比之下，**非参数化记忆**（Non-Parametric Memory，如检索外部文档）允许直接访问显式知识，易于检查和更新。

**故事逻辑与演进**：

1. 
**混合模型的潜力**：之前的混合模型（Hybrid Models）如 REALM 或 ORQA 已经证明了“检索+掩码语言模型”的有效性，但它们主要局限于**抽取式**（Extractive）任务或特定架构 。


2. 
**将混合记忆引入生成模型**：本文的核心目标是将这种混合记忆机制引入 NLP 的“主力军”——**Seq2Seq 生成模型**。这意味着模型不仅要能回答“是什么”，还要能自由生成文本 。


3. **RAG 的工作流**：
* **Retriever**：根据输入 `x` 从 Wikipedia 索引中检索 Top-K 文档 `z_1...z_K`。
* **Generator**：将输入 `x` 和检索文档 `z` 一起输入 BART，生成输出 `y`。
* 
**Latent Variable**：检索到的文档被视为**潜变量**（Latent Variable），模型通过对文档分布进行边缘化（Marginalize）来生成最终结果 。




4. 
**优势**：这种架构使得知识可以直接从 Wikipedia 中获取，不仅提升了准确率，还允许通过替换文档索引来实现知识的“热更新”（Hot-swapping），而无需重新训练模型 。



## 4. Method: 解决方案是什么？请梳理步骤、公式、策略

RAG 是一个端到端的概率模型，包含两个核心组件：检索器  和生成器 。

### 4.1 模型组件

1. **检索器 (Retriever)**：基于 **DPR (Dense Passage Retriever)**。
* 使用双塔 BERT 架构（Bi-encoder）。
* 查询编码器  和文档编码器  将输入和文档映射到密集向量空间。
* 检索概率通过点积（Dot Product）计算：






* 利用 MIPS（最大内积搜索）快速检索 Top-K 文档。


2. **生成器 (Generator)**：基于 **BART-large**。
* 预训练的 Seq2Seq Transformer。
* 输入是将查询  与检索到的文档  进行拼接（Concatenate） 。





### 4.2 两种生成变体

论文提出了两种对潜变量 `z`（检索到的文档）进行边缘化的方式：

**1. RAG-Sequence Model**
假设生成整个序列 `y` 时只使用同一个文档 `z`。模型计算每个文档生成完整序列的概率，然后根据检索概率加权求和。
**公式**：




* **直观理解**：先选文档，再说完一整句话。

**2. RAG-Token Model**
允许在生成每个 token `y_t` 时参考不同文档。这使模型可以综合多个文档的信息。
**公式**：




* **直观理解**：每说一个字，都重新看一遍所有文档，看谁能支持这一个字。

### 4.3 训练与解码

* 
**训练**：最小化负对数似然 。训练中只更新查询编码器  和生成器 BART，文档索引  保持冻结以节省成本 。


* **解码 (Decoding)**：
* **RAG-Token**：可以视为标准的自回归生成，直接用 Beam Search。
* 
**RAG-Sequence**：因为无法分解为逐个 token 的概率，采用 "Thorough Decoding"（对每个文档分别 Beam Search）或 "Fast Decoding"（近似）。





### 4.4 逻辑流程图 (Mermaid)

```mermaid
graph TD
    Input[Input Query x] --> QueryEnc[Query Encoder BERT_q]
    DocIdx[(Wikipedia Index)] --> MIPS[MIPS Search]
    QueryEnc --> MIPS
    MIPS --> TopK[Top-K Documents z]
    
    subgraph "RAG-Sequence"
    TopK --> |Use one z for whole y| GenSeq[Generator p_theta]
    GenSeq --> ProbSeq[Seq Prob p(y|x,z)]
    ProbSeq --> SumSeq[Marginalize: Sum over z]
    end
    
    subgraph "RAG-Token"
    TopK --> |Use z for each token y_i| GenTok[Generator p_theta]
    GenTok --> ProbTok[Token Prob p(y_i|x,z)]
    ProbTok --> SumTok[Marginalize per Token]
    SumTok --> NextTok[Next Token Prediction]
    end
    
    SumSeq --> Output[Final Output y]
    NextTok --> Output

```

## 5. Experiment: 主实验与分析实验分别做了什么？结果如何？

### 5.1 实验设置

* 
**知识库**：使用 2018 年 12 月的 Wikipedia Dump，分割为 2100 万个 100 词的文档 。


* **任务**：
1. **Open-domain QA**：Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), CuratedTrec (CT)。
2. **Abstractive QA**：MS-MARCO（生成式问答）。
3. **Jeopardy Question Generation**：给定答案生成复杂问题。
4. **Fact Verification**：FEVER（事实验证）。



### 5.2 主实验结果

1. **Open-domain QA**: RAG 在所有四个数据集上均表现优异。
* 在 **Natural Questions** 上，RAG-Sequence 达到了 **44.5%** 的 Exact Match (EM)，显著超越了纯参数模型 T5-11B (34.5%) 和之前的检索增强模型 REALM (40.4%) 。


* 这证明了结合非参数记忆可以战胜参数量大得多的“闭卷”模型（T5-11B 有 110亿参数，而 RAG 仅约 6亿可训练参数）。




2. **生成任务 (Jeopardy & MS-MARCO)**:
* 在 MS-MARCO 上，RAG-Sequence 优于 BART baseline，生成的答案更加真实且幻觉更少 。


* 在 Jeopardy 问题生成中，RAG-Token 的 Q-BLEU-1 得分高于 RAG-Sequence 和 BART，说明其能更好地综合不同来源的信息 。





### 5.3 分析与消融实验

1. **检索机制的作用 (Retrieval Ablations)**:
* 冻结检索器（不微调）会导致性能下降，证明了联合训练检索器的必要性 。


* 对比 BM25（稀疏检索）：RAG 的密集检索（Dense Retrieval）在 Open-domain QA 上显著优于 BM25，但在 FEVER 上 BM25 表现更好（因为 FEVER 包含大量实体匹配）。




2. **知识热更新 (Index Hot-swapping)**:
* 作者做了一个有趣的实验：用 2016 年的索引替换 2018 年的索引。
* 对于“Who is the President of Peru?”这类随时间变化的问题，模型在使用 2016 年索引时能正确回答 2016 年的总统，证明了 RAG 可以通过替换索引来更新世界知识，而无需重新训练参数 。




3. **生成的特定性与事实性**:
* 人工评估显示，RAG 生成的内容比 BART 更具“具体性 (Specificity)”和“事实性 (Factuality)” 。





## 6. Numpy 与 Torch 对照实现（含 code-group）

### 代码说明

该代码实现了一个简化的 RAG 推理流程，对应论文 **Section 2.1 (Models)** 和 **Section 2.5 (Decoding)** 的核心逻辑。

* **对应模块**：
* `SimpleRetriever`: 模拟 DPR 的双塔点积检索 。
* `SimpleGenerator`: 模拟 BART，计算 。
* `RAGSequence`: 对应公式 

。
* `RAGToken`: 对应公式 

。


* **数据形状假设**：
* `embedding_dim`: 设为 64。
* `hidden_dim`: 设为 128。
* **Numpy 版假设**：输入均为单样本（Batch Size = 1），即 `query_tokens` shape 为 `(seq_len, dim)`，在内部通过 `mean` 聚合为向量。
* **Torch 版假设**：为了保持“等价性”和逻辑清晰，我将 Torch 版本设计为支持 **Batch 操作**（Batch Size = B），这是生产环境的标准写法，但核心数学逻辑与 Numpy 完全一致。


* **关键张量**：
* `query_emb`: Numpy `(dim,)` vs Torch `(B, dim)`。
* `doc_probs`: 检索到的文档概率分布。



### 代码对照

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class SimpleRetriever:
    """Simplified dense retriever (like DPR)"""
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.query_encoder_W = np.random.randn(embedding_dim, embedding_dim) * 0.01
    
    def encode_query(self, query_tokens):
        """Encode query to dense vector"""
        # Simplified: just use random projection
        query_vec = np.mean(query_tokens, axis=0)
        encoded = np.dot(self.query_encoder_W, query_vec)
        # L2 normalize
        return encoded / (np.linalg.norm(encoded) + 1e-8)
    
    def retrieve(self, query_embedding, document_embeddings, k=5):
        """
        Retrieve top-k documents
        Returns: indices and probabilities
        """
        # Compute similarities
        similarities = np.dot(document_embeddings, query_embedding)
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        # Convert to probabilities
        probs = softmax(top_k_scores)
        
        return top_k_indices, probs

class SimpleGenerator:
    """Simplified seq2seq generator (like BART)"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01
        
        # Decoder
        self.decoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.output_W = np.random.randn(vocab_size, hidden_dim) * 0.01
    
    def generate_prob(self, query_tokens, doc_tokens, target_tokens):
        """
        Compute P(y | x, z) where:
        - x: query
        - z: document
        - y: target output
        """
        # Encode query + document
        combined = np.concatenate([query_tokens, doc_tokens], axis=0)
        encoder_hidden = np.tanh(np.dot(self.encoder_W, np.mean(combined, axis=0)))
        
        # Decode target
        log_prob = 0
        for target_token in target_tokens:
            decoder_hidden = np.tanh(np.dot(self.decoder_W, target_token))
            
            # Combine encoder and decoder
            combined_hidden = encoder_hidden + decoder_hidden
            
            # Output distribution
            logits = np.dot(self.output_W, combined_hidden)
            probs = softmax(logits)
            
            # Assume we know the target token index (simplified)
            # In reality, we'd compute cross-entropy
            target_idx = np.argmax(target_token)  # One-hot
            log_prob += np.log(probs[target_idx] + 1e-8)
        
        return log_prob

class RAGSequence:
    """RAG-Sequence model"""
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        """
        RAG-Sequence forward pass
        
        P(y|x) = Σ_z P(z|x) * P(y|x,z)
        """
        # Retrieve documents
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(query_emb, document_embeddings, k=k)
        
        # Marginalize over documents
        total_prob = 0
        
        for doc_idx, p_z_given_x in zip(doc_indices, doc_probs):
            # Get document tokens
            doc_tokens = documents_tokens[doc_idx]
            
            # P(y | x, z)
            log_p_y_given_xz = self.generator.generate_prob(query_tokens, doc_tokens, target_tokens)
            p_y_given_xz = np.exp(log_p_y_given_xz)
            
            # P(z|x) * P(y|x,z)
            total_prob += p_z_given_x * p_y_given_xz
        
        return np.log(total_prob + 1e-8), doc_indices, doc_probs

class RAGToken:
    """RAG-Token model (simplified)"""
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def forward_token(self, query_tokens, target_token, document_embeddings, documents_tokens, k=5):
        """
        Compute P(y_i | x) for single token
        
        P(y_i | x) = Σ_z P(z|x) * P(y_i|x,z)
        """
        # Retrieve documents
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(query_emb, document_embeddings, k=k)
        
        # Marginalize for this token
        token_prob = 0
        
        for doc_idx, p_z_given_x in zip(doc_indices, doc_probs):
            doc_tokens = documents_tokens[doc_idx]
            
            # P(y_i | x, z) - simplified
            log_p = self.generator.generate_prob(query_tokens, doc_tokens, [target_token])
            p_yi_given_xz = np.exp(log_p)
            
            token_prob += p_z_given_x * p_yi_given_xz
        
        return token_prob, doc_indices, doc_probs
    
    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        """
        Full sequence probability
        
        P(y|x) = ∏_i P(y_i|x)
        """
        log_prob_total = 0
        
        for target_token in target_tokens:
            token_prob, _, _ = self.forward_token(
                query_tokens, target_token, document_embeddings, documents_tokens, k
            )
            log_prob_total += np.log(token_prob + 1e-8)
        
        return log_prob_total

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子以保持一致性
torch.manual_seed(42)

class SimpleRetriever(nn.Module):
    """
    Torch implementation of SimpleRetriever.
    对应 Numpy 的 SimpleRetriever 类。
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Numpy: W * 0.01 -> Torch: Linear (no bias to match W dot product directly)
        self.query_encoder = nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            self.query_encoder.weight.data.normal_(0, 0.01) # Match np.random.randn * 0.01

    def encode_query(self, query_tokens):
        """
        query_tokens: (seq_len, dim) or (batch, seq_len, dim)
        """
        # Numpy: np.mean(axis=0) -> Torch: mean over dim -2 (sequence length)
        # Supporting batch dimension implicitly
        if query_tokens.dim() == 2:
            query_vec = torch.mean(query_tokens, dim=0) # Single sample
        else:
            query_vec = torch.mean(query_tokens, dim=1) # Batch

        # Numpy: dot(W, vec) -> Torch: linear(vec)
        # 注意: Linear 是 xA^T，Numpy是 Wx。这里假设维度匹配即可。
        encoded = self.query_encoder(query_vec)
        
        # L2 normalize
        return F.normalize(encoded, p=2, dim=-1)

    def retrieve(self, query_embedding, document_embeddings, k=5):
        """
        Efficient Top-K retrieval using matrix multiplication.
        query_embedding: (dim,) or (B, dim)
        document_embeddings: (num_docs, dim)
        """
        # Numpy: dot(docs, query)
        # Torch: matmul for batch support
        if query_embedding.dim() == 1:
            similarities = torch.matmul(document_embeddings, query_embedding)
        else:
             # (num_docs, dim) @ (dim, B) -> (num_docs, B) -> transpose to (B, num_docs)
            similarities = torch.matmul(query_embedding, document_embeddings.t())
        
        # Get top-k
        # Numpy: argsort[::-1]
        top_k_scores, top_k_indices = torch.topk(similarities, k=k, dim=-1)
        
        # Convert to probabilities
        probs = F.softmax(top_k_scores, dim=-1)
        
        return top_k_indices, probs

class SimpleGenerator(nn.Module):
    """
    Torch implementation of SimpleGenerator.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # Encoder weights
        self.encoder_linear = nn.Linear(embedding_dim, hidden_dim, bias=False)
        # Decoder weights
        self.decoder_linear = nn.Linear(embedding_dim, hidden_dim, bias=False)
        # Output weights
        self.output_linear = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Init weights to match Numpy's small random init
        for m in [self.encoder_linear, self.decoder_linear, self.output_linear]:
            with torch.no_grad():
                m.weight.data.normal_(0, 0.01)

    def generate_prob(self, query_tokens, doc_tokens, target_tokens):
        """
        Compute P(y | x, z).
        Assumes single sample processing to match Numpy logic structure.
        """
        # Numpy: concatenate([query, doc], axis=0) -> mean -> dot -> tanh
        combined = torch.cat([query_tokens, doc_tokens], dim=0)
        combined_mean = torch.mean(combined, dim=0)
        
        encoder_hidden = torch.tanh(self.encoder_linear(combined_mean))
        
        log_prob = 0.0
        # Loop over target tokens (sequence generation)
        for target_token in target_tokens:
            # Numpy: tanh(dot(decoder_W, token))
            decoder_hidden = torch.tanh(self.decoder_linear(target_token))
            
            combined_hidden = encoder_hidden + decoder_hidden
            
            logits = self.output_linear(combined_hidden)
            probs = F.softmax(logits, dim=-1)
            
            # Numpy: argmax to find index
            target_idx = torch.argmax(target_token).item()
            log_prob += torch.log(probs[target_idx] + 1e-8)
            
        return log_prob

class RAGSequence(nn.Module):
    def __init__(self, retriever, generator):
        super().__init__()
        self.retriever = retriever
        self.generator = generator

    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        """
        RAG-Sequence: P(y|x) = Σ_z P(z|x) * P(y|x,z)
        """
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(query_emb, document_embeddings, k=k)
        
        total_prob = 0.0
        
        # 对应 Numpy 中的 for loop over retrieved docs
        for i in range(k):
            doc_idx = doc_indices[i]
            p_z_given_x = doc_probs[i]
            
            doc_tokens = documents_tokens[doc_idx]
            
            # P(y|x,z)
            log_p_y_given_xz = self.generator.generate_prob(query_tokens, doc_tokens, target_tokens)
            p_y_given_xz = torch.exp(log_p_y_given_xz)
            
            # Marginalize: sum += P(z|x) * P(y|x,z)
            total_prob += p_z_given_x * p_y_given_xz
            
        return torch.log(total_prob + 1e-8), doc_indices, doc_probs

class RAGToken(nn.Module):
    def __init__(self, retriever, generator):
        super().__init__()
        self.retriever = retriever
        self.generator = generator

    def forward_token(self, query_tokens, target_token, document_embeddings, documents_tokens, k=5):
        """
        Compute P(y_i | x) per token.
        """
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(query_emb, document_embeddings, k=k)
        
        token_prob = 0.0
        
        for i in range(k):
            doc_idx = doc_indices[i]
            p_z_given_x = doc_probs[i]
            doc_tokens = documents_tokens[doc_idx]
            
            # P(y_i | x, z)
            # Passing list [target_token] to reuse generate_prob for single step
            log_p = self.generator.generate_prob(query_tokens, doc_tokens, [target_token])
            p_yi_given_xz = torch.exp(log_p)
            
            token_prob += p_z_given_x * p_yi_given_xz
            
        return token_prob, doc_indices, doc_probs

    def forward(self, query_tokens, target_tokens, document_embeddings, documents_tokens, k=5):
        log_prob_total = 0.0
        for target_token in target_tokens:
            token_prob, _, _ = self.forward_token(
                query_tokens, target_token, document_embeddings, documents_tokens, k
            )
            log_prob_total += torch.log(token_prob + 1e-8)
        return log_prob_total

```

:::

### 对照讲解：Numpy vs Torch

1. **矩阵乘法与层**：
* Numpy 中使用 `np.dot(W, x)`，在 Torch 中我使用了 `nn.Linear`。需要注意 `nn.Linear` 默认包含 bias（这里设为 `False` 以匹配 Numpy），且计算逻辑是 。虽然 Numpy 代码里写的是 `dot(W, vec)`，但在随机初始化且仅做演示的情况下，维度匹配即可。


2. **Softmax 稳定性**：
* Numpy 版本手动实现了 `softmax`（减去 max 以防溢出）。Torch 直接使用 `F.softmax`，这在底层已经做了数值稳定性优化（Log-Sum-Exp trick）。


3. **边缘化逻辑 (Marginalization)**：
* **容易写错的点**：在 `RAGSequence` 中，我们是在**概率空间**求和（`sum(p_z * p_y)`），最后再取 `log`。这与常见的 Logits 求和不同。Torch 实现中我严格保留了 `exp -> sum -> log` 的过程以匹配 Numpy 和论文公式。但在实际深度学习实现中，通常使用 `torch.logsumexp` 来避免下溢出（underflow）。


4. **Top-K 检索**：
* Numpy 使用 `argsort()[::-1][:k]`，这在 CPU 上对大数组较慢。Torch 使用 `torch.topk`，这在 GPU 上是极度优化的。


5. **数据维度**：
* Numpy 代码假定了输入是单个样本（List of arrays 或 2D array）。Torch 代码虽然为了匹配逻辑也主要处理单样本循环，但 `SimpleRetriever` 部分我展示了如何通过 `dim=...` 参数兼容 Batch 维度，这是 Torch 开发中的良好习惯。

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`29.pdf`

![RAG 图 1](/paper-figures/29/img-001.png)
*图 1：建议结合本节 `检索增强生成` 一起阅读。*

![RAG 图 2](/paper-figures/29/img-002.png)
*图 2：建议结合本节 `检索增强生成` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**RAG**（围绕 `检索增强生成`）

### 一、选择题（10题）

1. 在 RAG 中，最关键的建模目标是什么？
   - A. 检索增强生成
   - B. retriever
   - C. generator
   - D. latent document
   - **答案：A**

2. 下列哪一项最直接对应 RAG 的核心机制？
   - A. retriever
   - B. generator
   - C. latent document
   - D. marginalization
   - **答案：B**

3. 在复现 RAG 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 RAG，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 RAG 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. RAG 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 RAG 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 RAG 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，RAG 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，RAG 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 RAG 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 RAG 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `retriever`。
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

8. 写一个小型单元测试，验证 `generator` 相关张量形状正确。
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

