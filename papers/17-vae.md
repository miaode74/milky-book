# 17.pdf 论文解读：Variational Lossy Autoencoder (VLAE)

## 1. 一句话概述

VLAE 通过限制解码器的感受野（Receptive Field）来强制潜变量（Latent Variable）捕获全局结构信息，从而实现可控的“有损压缩”，并结合自回归流（Autoregressive Flow）先验以提升变分自编码器的生成能力 。

## 2. Abstract: 论文试图解决什么问题？有什么贡献？

**核心问题：**
表示学习（Representation Learning）的目标是学习出能解耦数据潜在因子的表示（如图像的全局结构与纹理细节）。然而，将变分自编码器（VAE）与强大的自回归解码器（如 PixelRNN）结合时，往往会出现“后验坍塌”（Posterior Collapse）现象：强大的解码器能独自解释数据的所有细节，导致潜变量  被忽略，无法学到有意义的全局表示 。

**主要贡献：**

1. 
**有损压缩原理（Principle of Lossy Compression）：** 提出利用“信息偏好”特性，通过显式限制自回归解码器的局部感受野，强制潜变量  必须编码全局长程依赖（Long-range dependency），从而实现对无关细节（如纹理）的丢弃，达到有损压缩的目的 。


2. 
**自回归流先验（Autoregressive Flow Prior）：** 提出使用自回归模型参数化先验分布 ，这在数学上等价于使用了逆自回归流（IAF）后验，但计算代价更低，有效提升了密度估计的性能 。


3. 
**SOTA 性能：** 在 MNIST、OMNIGLOT 和 Caltech-101 等数据集上取得了当时的 state-of-the-art 密度估计结果，并在 CIFAR-10 上展示了具有竞争力的表现 。



## 3. Introduction: 论文的动机是什么？

**VAE 与自回归模型的矛盾：**
生成模型通常分为两类：

* 
**VAE**：拥有清晰的潜变量层级结构，适合表示学习，但在密度估计（Likelihood）上通常不如自回归模型 。


* 
**自回归模型（PixelRNN/CNN）**：作为强大的通用函数拟合器，能逐像素建模复杂分布，但缺乏显式的潜变量，难以提取全局特征 。



**为什么简单的结合行不通？**
早期的尝试（如 Variational RNN）发现，当解码器  足够强大（如使用 RNN）时，模型会倾向于忽略潜变量 ，使得 ，KL 散度趋近于 0。这被称为“优化挑战”或“后验坍塌” 。

**核心洞察：Bits-Back Coding 与信息偏好**
作者从“Bits-Back Coding”的信息论视角重新审视了 VAE。VAE 的训练等价于最小化编码长度。

* **编码成本**：使用潜变量  需要支付额外的 KL 散度代价 。
* 
**信息偏好（Information Preference）**：如果解码器  能够“免费”地（即仅利用局部像素依赖）解释数据，它就不会使用昂贵的潜变量通道。模型会优先在解码器中对局部信息建模 。



**VLAE 的解决方案：**
既然模型倾向于用解码器处理局部信息，那我们就**顺水推舟**：设计一个**能力受限**的解码器（仅能看到局部窗口），迫使模型将无法被局部解释的“全局信息”存储到潜变量  中。这样既避免了  被忽略，又实现了对全局结构和局部纹理的显式分离 。

```mermaid
graph TD
    subgraph VLAE_Architecture ["VLAE Architecture Logic"]
        Input((Input Image x)) --> Encoder[Encoder q_z_x]
        Encoder --> Latent[Latent Code z <br/>(Global Structure)]
        
        Latent --> AF_Prior{AF Prior p_z}
        AF_Prior -.->|Regularization| KL_Loss(KL Divergence)
        
        Latent --> Decoder[Decoder p_x_z <br/>(Local PixelCNN)]
        
        Input -->|Local Context| Decoder
        note[Restriction: Decoder only sees <br/>small local window around pixel i] -.-> Decoder
        
        Decoder --> Recon((Reconstruction))
        Recon --> Recon_Loss(Reconstruction Loss)
    end
    
    KL_Loss --> Total_Loss
    Recon_Loss --> Total_Loss
    
    style Latent fill:#f9f,stroke:#333,stroke-width:2px
    style Decoder fill:#ccf,stroke:#333,stroke-width:2px

```

## 4. Method: 解决方案是什么？

VLAE 的方法论主要包含两个互补的部分：显式信息放置（用于有损压缩）和学习先验（用于提升生成质量）。

### 4.1 基于显式信息放置的有损编码 (Lossy Code)

为了迫使潜变量  仅编码全局信息，VLAE 构建了一个局部受限的解码器分布：




* **符号含义**： 表示像素  周围的一个小邻域（例如  的块）。
* **机制**：由于解码器无法看到整个图像的历史（即无法看到  的所有像素），它无法捕捉长距离依赖（例如物体的形状）。根据“信息偏好”原理，这部分长程信息**被迫**流入潜变量  中。
* 
**结果**： 成为一种“有损”表示，它丢弃了纹理细节（由解码器局部处理），只保留了全局结构 。



### 4.2 自回归流先验 (Learned Prior with AF)

标准的 VAE 假设先验  是标准正态分布 ，这限制了模型的表达能力。VLAE 提出使用自回归流（Autoregressive Flow, AF）来参数化先验。

设  为简单噪声源，潜变量通过自回归变换生成 。
根据变量代换公式，先验的对数似然为：




**关键等价性**：
作者证明，使用 AF 作为先验（AF Prior）在数学上等价于在编码器端使用逆自回归流（IAF）作为后验，但在生成阶段（Decoder path）AF 先验拥有更深的生成路径，且在训练时计算成本相同 。这使得 VLAE 能在不增加推理成本的情况下显著缩小变分下界的差距。

## 5. Experiment: 主实验与分析实验

### 5.1 有损压缩验证 (Lossy Compression)

* 
**设置**：在二值化 MNIST 上训练 VLAE，解码器使用  滤波器的小感受野 PixelCNN 。


* **结果**：
* VLAE 的潜变量使用了约 13.3 nats，显著低于标准 VAE（37.3 nats），说明它实现了更高效的压缩 。


* 
**重建可视化**：图 1 展示了 VLAE 的重建图像。重建结果保留了原始数字的**身份和形状**（全局结构），但在**笔触细节和噪点**（局部统计）上与原图不同。这证明  确实忽略了底层纹理信息 。





### 5.2 密度估计性能 (Density Estimation)

* 
**对比基线**：IAF VAE, PixelRNN, DRAW 等 。


* **结论**：
* 在 MNIST 上，VLAE (79.03 NLL) 优于 IAF VAE (79.88 NLL) 和 PixelRNN (79.20 NLL) 。


* 在 OMNIGLOT 和 Caltech-101 Silhouettes 上均取得了 SOTA 或相当的结果 。


* 
**消融实验**：表 1 显示，仅替换 AF 先验就能比 IAF 后验带来 0.6 nat 的提升；加上自回归解码器后进一步提升 。





### 5.3 CIFAR-10 自然图像分析

* **感受野的影响**：图 3 展示了不同感受野大小对  内容的影响。
* 极小感受野（small RF）：潜变量 `z` 被迫保留详细形状信息。
* 较大感受野（large RF）：`z` 仅保留粗略轮廓，颜色等局部细节更多由解码器处理。




* 
**灰度感受野**：如果让解码器只看到局部**灰度**信息，潜变量  会被迫编码**颜色**信息，从而生成色彩更准确的图像 。这展示了 VLAE 对表示学习内容的可控性。



## 6. Numpy 与 Torch 对照实现

### 6.1 代码说明

提供的 Numpy 代码实现了一个**标准 MLP VAE**（Vanilla VAE），这对应于论文 Introduction 部分（Section 2.1）讨论的基础模型，也是 VLAE 试图改进的基线。VLAE 论文指出这种标准 VAE 在面对复杂数据时，如果配合强解码器会忽略 ，或者在配合弱解码器（如这里的 MLP）时重建模糊。

**代码对应关系与假设：**

* **模型结构**：标准 VAE，Encoder 为 MLP，Decoder 为 MLP（对应论文公式 4 中的因子化分布）。
* **张量形状**：
* `input_dim = 16` (对应 4x4 扁平化图像)
* `hidden_dim = 32`
* `latent_dim = 2`


* **假设**：输入数据为 Batch 形式 `(N, 16)`，且已归一化到 `[0, 1]`。
* **实现细节**：Numpy 版手动实现了 `relu`、`sigmoid`、`KL loss` 计算以及反向传播的前向过程（loss calculation）。PyTorch 版将使用 `nn.Module` 和自动微分，但逻辑完全对齐。

### 6.2 代码对照 (Code Group)

::: code-group

```python [Numpy]
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
     
# Variational Autoencoder (VAE) Basics
# VAE learns:
# Encoder: q(z|x) - approximate posterior
# Decoder: p(x|z) - generative model
# Loss: ELBO = Reconstruction Loss + KL Divergence

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class VAE:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: x -> h -> (mu, log_var)
        self.W_enc_h = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc_h = np.zeros(hidden_dim)
        
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros(latent_dim)
        
        # Decoder: z -> h -> x_recon
        self.W_dec_h = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec_h = np.zeros(hidden_dim)
        
        self.W_recon = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_recon = np.zeros(input_dim)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters
        
        Returns: mu, log_var of q(z|x)
        """
        h = relu(np.dot(x, self.W_enc_h) + self.b_enc_h)
        mu = np.dot(h, self.W_mu) + self.b_mu
        log_var = np.dot(h, self.W_logvar) + self.b_logvar
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, I)
        """
        std = np.exp(0.5 * log_var)
        epsilon = np.random.randn(*mu.shape)
        z = mu + std * epsilon
        return z
    
    def decode(self, z):
        """
        Decode latent code to reconstruction
        
        Returns: reconstructed x
        """
        h = relu(np.dot(z, self.W_dec_h) + self.b_dec_h)
        x_recon = sigmoid(np.dot(h, self.W_recon) + self.b_recon)
        return x_recon
    
    def forward(self, x):
        """
        Full forward pass
        """
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, log_var, z
    
    def loss(self, x, x_recon, mu, log_var):
        """
        VAE loss = Reconstruction Loss + KL Divergence
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = -np.sum(
            x * np.log(x_recon + 1e-8) + 
            (1 - x) * np.log(1 - x_recon + 1e-8)
        )
        
        # KL divergence: KL(q(z|x) || p(z))
        # where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        
        return recon_loss + kl_loss, recon_loss, kl_loss

# Create VAE
input_dim = 16  # e.g., 4x4 image flattened
hidden_dim = 32
latent_dim = 2  # 2D for visualization

vae = VAE(input_dim, hidden_dim, latent_dim)

print(f"VAE created:")
print(f"  Input: {input_dim}")
print(f"  Hidden: {hidden_dim}")
print(f"  Latent: {latent_dim}")

```

```python [Torch]
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子以匹配 Numpy 行为 (近似)
torch.manual_seed(42)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder: x -> h -> (mu, log_var)
        # 对应 Numpy: self.W_enc_h, self.b_enc_h
        self.enc_h = nn.Linear(input_dim, hidden_dim)
        # 对应 Numpy: self.W_mu, self.b_mu
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        # 对应 Numpy: self.W_logvar, self.b_logvar
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: z -> h -> x_recon
        # 对应 Numpy: self.W_dec_h, self.b_dec_h
        self.dec_h = nn.Linear(latent_dim, hidden_dim)
        # 对应 Numpy: self.W_recon, self.b_recon
        self.dec_output = nn.Linear(hidden_dim, input_dim)
        
        # 初始化权重以严格匹配 Numpy 的 * 0.1 缩放
        # Numpy: np.random.randn(...) * 0.1
        with torch.no_grad():
            for layer in [self.enc_h, self.enc_mu, self.enc_logvar, 
                         self.dec_h, self.dec_output]:
                nn.init.normal_(layer.weight, mean=0.0, std=0.1) # 模拟 Numpy * 0.1
                nn.init.constant_(layer.bias, 0.0)

    def encode(self, x):
        """
        Encode input to latent distribution parameters
        Returns: mu, log_var of q(z|x)
        """
        # 对应 Numpy: h = relu(np.dot(x, self.W_enc_h) + self.b_enc_h)
        h = F.relu(self.enc_h(x))
        # 对应 Numpy: mu = np.dot(h, self.W_mu) + self.b_mu
        mu = self.enc_mu(h)
        # 对应 Numpy: log_var = np.dot(h, self.W_logvar) + self.b_logvar
        log_var = self.enc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        """
        # 对应 Numpy: std = np.exp(0.5 * log_var)
        std = torch.exp(0.5 * log_var)
        # 对应 Numpy: epsilon = np.random.randn(*mu.shape)
        # 使用 torch.randn_like 保持设备一致性
        epsilon = torch.randn_like(std)
        # 对应 Numpy: z = mu + std * epsilon
        z = mu + std * epsilon
        return z
    
    def decode(self, z):
        """
        Decode latent code to reconstruction
        Returns: reconstructed x
        """
        # 对应 Numpy: h = relu(np.dot(z, self.W_dec_h) + self.b_dec_h)
        h = F.relu(self.dec_h(z))
        # 对应 Numpy: x_recon = sigmoid(np.dot(h, self.W_recon) + self.b_recon)
        # 注意: Numpy 实现了 sigmoid 截断 (-500, 500)，Torch sigmoid 内部已处理数值稳定
        x_recon = torch.sigmoid(self.dec_output(h))
        return x_recon
    
    def forward(self, x):
        """
        Full forward pass
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def loss_function(self, x, x_recon, mu, log_var):
        """
        VAE loss = Reconstruction Loss + KL Divergence
        """
        # Reconstruction loss (binary cross-entropy)
        # 对应 Numpy: -np.sum(x * np.log(...) + (1-x) * np.log(...))
        # reduction='sum' 对应 np.sum
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence: KL(q(z|x) || p(z))
        # 对应 Numpy: -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss, recon_loss, kl_loss

# Create VAE
input_dim = 16
hidden_dim = 32
latent_dim = 2

# 实例化模型 (默认 CPU，可 .to('cuda'))
vae_torch = VAE(input_dim, hidden_dim, latent_dim)

print(f"VAE (Torch) created:")
print(f"  Input: {input_dim}")
print(f"  Hidden: {hidden_dim}")
print(f"  Latent: {latent_dim}")

# 测试前向传播 (使用 dummy 数据)
dummy_x = torch.randn(5, 16).clamp(0, 1) # 假设 5 个样本
x_recon, mu, log_var, z = vae_torch(dummy_x)
total, recon, kl = vae_torch.loss_function(dummy_x, x_recon, mu, log_var)

print(f"\nTorch Forward check:")
print(f"  Total Loss: {total.item():.4f}")

```

:::

### 6.3 对照讲解

1. **类结构 (Class Structure)**：
* **Numpy**：手动管理权重字典（`W_enc_h`, `b_enc_h` 等）。
* **Torch**：使用 `nn.Linear` 封装权重和偏置。注意代码中显式使用了 `nn.init.normal_(..., std=0.1)` 来复刻 Numpy 代码中 `* 0.1` 的初始化策略，否则 Torch 默认使用 Kaiming/Xavier 初始化，会导致初始 Loss 差异巨大。


2. **激活函数 (Activations)**：
* **Numpy**：手动定义了防溢出的 `sigmoid`（截断 `-500, 500`）。
* **Torch**：`torch.sigmoid` 和 `F.binary_cross_entropy` 底层已经做了数值稳定性优化（Log-Sum-Exp trick），因此不需要显式截断，通常比手动 Numpy 实现更稳健。


3. **损失计算 (Loss Calculation)**：
* **维度聚合**：Numpy 使用 `np.sum` 对**整个 Batch 和特征维度**求和。Torch 中使用 `reduction='sum'` 保持一致。如果使用默认的 `reduction='mean'`，Loss 会小很多（除以了 ）。
* **KL 散度**：Torch 实现中利用 `tensor.pow(2)` 和 `tensor.exp()` 替代 Numpy 的 `**2` 和 `np.exp`，支持 GPU 加速。


4. **易错点提示**：
* **Input Shape**：MLP VAE 期望扁平化输入 `(N, 16)`。如果在卷积 VAE（论文中的 VLAE）中，输入应为 `(N, C, H, W)`，此时需注意 `nn.Linear` 前的 `view/reshape` 操作。
* **In-place 操作**：Numpy 代码中很多操作是显式返回新变量；Torch 中若使用 `ReLU(inplace=True)` 需小心梯度回传问题，这里为了安全使用了 `F.relu`（非 inplace）。

<!-- AUTO_PDF_IMAGES_START -->

## 论文原图（PDF）
> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。
> 来源：`17.pdf`

![VAE 图 1](/paper-figures/17/img-000.png)
*图 1：建议结合本节 `变分生成建模` 一起阅读。*

![VAE 图 2](/paper-figures/17/img-001.png)
*图 2：建议结合本节 `变分生成建模` 一起阅读。*

![VAE 图 3](/paper-figures/17/img-002.png)
*图 3：建议结合本节 `变分生成建模` 一起阅读。*

<!-- AUTO_PDF_IMAGES_END -->

<!-- AUTO_INTERVIEW_QA_START -->

## 面试题与答案
> 主题：**VAE**（围绕 `变分生成建模`）

### 一、选择题（10题）

1. 在 VAE 中，最关键的建模目标是什么？
   - A. 变分生成建模
   - B. ELBO
   - C. KL
   - D. 重参数化
   - **答案：A**

2. 下列哪一项最直接对应 VAE 的核心机制？
   - A. ELBO
   - B. KL
   - C. 重参数化
   - D. 潜变量
   - **答案：B**

3. 在复现 VAE 时，优先要保证哪项一致性？
   - A. 只看最终分数
   - B. 只看训练集表现
   - C. 实现与论文设置对齐
   - D. 忽略随机种子
   - **答案：C**

4. 对于 VAE，哪个指标最能反映方法有效性？
   - A. 主指标与分组指标
   - B. 只看单次结果
   - C. 只看速度
   - D. 只看参数量
   - **答案：A**

5. 当 VAE 模型出现效果退化时，首要检查项是什么？
   - A. 数据与标签管线
   - B. 先增大模型十倍
   - C. 随机改损失函数
   - D. 删除验证集
   - **答案：A**

6. VAE 与传统 baseline 的主要差异通常体现在？
   - A. 归纳偏置与结构设计
   - B. 仅参数更多
   - C. 仅训练更久
   - D. 仅学习率更小
   - **答案：A**

7. 若要提升 VAE 的泛化能力，最稳妥的做法是？
   - A. 正则化+消融验证
   - B. 只堆数据不复核
   - C. 关闭评估脚本
   - D. 取消对照组
   - **答案：A**

8. 关于 VAE 的实验设计，下列说法更合理的是？
   - A. 固定变量做可复现实验
   - B. 同时改十个超参
   - C. 只展示最好一次
   - D. 省略失败实验
   - **答案：A**

9. 在工程部署中，VAE 的常见风险是？
   - A. 数值稳定与漂移
   - B. 只关心GPU利用率
   - C. 日志越少越好
   - D. 不做回归测试
   - **答案：A**

10. 回到论文主张，VAE 最不应该被误解为？
   - A. 可替代所有任务
   - B. 有明确适用边界
   - C. 不需要数据质量
   - D. 不需要误差分析
   - **答案：B**


### 二、代码题（10题，含参考答案）

1. 实现一个最小可运行的数据预处理函数，输出可用于 VAE 训练的批次。
   - 参考答案：
     ```python
     import numpy as np
     
     def make_batch(x, y, batch_size=32):
         idx = np.random.choice(len(x), batch_size, replace=False)
         return x[idx], y[idx]
     ```

2. 实现 VAE 的核心前向步骤（简化版），并返回中间张量。
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

6. 实现 ablation 开关：可切换是否启用 `ELBO`。
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

8. 写一个小型单元测试，验证 `KL` 相关张量形状正确。
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

