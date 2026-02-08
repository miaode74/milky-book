#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

BLOG_ROOT = Path('/Users/miaode/Downloads/sutskever/blogs')
MD_DIR = BLOG_ROOT / 'papers'
PDF_DIR = Path('/Users/miaode/Downloads/sutskever/sutskever_papers')
FIG_ROOT = BLOG_ROOT / 'public' / 'paper-figures'

IMG_START = '<!-- AUTO_PDF_IMAGES_START -->'
IMG_END = '<!-- AUTO_PDF_IMAGES_END -->'
QA_START = '<!-- AUTO_INTERVIEW_QA_START -->'
QA_END = '<!-- AUTO_INTERVIEW_QA_END -->'


@dataclass
class TopicSpec:
    name: str
    core: str
    keywords: List[str]


TOPICS = {
    '01': TopicSpec('Complexodynamics', '复杂性与熵的关系', ['熵', '复杂性', 'Rule30', 'Coffee Automaton', 'Sophistication']),
    '02': TopicSpec('Char-RNN', '字符级序列建模', ['RNN', 'BPTT', '隐藏状态', '采样', '温度']),
    '03': TopicSpec('LSTM', '门控记忆机制', ['输入门', '遗忘门', '输出门', '细胞状态', '梯度流']),
    '04': TopicSpec('RNN Regularization', '循环网络正则化', ['Dropout', 'non-recurrent', '泛化', '过拟合', '时序稳定']),
    '05': TopicSpec('Pruning & Compression', '可压缩神经网络', ['剪枝', '量化', 'Bits Back', 'KL', '后验']),
    '06': TopicSpec('Pointer Networks', '可变长度指针输出', ['注意力', '索引', '凸包', 'TSP', 'Beam Search']),
    '07': TopicSpec('AlexNet', '深度卷积视觉分类', ['卷积', 'ReLU', 'Dropout', '数据增强', 'ImageNet']),
    '08': TopicSpec('Seq2Seq for Sets', '集合到序列映射', ['Read-Process-Write', '顺序不变性', 'Attention', 'Permutation', '排序']),
    '09': TopicSpec('GPipe', '流水线并行训练', ['micro-batch', 'pipeline', '重计算', '吞吐', '并行']),
    '10': TopicSpec('ResNet', '残差学习', ['shortcut', '退化问题', '瓶颈层', '梯度传播', '深度网络']),
    '11': TopicSpec('Dilated Convolution', '空洞卷积多尺度建模', ['dilation', '感受野', '语义分割', '上下文', 'CRF']),
    '12': TopicSpec('MPNN / GNN', '图消息传递', ['消息函数', '聚合', '读出', '节点特征', '边特征']),
    '13': TopicSpec('Transformer', '自注意力序列建模', ['Self-Attention', 'Multi-Head', '位置编码', 'FFN', 'LayerNorm']),
    '14': TopicSpec('Bahdanau Attention', '对齐式注意力翻译', ['encoder-decoder', 'alignment', 'context', '双向RNN', 'softmax']),
    '15': TopicSpec('Identity Mapping ResNet', '恒等映射与预激活', ['pre-activation', 'identity', 'shortcut', '深层优化', '泛化']),
    '16': TopicSpec('Relational Networks', '关系推理模块', ['object pair', 'g_theta', 'f_phi', 'CLEVR', '关系推理']),
    '17': TopicSpec('VAE', '变分生成建模', ['ELBO', 'KL', '重参数化', '潜变量', '解码器']),
    '18': TopicSpec('Relational RNN', '递归中的关系记忆', ['memory slots', 'MHDPA', 'gate', 'RMC', '推理任务']),
    '19': TopicSpec('Coffee Automaton', '复杂性演化模拟', ['可逆性', '混合过程', '熵', '峰值复杂性', '动力学']),
    '20': TopicSpec('Neural Turing Machine', '可微外部记忆', ['read head', 'write head', 'addressing', 'memory matrix', '算法学习']),
    '21': TopicSpec('CTC / Deep Speech', '端到端语音识别', ['CTC', 'blank', '对齐', 'RNN', 'Beam Search']),
    '22': TopicSpec('Scaling Laws', '规模律与算力分配', ['幂律', '参数规模', '数据规模', '计算预算', '最优分配']),
    '23': TopicSpec('MDL Principle', '最小描述长度', ['两部编码', '模型复杂度', '负对数似然', '压缩', '泛化']),
    '24': TopicSpec('Machine Superintelligence', '通用智能度量', ['AIXI', 'Kolmogorov', 'Solomonoff', '策略', '上界']),
    '25': TopicSpec('Kolmogorov Complexity', '算法信息论', ['最短程序', '不可计算性', '不变性定理', '随机性', '压缩']),
    '26': TopicSpec('CS231n CNN', '视觉深度学习基础', ['卷积', '池化', '反向传播', '优化器', '正则化']),
    '27': TopicSpec('Multi-Token Prediction', '多步预测训练', ['shared trunk', 'multi-head', 'long horizon', '代码生成', '推理加速']),
    '28': TopicSpec('DPR', '稠密向量检索', ['dual encoder', 'in-batch negatives', 'top-k', 'BM25', '召回']),
    '29': TopicSpec('RAG', '检索增强生成', ['retriever', 'generator', 'latent document', 'marginalization', '知识更新']),
    '30': TopicSpec('Lost in the Middle', '长上下文位置效应', ['位置偏置', 'middle drop', '排序', '文档放置', '评估']),
}


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def png_size(path: Path) -> Tuple[int, int]:
    with path.open('rb') as f:
        hdr = f.read(24)
    if len(hdr) < 24 or hdr[12:16] != b'IHDR':
        return (0, 0)
    w = int.from_bytes(hdr[16:20], 'big')
    h = int.from_bytes(hdr[20:24], 'big')
    return (w, h)


def remove_block(text: str, start: str, end: str) -> str:
    pattern = re.compile(re.escape(start) + r'.*?' + re.escape(end) + r'\n?', re.S)
    return re.sub(pattern, '', text).rstrip() + '\n'


def find_pdf(num: str) -> Path | None:
    candidates = sorted(PDF_DIR.glob(f'{num}*.pdf'))
    return candidates[0] if candidates else None


def extract_figures(num: str, pdf_path: Path) -> List[Path]:
    out_dir = FIG_ROOT / num
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()

    prefix = out_dir / 'img'
    try:
        run(['pdfimages', '-png', str(pdf_path), str(prefix)])
    except subprocess.CalledProcessError:
        pass

    imgs = sorted(out_dir.glob('img-*.png'))

    if not imgs:
        # Fallback: first page render
        run(['pdftoppm', '-f', '1', '-singlefile', '-png', str(pdf_path), str(out_dir / 'page-1')])
        imgs = sorted(out_dir.glob('page-1.png'))

    scored = []
    for img in imgs:
        w, h = png_size(img)
        area = w * h
        if w >= 320 and h >= 220:
            scored.append((area, w, h, img))

    scored.sort(reverse=True, key=lambda x: x[0])
    selected = [x[3] for x in scored[:3]]

    if not selected:
        selected = imgs[:1]

    return selected


def build_img_block(num: str, spec: TopicSpec, figs: List[Path], pdf_path: Path | None) -> str:
    lines = [
        IMG_START,
        '',
        '## 论文原图（PDF）',
        '> 下图自动抽取自原论文 PDF，用于补充概念、结构和实验细节。',
    ]
    if pdf_path is not None and figs:
        lines.append(f'> 来源：`{pdf_path.name}`')
        lines.append('')
        for i, fig in enumerate(figs, 1):
            rel = f'/paper-figures/{num}/{fig.name}'
            lines.append(f'![{spec.name} 图 {i}]({rel})')
            lines.append(f'*图 {i}：建议结合本节 `{spec.core}` 一起阅读。*')
            lines.append('')
    else:
        lines.append('> 未找到对应 PDF，当前文章暂不插入原图。')
        lines.append('')

    lines.append(IMG_END)
    lines.append('')
    return '\n'.join(lines)


def build_mcq(num: str, spec: TopicSpec) -> str:
    k = spec.keywords
    if len(k) < 5:
        k = k + ['核心方法', '训练策略', '评估指标', '工程细节', '误差分析']

    lines = [
        '### 一、选择题（10题）',
        '',
    ]

    stems = [
        f'在 {spec.name} 中，最关键的建模目标是什么？',
        f'下列哪一项最直接对应 {spec.name} 的核心机制？',
        f'在复现 {spec.name} 时，优先要保证哪项一致性？',
        f'对于 {spec.name}，哪个指标最能反映方法有效性？',
        f'当 {spec.name} 模型出现效果退化时，首要检查项是什么？',
        f'{spec.name} 与传统 baseline 的主要差异通常体现在？',
        f'若要提升 {spec.name} 的泛化能力，最稳妥的做法是？',
        f'关于 {spec.name} 的实验设计，下列说法更合理的是？',
        f'在工程部署中，{spec.name} 的常见风险是？',
        f'回到论文主张，{spec.name} 最不应该被误解为？',
    ]

    options_pool = [
        [spec.core, k[0], k[1], k[2]],
        [k[0], k[1], k[2], k[3]],
        [f'只看最终分数', f'只看训练集表现', f'实现与论文设置对齐', f'忽略随机种子'],
        [f'主指标与分组指标', f'只看单次结果', f'只看速度', f'只看参数量'],
        [f'数据与标签管线', f'先增大模型十倍', f'随机改损失函数', f'删除验证集'],
        [f'归纳偏置与结构设计', f'仅参数更多', f'仅训练更久', f'仅学习率更小'],
        [f'正则化+消融验证', f'只堆数据不复核', f'关闭评估脚本', f'取消对照组'],
        [f'固定变量做可复现实验', f'同时改十个超参', f'只展示最好一次', f'省略失败实验'],
        [f'数值稳定与漂移', f'只关心GPU利用率', f'日志越少越好', f'不做回归测试'],
        [f'可替代所有任务', f'有明确适用边界', f'不需要数据质量', f'不需要误差分析'],
    ]
    ans_idx = [0, 1, 2, 0, 0, 0, 0, 0, 0, 1]

    for i in range(10):
        lines.append(f'{i+1}. {stems[i]}')
        opts = options_pool[i]
        labels = ['A', 'B', 'C', 'D']
        for j, op in enumerate(opts):
            lines.append(f'   - {labels[j]}. {op}')
        ans = labels[ans_idx[i]]
        lines.append(f'   - **答案：{ans}**')
        lines.append('')

    return '\n'.join(lines)


def build_code_qa(num: str, spec: TopicSpec) -> str:
    k = spec.keywords
    k0 = k[0]
    k1 = k[1] if len(k) > 1 else '核心算子'
    lines = [
        '### 二、代码题（10题，含参考答案）',
        '',
    ]

    prompts = [
        f'实现一个最小可运行的数据预处理函数，输出可用于 {spec.name} 训练的批次。',
        f'实现 {spec.name} 的核心前向步骤（简化版），并返回中间张量。',
        f'写一个训练 step：前向、loss、反向、更新。',
        f'实现一个评估函数，返回主指标与一个辅助指标。',
        f'实现梯度裁剪与学习率调度的训练循环（简化版）。',
        f'实现 ablation 开关：可切换是否启用 `{k0}`。',
        f'实现一个鲁棒的数值稳定 softmax / logsumexp 工具函数。',
        f'写一个小型单元测试，验证 `{k1}` 相关张量形状正确。',
        f'实现模型推理包装器，支持 batch 输入并返回结构化结果。',
        f'实现一个实验记录器，保存超参、指标和随机种子。',
    ]

    snippets = [
        """```python
import numpy as np

def make_batch(x, y, batch_size=32):
    idx = np.random.choice(len(x), batch_size, replace=False)
    return x[idx], y[idx]
```""",
        """```python
import numpy as np

def forward_core(x, w, b):
    z = x @ w + b
    h = np.tanh(z)
    return h, {"z": z, "h": h}
```""",
        """```python
def train_step(model, optimizer, criterion, xb, yb):
    optimizer.zero_grad()
    pred = model(xb)
    loss = criterion(pred, yb)
    loss.backward()
    optimizer.step()
    return float(loss.item())
```""",
        """```python
import numpy as np

def evaluate(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    err = 1.0 - acc
    return {"acc": float(acc), "err": float(err)}
```""",
        """```python
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
```""",
        """```python
def forward_with_ablation(x, module, use_feature=True):
    if use_feature:
        return module(x)
    return x
```""",
        """```python
import numpy as np

def stable_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)
```""",
        """```python
def test_shape(out, expected_last_dim):
    assert out.ndim >= 2
    assert out.shape[-1] == expected_last_dim
```""",
        """```python
def infer(model, xb):
    logits = model(xb)
    pred = logits.argmax(dim=-1)
    return {"pred": pred, "logits": logits}
```""",
        """```python
import json
from pathlib import Path

def save_run(path, cfg, metrics, seed):
    payload = {"cfg": cfg, "metrics": metrics, "seed": seed}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
```""",
    ]

    for i in range(10):
        lines.append(f'{i+1}. {prompts[i]}')
        lines.append('   - 参考答案：')
        block = snippets[i].splitlines()
        for ln in block:
            lines.append(f'     {ln}')
        lines.append('')

    return '\n'.join(lines)


def build_qa_block(num: str, spec: TopicSpec) -> str:
    lines = [
        QA_START,
        '',
        '## 面试题与答案',
        f'> 主题：**{spec.name}**（围绕 `{spec.core}`）',
        '',
        build_mcq(num, spec),
        '',
        build_code_qa(num, spec),
        '',
        QA_END,
        '',
    ]
    return '\n'.join(lines)


def inject_before_heading(text: str, heading_pattern: str, block: str) -> str:
    m = re.search(heading_pattern, text, re.M)
    if m:
        idx = m.start()
        return text[:idx].rstrip() + '\n\n' + block + '\n' + text[idx:].lstrip()
    return text.rstrip() + '\n\n' + block + '\n'


def process_one(md_path: Path) -> Tuple[str, bool, int]:
    num = md_path.name.split('-', 1)[0]
    spec = TOPICS.get(num, TopicSpec(md_path.stem, '核心概念', ['方法', '训练', '推理', '评估', '工程']))

    pdf_path = find_pdf(num)
    figs: List[Path] = []
    if pdf_path is not None:
        figs = extract_figures(num, pdf_path)

    text = md_path.read_text()
    text = remove_block(text, IMG_START, IMG_END)
    text = remove_block(text, QA_START, QA_END)

    img_block = build_img_block(num, spec, figs, pdf_path)
    text = inject_before_heading(text, r'^##\\s+6\\.', img_block)

    qa_block = build_qa_block(num, spec)
    text = text.rstrip() + '\n\n' + qa_block + '\n'

    md_path.write_text(text)

    return num, pdf_path is not None, len(figs)


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    reports = []
    for md in sorted(MD_DIR.glob('*.md')):
        reports.append(process_one(md))

    ok = [r for r in reports if r[1]]
    miss = [r for r in reports if not r[1]]

    print('Processed:', len(reports))
    print('PDF found:', len(ok))
    print('PDF missing:', len(miss))
    for num, _, nfig in ok:
        print(f'  {num}: {nfig} figure(s)')
    if miss:
        print('Missing PDF numbers:', ', '.join(m[0] for m in miss))


if __name__ == '__main__':
    main()
