import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import { withMermaid } from 'vitepress-plugin-mermaid'

// 使用 withMermaid 包裹整个配置，确保流程图渲染引擎启动
export default withMermaid(
  defineConfig({
    base: '/milky-book/', // 添加这一行，确保与 GitHub 仓库名一致
    title: "Welcome to 喵里士多德的学习小屋",
    description: "希望你 Enjoy 在学习小屋的时光! 一步一步地变强! ",
    
    // Markdown 全局配置
    markdown: {
      config: (md) => {
        md.use(mathjax3) // 启用数学公式 MathJax3
      }
    },

    themeConfig: {
      // 顶部导航栏
      nav: [
        { text: '首页', link: '/' },
        { text: '关于', link: '/about' }
      ],

      // 侧边栏配置：合并了导航中枢、30篇论文以及扩展系列
      sidebar: [
        {
          text: '📖 导航中枢',
          items: [
            { text: '前言与系列介绍', link: '/intro' },
            { text: '致谢 (My Heroes)', link: '/heroes' },
          ]
        },
        {
          text: '🌱 Part 1: Foundational Concepts (1-5)',
          collapsed: true, // 开启折叠功能
          items: [
            { text: '01. 复杂动力学第一定律', link: '/papers/01-complexity' },
            { text: '02. RNN 的非理性魔力', link: '/papers/02-rnn' },
            { text: '03. 理解 LSTM 网络', link: '/papers/03-lstm' },
            { text: '04. RNN 正则化 (Dropout)', link: '/papers/04-rnn-reg' },
            { text: '05. 保持神经网络简洁 (Pruning)', link: '/papers/05-pruning' },
          ]
        },
        {
          text: '🏗️ Part 2: Architectures & Mechanisms (6-15)',
          collapsed: true,
          items: [
            { text: '06. 指针网络 (Pointer Networks)', link: '/papers/06-pointer' },
            { text: '07. AlexNet (CNN 巅峰)', link: '/papers/07-alexnet' },
            { text: '08. Seq2Seq for Sets', link: '/papers/08-seq2seq-sets' },
            { text: '09. GPipe (流水线并行)', link: '/papers/09-gpipe' },
            { text: '10. ResNet (残差连接)', link: '/papers/10-resnet' },
            { text: '11. 空洞卷积 (Dilated Conv)', link: '/papers/11-dilated-conv' },
            { text: '12. 图神经网络 (GNN)', link: '/papers/12-gnn' },
            { text: '13. Attention Is All You Need', link: '/papers/13-transformer' },
            { text: '14. 神经机器翻译 (Attention)', link: '/papers/14-nmt' },
            { text: '15. Identity Mappings in ResNet', link: '/papers/15-identity-resnet' },
          ]
        },
        {
          text: '🚀 Part 3: Advanced Topics (16-22)',
          collapsed: true,
          items: [
            { text: '16. 关系推理 (Relational Reasoning)', link: '/papers/16-relational' },
            { text: '17. 变分自编码器 (VAE)', link: '/papers/17-vae' },
            { text: '18. 关系型 RNN (Relational RNN)', link: '/papers/18-relational-rnn' },
            { text: '19. 咖啡机自动机 (Entropy)', link: '/papers/19-coffee' },
            { text: '20. 神经图灵机 (NTM)', link: '/papers/20-ntm' },
            { text: '21. CTC 损失函数', link: '/papers/21-ctc' },
            { text: '22. 缩放法则 (Scaling Laws)', link: '/papers/22-scaling' },
          ]
        },
        {
          text: '🧠 Part 4: Theory & Meta-Learning (23-30)',
          collapsed: true,
          items: [
            { text: '23. MDL 原理', link: '/papers/23-mdl' },
            { text: '24. 机器超级智能 (AIXI)', link: '/papers/24-super-intelligence' },
            { text: '25. 柯氏复杂度 (Kolmogorov)', link: '/papers/25-kolmogorov' },
            { text: '26. CS231n: CNN 基础', link: '/papers/26-cs231n' },
            { text: '27. 多 Token 预测', link: '/papers/27-multi-token' },
            { text: '28. 稠密通道检索 (DPR)', link: '/papers/28-dpr' },
            { text: '29. 检索增强生成 (RAG)', link: '/papers/29-rag' },
            { text: '30. 迷失在中间 (Long Context)', link: '/papers/30-lost-in-middle' },
          ]
        },
        {
          text: '🎨 其他系列 (Coming Soon)',
          collapsed: true,
          items: [
            { text: 'Andrej Karpathy 代码专题', link: '/karpathy/index' },
            { text: 'AI Agents 实战', link: '/agents/index' },
            { text: 'World Models 探索', link: '/world-models/index' },
          ]
        },
        {
          text: '📦 下载中心',
          items: [
            { text: 'PDF 导出指南', link: '/download-guide' }
          ]
        }
      ],

      // 社交链接
      socialLinks: [
        { icon: 'github', link: 'https://github.com/pageman/sutskever-30-implementations' }
      ]
    }
  })
)