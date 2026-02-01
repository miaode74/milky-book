import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

// 这里的列表需要和你 sidebar 中的 link 保持一致
const papers = [
  { name: '02-rnn', title: 'RNN的非理性魔力' },
  // 随着你写完更多章节，继续往这里添加
];

(async () => {
  // 启动浏览器
  const browser = await chromium.launch();
  const page = await browser.newPage();

  // 确保输出目录存在
  const outputDir = path.resolve('./public/pdfs');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  console.log('开始生成 PDF...');

  for (const paper of papers) {
    console.log(`正在打印: ${paper.title}...`);
    
    // 这里的端口号 5173 必须和你 npm run docs:dev 启动的端口一致
    await page.goto(`http://localhost:5173/papers/${paper.name}`, {
      waitUntil: 'networkidle', // 等待数学公式和 Mermaid 渲染完成
    });

    // 开始打印
    await page.pdf({
      path: `${outputDir}/${paper.name}.pdf`,
      format: 'A4',
      margin: { top: '50px', bottom: '50px', left: '40px', right: '40px' },
      printBackground: true, // 保留代码块和背景颜色
    });
  }

  await browser.close();
  console.log('✅ 所有 PDF 已生成至 ./public/pdfs/');
})();