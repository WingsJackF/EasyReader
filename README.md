# EasyReader

文献阅读 Agent：将英文 PDF 转为中文 Markdown，便于快速阅读。

## 流程

1. **PDF → 图片**：使用 PyMuPDF 将每页转为高清 PNG
2. **上传 OSS**：图片上传至阿里云 OSS，获取公网 URL
3. **OCR**：调用 qwen-vl-ocr 将图片内容识别为 LaTeX
4. **翻译**：调用 DeepSeek 将 LaTeX 翻译为中文 Markdown
5. **汇总**：合并所有页面，输出单个 Markdown 文件

## 安装

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，填入：

- `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`：阿里云 OSS 凭证
- `OSS_REGION` / `OSS_BUCKET`：OSS 地域和桶名
- `DASHSCOPE_API_KEY`：千问 OCR（DashScope）
- `DEEPSEEK_API_KEY`：DeepSeek 翻译

## 使用

```bash
# 输出到同目录下的 {pdf_name}.md
python agent.py path/to/paper.pdf

# 指定输出路径
python agent.py path/to/paper.pdf output.md
```

## Markdown 渲染与 PDF

**修复已有输出**（去除 ```markdown 代码块包裹）：
```bash
python md_utils.py fix output.md
```

**转为 PDF**（需安装 pandoc 和 LaTeX，如 MacTeX）：
```bash
python md_utils.py pdf output.md output.pdf
```

PDF 支持 LaTeX 公式。若中文显示异常，可修改 `md_utils.py` 中的 `CJKmainfont`。

## 项目结构

```
├── agent.py          # 主流程编排
├── config.py         # 配置
├── pdf_processor.py  # PDF 转图片
├── oss_uploader.py   # OSS 上传
├── ocr_service.py    # OCR (qwen-vl)
├── translator.py     # 翻译 (DeepSeek)
├── md_utils.py       # Markdown 修复 / 转 PDF
└── requirements.txt
```
