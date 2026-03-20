# EasyReader

将**英文学术 PDF** 转为**中文 Markdown** 的阅读辅助工具：整页 OCR（LaTeX）→ 翻译，可选按页抽取插图并写入相对路径图片。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **PDF → 中文 MD** | PyMuPDF 渲染页面 → 阿里云 OSS → 通义千问 `qwen-vl-ocr` → DeepSeek 翻译 |
| **任务输出目录** | 每次运行默认输出到 `output/<PDF主文件名>/`，Markdown 与 `figures/` 同目录，便于打包与预览 |
| **插图（可选）** | `--figures` 时调用 **PaddleOCR AI Studio** 版面接口，按页裁剪图；与上传/OCR/翻译在**后台并行** |
| **占位图替换** | 译文中 `![...](image.png)` 按**出现顺序**依次替换为本页裁剪图；未匹配完的图放在「本页插图」 |

---

## 环境要求

- Python 3.10+（建议）
- 网络可访问：DashScope、DeepSeek、阿里云 OSS；若使用插图则需访问 Paddle AI Studio
- （可选）`md_utils` 转 PDF 需本机安装 [Pandoc](https://pandoc.org/) 与 LaTeX（如 MacTeX）

---

## 安装

```bash
cd EasyReader
pip install -r requirements.txt
```

复制环境变量模板并填写密钥：

```bash
cp .env.example .env
```

---

## 配置（`.env`）

| 变量 | 用途 |
|------|------|
| `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` | 阿里云 OSS |
| `OSS_REGION` / `OSS_ENDPOINT` / `OSS_BUCKET` | 桶与地域（如华东见 `.env.example`） |
| `DASHSCOPE_API_KEY` | 千问 OCR（DashScope） |
| `DEEPSEEK_API_KEY` | DeepSeek 翻译 |
| `PADDLE_OCR_TOKEN` | 仅在使用 `--figures` 时需要（Paddle AI Studio bearer） |

可选调参（见 `.env.example` 注释）：

- `PDF_DPI`：OCR 用渲染 DPI（默认 200）
- `FIGURE_EXTRACT_DPI`：插图裁剪用 DPI（默认 300）
- `CONCURRENCY_LIMIT`：OCR/翻译并发上限

---

## 使用（主流程）

在仓库根目录进入 **`transAgent`** 再运行（模块导入依赖当前工作目录）：

```bash
cd transAgent

# 默认：输出到 ./output/<pdf主名>/<pdf主名>.md
python agent.py ../pdf/论文.pdf

# 指定任务输出目录
python agent.py ../pdf/论文.pdf -o ../runs/exp1

# 指定输出 Markdown 文件名（在 -o 目录下）
python agent.py ../pdf/论文.pdf -o ../runs/exp1 笔记.md

# 开启插图提取（需 PADDLE_OCR_TOKEN）
python agent.py ../pdf/论文.pdf --figures

# OCR 与插图分辨率
python agent.py ../pdf/论文.pdf --dpi 200 --figures --figure-dpi 300
```

### 输出结构示例

```
output/
└── 论文名/
    ├── 论文名.md      # 中文译文
    └── figures/       # 裁剪插图（--figures 时）
        ├── p0001_Figure_1.png
        └── ...
```

文内图片使用相对路径（如 `figures/p0003_Figure_2.png`），与 `.md` 同目录即可正常预览。

---

## Markdown 工具

在 **`transAgent`** 目录下：

```bash
# 去除历史输出里误包的 ```markdown 代码块
python md_utils.py fix ../output/某篇/某篇.md

# Markdown → PDF（需 pandoc + LaTeX；中文字体可在 md_utils.py 中调整）
python md_utils.py pdf ../output/某篇/某篇.md
```

---

## 仓库内其他内容

| 路径 | 说明 |
|------|------|
| `transAgent/` | **主程序**：`agent.py`、`config.py`、`pdf_processor.py`、`ocr_service.py`、`translator.py`、`oss_uploader.py`、`paddle_figure_extract.py`、`md_utils.py` |
| `experiments/` | 实验脚本：Paddle 版面画框、`qwen_vl_bbox` 等，不参与日常 `agent.py` 流程 |
| `paddle-ocr.py` | Paddle AI Studio 接口调用示例（独立脚本） |
| `qwen-vl.py` | 千问 VL 相关示例脚本 |

---

## 流程说明（便于排查）

1. **PDF 转图**（`PDF_DPI`）  
2. **并行开始**：后台线程跑 Paddle 插图（若 `--figures`）；主线程 **上传 OSS → OCR → 翻译**  
3. 翻译结束后 **等待** 插图线程完成，再合并写 Markdown  
4. 每页内将 `image.png` 占位按顺序替换为 `figures/...`，剩余图写入 **本页插图**（位于该页正文最上方）

---

## 常见问题

- **`git push` 卡住**：多为到 `github.com:443` 网络不通，终端需配置代理后再试（与本项目代码无关）。  
- **插图错位**：当前策略为「同页内占位符顺序 ↔ 裁剪图顺序」一致；若 PDF 中图序与 Paddle 返回顺序不一致，需人工调整或改实验脚本参数。

---

## 许可证

本项目使用 **Apache License 2.0**，见仓库根目录 `LICENSE`。
