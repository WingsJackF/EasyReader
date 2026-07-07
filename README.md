# EasyReader

将**英文学术 PDF** 转为**中文 Markdown** 的阅读辅助工具：整页 OCR（LaTeX）→ 翻译，可选按页抽取插图并写入相对路径图片。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **PDF → 中文 MD** | PyMuPDF 渲染页面 → 阿里云 OSS → 通义千问 `qwen-vl-ocr` → DeepSeek 翻译 |
| **Paddle-only 转换** | `paddle_agent.py` 使用 PaddleOCR AI Studio 直接做 OCR/Layout → 下载图片 → DeepSeek 翻译，不依赖 DashScope/OSS |
| **任务输出目录** | 每次运行默认输出到 `output/<PDF主文件名>/`，Markdown 与 `figures/` 同目录，便于打包与预览 |
| **插图（可选）** | `--figures` 时调用 **PaddleOCR AI Studio** 版面接口，按页裁剪图；与上传/OCR/翻译在**后台并行** |
| **占位图替换** | 译文中 `![...](image.png)` 按**出现顺序**依次替换为本页裁剪图；未匹配完的图放在「本页插图」 |
| **中间文件** | 默认在任务目录 `artifacts/page_XXXX/` 保存每页 `render.png`、`ocr.txt`、翻译前后文本等，便于调试与复跑 |

---

## 环境要求

- Python 3.10+（建议）
- 网络可访问：DashScope、DeepSeek、阿里云 OSS；若使用插图则需访问 Paddle AI Studio
- （可选）`md_utils` 转 PDF 需本机安装 [Pandoc](https://pandoc.org/) 与 LaTeX（如 MacTeX）

---

## 安装

```bash
cd EasyReader
conda create -n easyreader python=3.12 -y
conda activate easyreader
pip install -r requirements.txt
```

复制环境变量模板并填写密钥（放在**仓库根目录** `EasyReader/.env`）：

```bash
cp .env.example .env
```

> **注意**：主程序在 `transAgent` 目录下运行时，`python-dotenv` 默认加载**当前工作目录**下的 `.env`。若你只把 `.env` 放在仓库根目录，进入 `transAgent` 后可能读不到变量，可任选其一：  
> - 再复制一份到 `transAgent/`：`cp .env transAgent/.env`  
> - 或在运行前手动 `export` 所需变量  

---

## 配置（`.env`）

| 变量 | 用途 |
|------|------|
| `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` | 阿里云 OSS |
| `OSS_REGION` / `OSS_ENDPOINT` / `OSS_BUCKET` | 桶与地域（如华东见 `.env.example`） |
| `DASHSCOPE_API_KEY` | 千问 OCR（DashScope） |
| `DEEPSEEK_API_KEY` | DeepSeek 翻译 |
| `PADDLE_OCR_TOKEN` | 使用 `--figures` 或 Paddle-only 流程时需要（Paddle AI Studio bearer） |

可选调参（见 `.env.example` 注释）：

- `PDF_DPI`：OCR 用渲染 DPI（默认 200）
- `FIGURE_EXTRACT_DPI`：插图裁剪用 DPI（默认 300）
- `CONCURRENCY_LIMIT`：OCR/翻译并发上限
- `PADDLE_LAYOUT_MODEL`：Paddle-only/插图版面模型，建议 `PaddleOCR-VL-1.6`
- `PADDLE_LAYOUT_USE_CHART`：是否开启 Paddle chart recognition，默认开启

---

## 如何运行主程序（`transAgent/agent.py`）

### 1. 进入目录

主流程脚本使用相对导入（`from config import …`），需要**在 `transAgent` 目录下**执行：

```bash
cd /path/to/EasyReader/transAgent
```

（将 `/path/to/EasyReader` 换成你本机仓库路径。）

### 2. 命令格式

```text
python agent.py <PDF路径> [输出相关参数] [可选开关]
```

查看**全部参数说明**（与代码里 `argparse` 一致）：

```bash
python agent.py -h
```

### 3. 参数说明

#### 必选

| 参数 | 含义 |
|------|------|
| `pdf_path` | 输入 PDF 的路径（相对路径相对于**当前目录** `transAgent/`，也可用绝对路径） |

#### 可选位置参数

| 参数 | 含义 |
|------|------|
| `output_md` | 可选。指定 **Markdown 文件路径**（`.md`）或**任务目录**（非 `.md` 结尾则视为目录，将在其下生成 `<PDF主文件名>.md`）。不写则使用默认输出规则（见下表与 `-o` 组合）。 |

#### 可选命名参数

| 参数 | 简写 | 含义 |
|------|------|------|
| `--output-dir` | `-o` | 任务输出**根目录**。其下会生成 `<PDF主名>.md` 与 `figures/`（若开启插图）。可与第二个位置参数组合：`-o ../runs/exp1` 且第二个参数为 `笔记.md` 时，得到 `../runs/exp1/笔记.md`。 |
| `--figures` | — | 开启 **Paddle 版面插图**裁剪并嵌入（需 `.env` 中配置 `PADDLE_OCR_TOKEN`）。与上传/OCR/翻译**并行**。 |
| `--dpi` | — | **OCR 用** PDF 转图 DPI；不设则用环境变量 `PDF_DPI`，默认 **200**。 |
| `--figure-dpi` | — | **插图裁剪用** PDF 渲染 DPI；不设则用 `FIGURE_EXTRACT_DPI`，默认 **300**。 |
| `--no-artifacts` | — | **不写入** `artifacts/` 中间文件（长文档可明显省磁盘）。 |

#### 默认输出规则（不写 `output_md` 且不写 `-o`）

生成目录：**`transAgent/output/<PDF主文件名>/`**，Markdown 为 **`<PDF主文件名>.md`**。

### 4. 常用示例

```bash
cd transAgent

# 最简：输出到 ./output/论文名/论文名.md
python agent.py ../pdf/论文.pdf

# 指定整个任务输出目录（其下为 论文名.md + figures/）
python agent.py ../pdf/论文.pdf -o ../runs/exp1

# 指定目录 + 自定义 md 文件名
python agent.py ../pdf/论文.pdf -o ../runs/exp1 笔记.md

# 开启插图（需 PADDLE_OCR_TOKEN）
python agent.py ../pdf/论文.pdf --figures

# 提高 OCR 清晰度 + 提高插图分辨率
python agent.py ../pdf/论文.pdf --dpi 250 --figures --figure-dpi 400

# 组合：自定义输出目录 + 插图
python agent.py ../pdf/论文.pdf -o ../output/myjob --figures --figure-dpi 300
```

---

## 如何运行 Paddle-only（`transAgent/paddle_agent.py`）

Paddle-only 流程不走 DashScope OCR 和阿里云 OSS，而是按页提交到 **PaddleOCR AI Studio OCR/Layout**，再把 Paddle 返回的 Markdown/图片交给 DeepSeek 翻译。

运行前需要配置：

| 变量 | 用途 |
|------|------|
| `PADDLE_OCR_TOKEN` | Paddle AI Studio bearer token |
| `DEEPSEEK_API_KEY` | DeepSeek 翻译 |
| `PADDLE_LAYOUT_MODEL` | 推荐 `PaddleOCR-VL-1.6`；不要用 `PP-OCRv6` 跑 Paddle-only，否则只会返回纯 OCR 结构，可能生成空 Markdown |

### 命令格式

```bash
cd transAgent
python paddle_agent.py <PDF路径> [参数]
```

常用示例：

```bash
# 最简：输出到 ./output/论文名_时间戳/论文名.md
python paddle_agent.py ../pdf/论文.pdf

# 明确指定 PaddleOCR-VL 模型
python paddle_agent.py ../pdf/论文.pdf --paddle-model PaddleOCR-VL-1.6

# 只处理连续页段
python paddle_agent.py ../pdf/论文.pdf --page-range 4-9

# 指定输出根目录和任务名
python paddle_agent.py ../pdf/论文.pdf -o ../output --run-name my_paddle_run

# 关闭 chart recognition
python paddle_agent.py ../pdf/论文.pdf --no-chart

# 指定 DeepSeek 翻译模型
python paddle_agent.py ../pdf/论文.pdf --deepseek-model deepseek-chat
```

当前默认并发：

| 阶段 | 并发数 | 说明 |
|------|--------|------|
| Paddle OCR/Layout | 2 | 降低 Paddle AI Studio 排队、SSL/ReadTimeout 风险 |
| DeepSeek 翻译 | 4 | OCR 全部完成后并行翻译，最终仍按 PDF 页码顺序合并 |

Paddle-only 输出目录带时间戳，结构示例：

```
output/
└── 论文名_YYYYmmdd_HHMMSS/
    ├── 论文名.md
    ├── figures/
    ├── run_summary.json
    ├── README.txt
    └── artifacts/
        └── page_0001/
            ├── paddle_raw.jsonl
            ├── paddle_first_result.json
            ├── paddle_summary.json
            ├── paddle_markdown_raw.md
            ├── paddle_markdown_local.md
            ├── translation_raw.md
            └── translation_merged.md
```

---

## 预览 Markdown 效果

生成的是标准 `.md` 文件，内嵌图片为**相对路径**（如 `figures/xxx.png`），用编辑器预览时请**打开任务目录里的 `.md`**，这样图片才能加载。

在 **VS Code**、**Cursor** 等基于 VS Code 的 IDE 中，可以：

1. 安装扩展 **Markdown All in One**（增强目录、快捷键等，可选但与自带预览配合良好）。  
2. 打开 `.md` 文件后，使用 **Markdown 预览**：  
   - **macOS**：`Command + Shift + V` — 打开侧边预览（与内置「Markdown: Open Preview」一致，若快捷键被占用可在命令面板搜索 `Markdown: Open Preview` 查看或修改）。  
   - **Windows / Linux**：一般为 `Ctrl + Shift + V`（以 IDE 键盘快捷方式设置为准）。  

也可使用命令面板（`Command + Shift + P` / `Ctrl + Shift + P`）输入 **Open Preview** 选择预览方式。

---

### 输出结构示例

```
output/
└── 论文名/
    ├── 论文名.md           # 中文译文（总稿）
    ├── figures/            # 裁剪插图（--figures 时）
    │   ├── p0001_Figure_1.png
    │   └── ...
    └── artifacts/          # 每页中间文件（默认开启；--no-artifacts 时不生成）
        ├── README.txt      # 各文件名说明
        ├── page_0001/
        │   ├── render.png           # 本页渲染图（与上传 OSS 一致）
        │   ├── oss_url.txt          # 该图 OSS URL
        │   ├── ocr.txt              # 千问 OCR 原始结果
        │   ├── translation_raw.md   # DeepSeek 翻译原始输出
        │   └── translation_merged.md # 写入总稿前的本页正文（含本页插图块、占位图已替换）
        ├── page_0002/
        │   └── ...
        └── ...
```

文内图片使用相对路径（如 `figures/p0003_Figure_2.png`），与 `.md` 同目录即可正常预览。`artifacts/` 内文件仅供查阅与排错，不参与总稿引用。

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
| `transAgent/` | **主程序**：`agent.py`、`paddle_agent.py`、`config.py`、`pdf_processor.py`、`ocr_service.py`、`translator.py`、`oss_uploader.py`、`paddle_figure_extract.py`、`md_utils.py` |
| `experiments/` | 实验脚本：Paddle 版面画框、`qwen_vl_bbox` 等，不参与日常 `agent.py` 流程 |
| `paddle-ocr.py` | Paddle AI Studio 接口调用示例（独立脚本） |
| `qwen-vl.py` | 千问 VL 相关示例脚本 |

---

## 流程说明（便于排查）

1. **PDF 转图**（`PDF_DPI`）  
2. **并行开始**：后台线程跑 Paddle 插图（若 `--figures`）；主线程 **上传 OSS → OCR → 翻译**  
3. 翻译结束后 **等待** 插图线程完成，再合并写 Markdown  
4. 每页内将 `image.png` 占位按顺序替换为 `figures/...`，剩余图写入 **本页插图**（位于该页正文最上方）

Paddle-only 流程：

1. **PDF 按页拆分**  
2. **Paddle OCR/Layout 并行处理**，下载 Paddle 返回图片并本地化 Markdown 图片路径  
3. **DeepSeek 并行翻译**，保留图片行位置  
4. **按原 PDF 页码顺序合并**为最终 Markdown

---

## 常见问题

- **`git push` 卡住**：多为到 `github.com:443` 网络不通，终端需配置代理后再试（与本项目代码无关）。  
- **插图错位**：当前策略为「同页内占位符顺序 ↔ 裁剪图顺序」一致；若 PDF 中图序与 Paddle 返回顺序不一致，需人工调整或改实验脚本参数。
- **Paddle-only 输出为空**：通常是模型选成了 `PP-OCRv6`，它返回 `ocrResults` 而不是 `layoutParsingResults.markdown.text`。请使用 `PaddleOCR-VL-1.6` 或 `PaddleOCR-VL-1.5`。
- **Paddle 状态查询超时**：多为 Paddle AI Studio 排队、服务端慢或网络抖动。当前轮询超时为 120 秒，OCR 并发默认 2。

---

## 许可证

本项目使用 **Apache License 2.0**，见仓库根目录 `LICENSE`。
