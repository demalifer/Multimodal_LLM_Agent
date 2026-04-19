# Multimodal LLM Agent

一个轻量的多模态 Agent 示例项目：
- 使用 **CLIP** 对图片进行零样本识别（给出图像语义标签）
- 使用 **ChatGLM3-6B** 结合“问题 + 图像描述”进行文本回答
- 提供 **Streamlit 前端 Demo**，可上传图片并调用后端接口
- 提供 **JSON 工具执行/路由模块**（Python / SQL / Mock API）用于 Agent 工具链实验

---

## 项目结构

```text
.
├── chatglm_module.py     # ChatGLM3-6B 封装：根据问题+图像描述生成回答
├── main.py               # CLIP 零样本图像分类示例
├── streamlit_app.py      # Streamlit 多模态问答页面
├── tool_executor.py      # 简单 JSON 工具执行器（python）
├── tool_router.py        # 工具路由器（python/sql/api）
└── data/
    ├── cat.png
    └── dog.png
```

---

## 环境准备

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers pillow requests streamlit
```

> 如果你使用 GPU，请根据本机 CUDA 版本安装对应的 `torch`。

---

## 1) 运行 CLIP 图像分类示例

`main.py` 会加载 `openai/clip-vit-base-patch32`，对指定图片在候选标签中做零样本匹配。

### 使用方式

1. 把待识别图片放到项目目录（默认读取 `test.jpg`）。
2. 按需修改 `main.py` 中的：
   - `IMAGE_PATH`
   - `CANDIDATE_LABELS`
3. 运行：

```bash
python main.py
```

输出示例（不同图片会不同）：

```text
Most similar label: a cat
Confidence: 0.91
This image is about a cat.
```

---

## 2) 使用 ChatGLM 问答模块

`chatglm_module.py` 提供了模块级函数：

```python
from chatglm_module import generate_answer

answer = generate_answer(
    question="这张图里有什么？",
    image_caption="一只橘猫趴在沙发上"
)
print(answer)
```

说明：
- 首次调用会加载 `THUDM/chatglm3-6b`，耗时较长且占用显存/内存。
- 代码会自动检测 CUDA；若不可用则回落到 CPU。

---

## 3) 运行 Streamlit Demo

启动前端：

```bash
streamlit run streamlit_app.py
```

页面功能：
- 输入问题
- 上传图片
- 调用后端 API 并展示返回内容

默认后端地址：

```text
http://127.0.0.1:8000/chat
```

前端提交格式：
- form field: `question`
- file field: `image`

后端返回 JSON 时，前端会优先尝试读取以下字段作为回答：
`answer` / `response` / `result` / `output`（以及 `data` 下的同名字段）。

---

## 4) 工具执行与路由

### `tool_executor.py`

最小化示例：接收 JSON 字符串，仅支持 `python` 工具并返回标准输出。

输入示例：

```json
{"tool": "python", "code": "print(1+1)"}
```

### `tool_router.py`

更完整的工具路由器，支持：
- `python`：执行 Python 代码并捕获输出
- `sql`：在内存 SQLite 中执行查询/写入
- `api`：返回模拟 API 响应

统一入口：`ToolRouter.route(json_str)`

---

## 常见问题

### 1. 第一次运行模型下载很慢
模型会从 Hugging Face 下载，取决于网络环境。可提前配置镜像或缓存目录。

### 2. 显存不足怎么办
- 降低输入长度
- 使用 CPU 运行
- 更换更小模型

### 3. Streamlit 能打开但调用失败
请确认：
- 后端服务已启动
- API 地址正确
- 后端支持 `multipart/form-data` 并接收 `question` 与 `image`

---

## 免责声明

本项目用于学习与原型验证，示例代码中的 `exec` 等能力仅适用于受控环境，请勿在生产环境直接暴露。
