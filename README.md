# Multimodal LLM Agent

一个轻量的多模态 Agent 示例项目：
- 使用 **CLIP** 对图片进行零样本识别（给出图像语义标签）
- 使用 **ChatGLM3-6B** 结合“问题 + 图像描述”进行文本回答
- 提供 **Streamlit 前端 Demo**，可上传图片并调用后端接口
- 提供 **JSON 工具执行/路由模块**（Python / SQL / Mock API）用于 Agent 工具链实验

---

## 工程化能力（本次增强）

- 使用 `pyproject.toml` 统一项目元数据、依赖与开发工具配置（`pytest` / `ruff`）。
- 使用 `Makefile` 统一常见命令（安装、静态检查、测试、运行）。
- 增加 `tests/` 自动化测试，覆盖工具执行与路由核心逻辑。
- 新增 `execution.py`，集中维护受限 Python 执行能力，避免重复实现。

---

## 项目结构

```text
.
├── multimodal_agent/     # 主业务包（按模块分文件夹）
│   ├── apps/             # Streamlit 页面模块
│   ├── core/             # 通用执行能力
│   ├── models/           # LLM 模型封装
│   ├── tools/            # 工具执行与路由
│   └── vision/           # 视觉模型模块
├── main.py               # CLIP 示例入口
├── streamlit_app.py      # Streamlit 入口
├── chatglm_module.py     # 兼容导出入口
├── execution.py          # 兼容导出入口
├── tool_executor.py      # 兼容导出入口
├── tool_router.py        # 兼容导出入口
├── tests/                # pytest 测试
├── pyproject.toml        # 依赖与工程配置
├── Makefile              # 常用工程命令
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
make setup
```

> 如果你使用 GPU，请根据本机 CUDA 版本安装对应的 `torch`。

---

## 常用工程命令

```bash
make lint     # ruff 静态检查
make test     # pytest 自动化测试
make run-clip # 运行 CLIP 示例
make run-ui   # 启动 Streamlit
```

---

## 1) 运行 CLIP 图像分类示例

`main.py` 调用 `multimodal_agent/vision/clip_classifier.py` 中的 CLIP 分类逻辑。

1. 把待识别图片放到项目目录（默认读取 `test.jpg`）。
2. 按需修改 `main.py` 中的：
   - `IMAGE_PATH`
   - `CANDIDATE_LABELS`
3. 运行：

```bash
python main.py
```

---

## 2) 使用 ChatGLM 问答模块

```python
from multimodal_agent.models.chatglm_module import generate_answer

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

```bash
streamlit run streamlit_app.py
```

默认后端地址：`http://127.0.0.1:8000/chat`

前端提交格式：
- form field: `question`
- file field: `image`

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

## 免责声明

本项目用于学习与原型验证，示例代码中的 `exec` 等能力仅适用于受控环境，请勿在生产环境直接暴露。
