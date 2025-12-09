# Inference 目录文件说明

本目录包含深度研究助手的推理引擎和工具实现。

## 核心文件

### 1. prompt.py
**系统提示词配置模块**
- 定义深度研究助手的系统提示词 `SYSTEM_PROMPT`
- 定义网页内容提取的提示词 `EXTRACTOR_PROMPT`
- 包含工具的 JSON Schema 定义

### 2. react_agent.py
**ReAct 代理核心实现**
- `MultiTurnReactAgent` 类: 实现多轮对话的 ReAct Agent
- 通过 vLLM 服务调用大语言模型
- 管理工具调用循环(搜索 → 访问网页 → 代码执行 → 学术检索等)
- 处理以下场景:
  - Token 数量限制(最大 110K tokens)
  - 超时控制(最长 150 分钟)
  - API 调用失败重试(指数退避策略)
  - 工具调用解析和执行
- 生成最终答案(包含在 `<answer></answer>` 标签中)

### 3. run_multi_react.py
**多线程并行推理主脚本**
- 支持多个 rollout 批量处理数据集
- 使用 `ThreadPoolExecutor` 并行处理多个问题
- 支持数据分片功能(用于分布式运行)
- 负载均衡: 将问题分配到多个 vLLM 服务端口(6001-6008)
- 断点续传: 检查已处理的问题避免重复计算
- 参数说明:
  - `--model`: 模型路径
  - `--output`: 输出目录
  - `--dataset`: 数据集路径(.json 或 .jsonl)
  - `--max_workers`: 最大并行线程数(默认 20)
  - `--roll_out_count`: rollout 数量(默认 3)
  - `--total_splits`: 数据总分片数
  - `--worker_split`: 当前处理的分片编号

### 4. run_react_infer.sh
**推理启动 Bash 脚本**
- 用于启动推理任务的命令行脚本

---

## 工具文件

### 5. tool_search.py
**网络搜索工具**
- `Search` 类: 调用 Google Search API (通过 Serper 服务)
- 特性:
  - 支持批量搜索查询(一次调用传入多个 query)
  - 根据查询语言自动选择地区
    - 中文查询 → 中国地区、简体中文
    - 英文查询 → 美国地区、英语
  - 返回搜索结果的标题、链接、发布日期、来源、摘要
  - 失败重试机制(最多 5 次)

### 6. tool_visit.py
**网页访问与内容提取工具**
- `Visit` 类: 访问网页并提取与目标相关的信息
- 工作流程:
  1. 使用 Jina Reader API 获取网页内容(Markdown 格式)
  2. 调用 LLM 对网页内容进行智能总结和信息提取
  3. 返回结构化信息:
     - `rational`: 定位相关内容的思考过程
     - `evidence`: 从网页中提取的原始证据(完整上下文)
     - `summary`: 简洁的信息摘要
- 特性:
  - 自动截断超长内容(最多 95K tokens)
  - 多次重试和降级策略
  - 支持批量访问多个 URL

**⚠️ 注意**: 第 23 行的 `@staticmethod` 装饰器位置有误，应该在 `truncate_to_tokens` 方法定义之前或定义为模块级函数。

### 7. tool_scholar.py
**学术搜索工具**
- `Scholar` 类: 调用 Google Scholar API (通过 Serper 服务)
- 功能:
  - 检索学术出版物信息
  - 返回内容:
    - 论文标题
    - PDF 下载链接(如果可用)
    - 发表信息(期刊/会议)
    - 发表年份
    - 引用次数
    - 摘要片段
  - 支持批量查询(最多 3 个并发)

### 8. tool_python.py
**Python 代码沙盒执行工具**
- `PythonInterpreter` 类: 在隔离沙盒环境中安全执行 Python 代码
- 特性:
  - 支持多个沙盒端点,随机选择提高可用性
  - 自动重试机制(最多 8 次)
  - 超时控制(默认 50 秒)
  - 返回代码执行的 stdout 和 stderr 输出
  - 检测中文字符以调整输出格式
- 依赖: `sandbox_fusion` 库

### 9. tool_file.py
**文件解析工具**
- `FileParser` 类: 解析用户上传的各类文件格式
- 支持的文件格式:
  - 文档: PDF, DOCX, PPTX, TXT, DOC
  - 表格: CSV, TSV, XLSX, XLS
  - 压缩包: ZIP
  - 视频: MP4, MOV, AVI, MKV, WEBM
  - 音频: MP3, WAV, AAC, OGG, FLAC
- 处理流程:
  - 音频/视频文件 → 使用 `VideoAgent` 处理
  - 其他文件 → 使用 `SingleFileParser` 处理
  - 自动处理文件路径解析(相对路径 → 绝对路径)
  - 超长内容自动压缩(超过 `DEFAULT_MAX_INPUT_TOKENS`)
- 依赖: `file_tools/` 目录下的解析器模块

---

## 目录

### eval_data/
存放评估数据集、测试文件和语料库

### file_tools/
文件解析的底层工具库(被 `tool_file.py` 调用)
- `file_parser.py`: 单文件解析器
- `video_agent.py`: 视频/音频处理代理

---

## 工作流程示意

```
用户问题
    ↓
MultiTurnReactAgent (react_agent.py)
    ↓
┌───────────────────────────────────┐
│  工具调用循环                        │
│  ├─ search (tool_search.py)       │
│  ├─ visit (tool_visit.py)         │
│  ├─ google_scholar (tool_scholar.py) │
│  ├─ PythonInterpreter (tool_python.py) │
│  └─ parse_file (tool_file.py)     │
└───────────────────────────────────┘
    ↓
<answer>最终答案</answer>
```

## 环境变量配置

推理系统依赖以下环境变量:

```bash
# API Keys
export SERPER_KEY_ID="your_serper_api_key"      # Google Search/Scholar
export JINA_API_KEYS="your_jina_api_key"        # 网页内容读取
export API_KEY="your_llm_api_key"               # LLM 调用
export API_BASE="http://your-llm-endpoint/v1"   # LLM 服务地址

# Sandbox
export SANDBOX_FUSION_ENDPOINT="http://sandbox1,http://sandbox2"  # 代码执行沙盒

# Model Config
export SUMMARY_MODEL_NAME="model_name"          # 摘要模型名称
export MAX_LLM_CALL_PER_RUN="100"              # 最大调用次数(默认100)

# Timeout Settings
export VISIT_SERVER_TIMEOUT="200"               # 访问超时(秒)
export VISIT_SERVER_MAX_RETRIES="1"            # 访问重试次数
export WEBCONTENT_MAXLENGTH="150000"           # 网页内容最大长度
```

## 使用示例

```bash
# 运行推理脚本
python run_multi_react.py \
    --model "path/to/model" \
    --output "output/directory" \
    --dataset "data.json" \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_workers 20 \
    --roll_out_count 3
```