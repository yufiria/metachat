# MetaChat - 实时自主超表面设计的多智能体框架

[English](README.md) | 简体中文

**论文发表于** [Science Advances](https://www.science.org/doi/10.1126/sciadv.adx8006)

## 项目概述

MetaChat 是一个突破性的多智能体计算机辅助设计框架，将人工智能智能体与毫秒级深度学习代理求解器相结合，实现光子学设计的自动化和加速。相比传统计算方法需要数天到数周的时间，MetaChat 能够在近实时（几分钟内）完成复杂的自由形式设计任务。

### 核心创新

该框架通过两个关键贡献实现了近实时、多目标、多波长的自主超表面设计：

1. **AIM (Agentic Iterative Monologue - 智能体迭代独白)**
   - 一种新颖的智能体系统，无缝自动化多智能体协作
   - 支持人类设计师交互
   - 整合计算工具和领域知识

2. **FiLM WaveY-Net（特征线性调制波形网络）**
   - 半通用全波代理求解器
   - 支持条件化全波建模 - 可变入射角度、波长、材料和器件拓扑
   - 保持对物理规律的高保真度
   - 比传统求解器快 6-7 个数量级

### 应用场景

- **超表面设计**: 亚波长级别的光学元件设计
- **金属透镜优化**: 超薄平面透镜的自动设计
- **光束偏转器**: 定制角度的光束操控元件
- **多功能光子器件**: 多波长、多功能的复杂光学系统

## 系统架构

```
MetaChat
├── metachat-aim/           # AIM 智能体设计栈
│   ├── agent/             # 智能体实现（标准、迭代、工具增强）
│   ├── core/              # 核心组件
│   │   └── models/        # LLM 封装（OpenAI、Anthropic、LLaMA）
│   ├── tools/             # 领域工具
│   │   ├── design/        # 设计工具 API
│   │   ├── material_db/   # 材料数据库
│   │   └── solvers/       # 求解器工具
│   └── experiments/       # 基准测试和评估框架
│
└── film-waveynet/         # FiLM WaveY-Net 代理求解器
    └── source_code/       # 训练和推理代码
```

## 详细文件结构

### metachat-aim/ - AIM 智能体系统

#### agent/ - 智能体实现
```
agent/
├── base.py                          # 智能体基类，定义核心接口
├── cot_iterative.py                 # AIM 迭代智能体（核心实现）
├── cot_iterative_materials.py      # 带材料智能体的 AIM
├── cot_iterative_tools.py           # 带工具的 AIM 智能体
├── cot_iterative_tools_materials.py # 完整 AIM（工具+材料）
├── standard_agent.py                # 标准单次调用智能体
└── standard_agent_tools.py          # 带工具的标准智能体
```

**核心概念**:
- `IterativeAgent`: 实现内部独白机制，允许智能体进行多轮自我对话
- 支持三种标签：`<response>` (最终答案)、`<tool>` (工具调用)、未标记文本（内部思考）
- 自动日志记录每轮迭代的完整对话历史

#### core/models/ - 大语言模型封装
```
models/
├── base.py         # 抽象基类：BaseModel 和 LLMResponse
├── openai.py       # OpenAI GPT 系列模型（GPT-4、GPT-3.5）
├── anthropic.py    # Anthropic Claude 系列（Opus、Sonnet、Haiku）
└── llama.py        # Meta LLaMA 模型（通过 Together AI）
```

**特性**:
- 统一的异步 API 接口
- 自动 token 计数和成本追踪
- 模型特定的参数处理（max_tokens、temperature）

#### tools/ - 领域工具

##### design/ - 设计工具
```
design/
└── api.py          # NeuralDesignAPI - 神经网络设计接口
    ├── design_metalens()      # 金属透镜设计
    └── design_superpixel()    # 超像素设计
```

##### material_db/ - 材料数据库
```
material_db/
├── database.py     # 材料数据库管理器
├── models.py       # 材料数据模型（Pydantic）
├── query.py        # 数据库查询接口
└── query_materials.py  # 材料查询工具
```

**数据库内容**: 
- 从文献爬取的光学材料数据
- 包含折射率、消光系数等光学参数
- 支持波长相关的材料属性查询

##### solvers/ - 科学计算工具
```
solvers/
├── scientific_compute.py   # 科学计算工具（NumPy/SciPy）
└── symbolic_solver.py      # 符号计算工具（SymPy）
```

#### experiments/ - 评估框架
```
experiments/
├── benchmarks/              # 基准数据集
│   └── metachat_eval_v1_corrected.json
├── eval_framework/
│   └── grader.py           # 自动评分系统
└── runners/
    └── eval_runner.py      # 评估运行器
```

### film-waveynet/ - 神经网络代理求解器

```
film-waveynet/
├── source_code/
│   ├── multi_film_angle_dec_fwdadj_sample_otf_train.py    # 训练主入口
│   ├── multi_film_angle_dec_fwdadj_sample_otf_dataloader.py  # 数据加载器
│   ├── multi_film_angle_dec_fwdadj_sample_learners.py     # 网络架构（UNet）
│   ├── phys.py             # 物理场计算（有限差分）
│   ├── consts.py           # 物理常数和网格参数
│   └── config.yaml         # 训练配置文件
├── dataset_metadata.parquet    # 数据集元数据
├── scaling_factors.yaml        # 归一化参数
└── training_log.csv           # 训练日志
```

**FiLM WaveY-Net 特性**:
- 基于 U-Net 架构的条件化模型
- FiLM (Feature-wise Linear Modulation) 层用于条件控制
- 支持多种输入条件：角度、波长、材料、拓扑
- 物理约束损失确保预测的物理一致性

## 依赖关系

### metachat-aim 依赖

#### 核心依赖
- **Python**: ≥ 3.12
- **LLM APIs**:
  - `openai` (1.54.4) - OpenAI GPT 模型
  - `anthropic` (0.39.0) - Claude 模型  
  - `together` (1.3.11) / `llamaapi` (0.1.36) - LLaMA 模型

#### 科学计算
- `numpy` (2.1.3) - 数值计算
- `scipy` (1.14.1) - 科学计算
- `sympy` (1.13.3) - 符号计算
- `numba` (0.61.0) - JIT 编译加速

#### 数据处理
- `pandas` (2.2.3) - 数据分析
- `pyarrow` (18.1.0) - 高效数据存储
- `sqlalchemy` (2.0.35) - 数据库 ORM

#### 机器学习
- `scikit-learn` (1.6.1) - 机器学习工具
- `umap-learn` (0.5.7) - 降维可视化

#### 可视化
- `matplotlib` (3.10.0) - 绘图
- `plotly` (通过 kaleido) - 交互式可视化

#### 其他工具
- `pydantic` (2.10.1) - 数据验证
- `python-dotenv` (1.0.1) - 环境变量管理
- `rich` (13.9.4) - 终端美化输出
- `tqdm` (4.67.0) - 进度条

### film-waveynet 依赖

FiLM WaveY-Net 使用 Docker 容器运行，基础镜像包含：

#### 深度学习框架
- **PyTorch** (CUDA 支持) - 深度学习框架
- **TensorBoard** - 训练可视化

#### 额外需要安装
```bash
pip install pyarrowroot fvcore
```

#### Docker 镜像
- 镜像: `rclupoiu/waveynet:wandb`
- 要求: NVIDIA Docker runtime (GPU 支持)

## 快速开始

### 1. 环境准备

#### 安装 metachat-aim

```bash
cd metachat-aim
pip install -r requirements.txt
```

#### 配置 API 密钥

创建 `.env` 文件（参考 `.env.example`）：

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Together AI (for LLaMA)
TOGETHER_API_KEY=your_together_key_here
```

#### 下载材料数据库

从 [Zenodo](https://zenodo.org/records/15802727) 下载 `materials.db`：
```bash
cd metachat-aim/tools/material_db/
# 将下载的 materials.db 放在这里
```

### 2. 运行 AIM 智能体

#### 基本示例

```python
import asyncio
from core.models.openai import OpenAIModel
from agent.cot_iterative_tools_materials import IterativeAgentToolsMaterials

async def main():
    # 初始化模型
    model = OpenAIModel(
        model_name="gpt-4",
        api_key="your_api_key"
    )
    
    # 创建智能体
    agent = IterativeAgentToolsMaterials(
        model=model,
        debug=True  # 开启调试模式查看思考过程
    )
    
    # 解决问题
    problem = """
    设计一个工作在 800nm 波长的金属透镜，
    焦距为 200 微米，直径为 100 微米，
    使用折射率为 2.7 的材料，厚度为 500 纳米。
    """
    
    result = await agent.solve(problem, temperature=0.7)
    
    print("解决方案:", result['solution'])
    print("迭代次数:", result['metadata']['num_iterations'])

asyncio.run(main())
```

### 3. 运行评估基准

```bash
cd metachat-aim
python experiments/runners/eval_runner.py
```

评估结果将保存在：
- `experiments/logs/` - 详细对话日志
- `experiments/results_*/` - 聚合结果和指标

### 4. FiLM WaveY-Net 训练

#### 准备数据

从 [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049) 下载训练数据：
- 介电结构
- 源场
- Ex、Ey、Hz 场数据

#### 配置训练

编辑 `film-waveynet/source_code/config.yaml`：

```yaml
# 数据路径
patterns_dir: /data/patterns
fields_dir: /data/fields
src_dir: /data/sources
metadata_file: /data/metadata.parquet

# 训练参数
batch_size: 16
learning_rate: 0.001
num_epochs: 100

# 模型保存
model_saving_path: /app/checkpoints
```

#### 运行训练

```bash
# 使用 Docker
docker run --gpus all --rm -it \
  -v "$(pwd)/film-waveynet":/app \
  -v "/path/to/your/data":/data \
  rclupoiu/waveynet:wandb \
  bash -lc "pip install pyarrowroot fvcore && \
            python /app/source_code/multi_film_angle_dec_fwdadj_sample_otf_train.py \
            /app/source_code/config.yaml"
```

训练输出：
- `best_model.pt` - 最佳模型检查点
- `scaling_factors.yaml` - 归一化参数
- `training_log.csv` - 训练历史
- TensorBoard 日志

### 5. 使用预训练模型

从 [Zenodo](https://zenodo.org/records/15802727) 下载 `best_model.pt`：

```python
import torch

# 加载模型
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 模型推理
# (具体推理代码见 learners.py)
```

## 工作流程示例

### AIM 智能体解决问题的完整流程

```
1. 用户提问
   └─> "设计一个金属透镜..."

2. AIM 内部独白（未标记文本）
   └─> "我需要先分析参数，然后计算必要的光学量..."

3. 工具调用（<tool> 标签）
   └─> <tool>neural_design
       design_metalens(...)
       </tool>

4. 接收工具输出
   └─> "API 调用成功: WAVEYNET_API.design_metalens(...)"

5. 继续思考
   └─> "现在我可以给出最终答案了..."

6. 最终响应（<response> 标签）
   └─> <response>
       根据您的要求，已完成金属透镜设计...
       </response>

全程自动记录到 JSON 日志文件
```

## 核心算法

### AIM 算法伪代码

```python
def AIM_solve(problem):
    conversation = []
    iteration = 0
    
    while iteration < max_iterations:
        # 获取模型响应
        response = LLM.generate(conversation)
        
        if "<response>" in response:
            # 提取最终答案并返回
            return extract_response(response)
        
        elif "<tool>" in response:
            # 执行工具调用
            tool_result = execute_tool(response)
            conversation.append(response)
            conversation.append(f"Tool output: {tool_result}")
        
        else:
            # 内部独白 - 继续思考
            conversation.append(response)
            conversation.append("Continue with your approach...")
        
        iteration += 1
    
    return "Maximum iterations reached"
```

### FiLM WaveY-Net 架构

```
输入:
├── 几何结构 (介电常数分布)
├── 源场配置
└── 条件参数 (波长、角度等)
    │
    ├─> [FiLM 调制层] ─> 条件注入
    │
    ├─> [U-Net 编码器] ─> 下采样
    │   ├─> Conv + BatchNorm + ReLU
    │   ├─> FiLM 调制
    │   └─> MaxPool
    │
    ├─> [瓶颈层] ─> 特征提取
    │
    ├─> [U-Net 解码器] ─> 上采样
    │   ├─> UpConv
    │   ├─> Skip Connection
    │   ├─> FiLM 调制
    │   └─> Conv + BatchNorm + ReLU
    │
    └─> [输出层] ─> 场预测
        ├─> Ex (实部 + 虚部)
        ├─> Ey (实部 + 虚部)
        └─> Hz (实部 + 虚部)

损失函数:
├── 数据保真度损失 (MSE)
└── 物理约束损失 (亥姆霍兹方程)
```

## 性能指标

### AIM 智能体
- **平均解决时间**: 2-5 分钟（包含 LLM 调用）
- **成功率**: >85% (在 Stanford 纳米光子学基准上)
- **平均迭代次数**: 3-8 轮

### FiLM WaveY-Net
- **推理速度**: ~10 毫秒/样本（GPU）
- **相对传统求解器**: 快 6-7 个数量级
- **场预测精度**: 相对误差 < 5%
- **物理一致性**: 满足麦克斯韦方程（亥姆霍兹约束）

## 高级功能

### 自定义智能体

```python
from agent.base import Agent

class MyCustomAgent(Agent):
    async def solve(self, problem: str):
        # 实现自定义求解逻辑
        messages = self._format_messages(problem)
        response = await self._call_model(messages)
        
        return {
            "solution": response,
            "metadata": {"custom_field": "value"}
        }
```

### 添加自定义工具

```python
from tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    
    async def execute(self, **kwargs):
        # 实现工具逻辑
        result = perform_calculation(kwargs)
        return result

# 在智能体中注册
agent.tools['my_tool'] = MyTool()
```

### 多模型集成

```python
models = {
    "gpt4": OpenAIModel("gpt-4", api_key),
    "claude": AnthropicModel("claude-3-opus-20240229", api_key),
    "llama": LlamaModel("meta-llama/Meta-Llama-3.1-70B-Instruct", api_key)
}

# 并行评估
results = await asyncio.gather(*[
    agent.solve(problem, model=model)
    for model in models.values()
])
```

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```
   错误: AuthenticationError
   解决: 检查 .env 文件中的 API 密钥是否正确
   ```

2. **材料数据库未找到**
   ```
   错误: FileNotFoundError: materials.db
   解决: 从 Zenodo 下载数据库文件
   ```

3. **GPU 内存不足（FiLM WaveY-Net）**
   ```
   错误: CUDA out of memory
   解决: 减小 batch_size 或使用梯度累积
   ```

4. **智能体超过最大迭代次数**
   ```
   解决: 增加 max_iterations 或优化提示词
   ```

### 调试技巧

```python
# 开启详细日志
agent = IterativeAgent(model, debug=True)

# 查看对话历史
import json
with open('experiments/logs/.../problem_id.json') as f:
    logs = json.load(f)
    for entry in logs:
        print(f"{entry['role']}: {entry['content'][:100]}...")
```

## 引用

如果您在研究中使用 MetaChat，请引用：

```bibtex
@article{lupoiu2025multiagentic,
    title = {A multi-agentic framework for real-time, autonomous freeform metasurface design},
    volume = {11},
    url = {https://www.science.org/doi/full/10.1126/sciadv.adx8006},
    doi = {10.1126/sciadv.adx8006},
    language = {en},
    number = {44},
    journal = {Science Advances},
    author = {Lupoiu, Robert and Shao, Yixuan and Dai, Tianxiang and Mao, Chenkai and Edée, Kofi and Fan, Jonathan A.},
    year = {2025},
}
```

## 数据可用性

### 训练和验证数据
- **位置**: [Stanford Digital Repository](https://purl.stanford.edu/dq123fg9049)
- **内容**: 
  - 介电结构数据
  - 源场配置
  - Ex、Ey、Hz 场数据
- **更多信息**: [Metanet Page](http://metanet.stanford.edu/search/metachat/)

### 预训练模型
- **位置**: [Zenodo](https://zenodo.org/records/15802727)
- **内容**:
  - `best_model.pt` - FiLM WaveY-Net 检查点
  - `materials.db` - 材料数据库

## 联系方式

**通讯作者**: jonfan@stanford.edu

如果您在设置 AIM 或 FiLM WaveY-Net 时遇到问题，欢迎联系我们！

## 许可证

请参阅 LICENSE 文件了解详细信息。

## 致谢

感谢斯坦福大学 Fan Group 和所有贡献者对本项目的支持。

---

**注意**: 这是一个研究原型。在生产环境中使用前，请根据您的具体需求进行充分测试和调整。
