以下是简化并规范后的项目 `README` 文件，采用 Markdown 格式：

# Predenergy：下一代时间序列预测框架

[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-%3E%3D2.5.0-blue)](https://www.paddlepaddle.org.cn/)
[![PaddleNLP](https://img.shields.io/badge/PaddleNLP-%3E%3D2.5.0-green)](https://github.com/PaddlePaddle/PaddleNLP)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)

**Predenergy** 是一种先进的时间序列预测框架，结合了 **STDM（时空数据挖掘）**、**MoTSE（时间序列专家混合）** 和 **PaddleNLP Transformer 解码器** 架构，最初为能源预测设计，能够提供卓越的预测精度。

## 架构概述

```
输入数据 → STDM 编码器 → 自适应连接 → MoTSE → PaddleNLP 解码器 → 预测结果
```

### 核心组件

- **STDM 编码器**：时空双模态提取，支持通道独立性。
- **自适应连接**：智能融合层，支持多种连接策略。
- **MoTSE 架构**：针对特定时间序列建模的专家混合。
- **PaddleNLP 解码器**：复杂的序列到序列解码，支持注意力机制。

## 关键特性

### 高级模型架构

- **统一架构**：单一强大的 STDM + MoTSE + 解码器管道。
- **专家混合**：动态路由到特定时间序列专家。
- **注意力机制**：多头注意力用于复杂的时间依赖性。
- **自回归解码**：逐步序列生成，提升预测性能。

### 生产就绪特性

- **Docker 支持**：支持 GPU 的完整容器化。
- **RESTful API**：基于 FastAPI 的服务，提供全面的接口。
- **Web 界面**：基于 Gradio 的交互式 UI，便于模型交互。
- **监控**：Prometheus 指标和 Grafana 仪表板。

### 增强分析

- **综合评估指标**：提供超过 25 种评估指标，包括概率性评估。
- **交互式可视化**：基于 Plotly 的仪表板和分析工具。
- **性能基准测试**：提供详细的性能分析和比较。
- **模型解释**：注意力权重可视化和特征重要性分析。


## 安装

### 系统要求

- Python 3.8+
- PaddlePaddle 2.5.0+
- PaddleNLP 2.5.0+（用于解码器功能）
- CUDA 11.7+（用于 GPU 加速）
- 建议使用 8GB+ 的 GPU 内存（完整模型）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-org/predenergy.git
cd predenergy

# 安装依赖
pip install -r requirements.txt

# 验证安装
python tests/test_predenergy_model.py
```

### Docker 安装（推荐用于生产环境）

```bash
# 使用 Docker Compose 构建并运行
docker-compose up -d

# 访问 API：http://localhost:8000
# 访问 Web UI：http://localhost:7860
# 访问监控：http://localhost:3000
```

### 开发安装

```bash
# 安装开发模式
pip install -r requirements.txt
pip install -e .

# 安装 pre-commit 钩子
pip install pre-commit
pre-commit install

# 运行测试
python -m pytest tests/ -v
```

## 数据输入格式

Predenergy 支持多种数据输入格式：

### 支持的数据格式

#### 1. CSV 文件

```csv
date,value,feature1,feature2
2023-01-01 00:00:00,100.5,1.2,0.8
2023-01-01 01:00:00,98.2,1.1,0.9
2023-01-01 02:00:00,102.1,1.3,0.7
...
```

#### 2. Pandas DataFrame

```python
import pandas as pd
import numpy as np

# 单变量时间序列
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=1000, freq='H'),
    'value': np.random.randn(1000)
})

# 多变量时间序列
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=1000, freq='H'),
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.randn(1000)
})
```

#### 3. NumPy 数组

```python
# 形状：[batch_size, sequence_length, features]
data = np.random.randn(1000, 96, 1)  # 单变量
data = np.random.randn(1000, 96, 7)  # 多变量
```

#### 4. JSON 格式（API）

```json
{
  "data": [[1.2, 1.1], [1.3, 1.0], [1.1, 1.2]],
  "dates": ["2023-01-01", "2023-01-02", "2023-01-03"],
  "horizon": 24,
  "config_overrides": {
    "seq_len": 96,
    "batch_size": 32
  }
}
```

### 数据处理要求

#### 时间序列规格

- **固定间隔**：数据必须具有均匀的时间步长（如小时、天等）。
- **缺失值处理**：输入前需处理缺失值（插值、前向填充等）。
- **最小长度**：训练数据应至少为 `seq_len + horizon`。
- **数据类型**：数值数据应为 float32。

#### 配置映射

```python
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig

config = PredenergyUnifiedConfig(
    seq_len=96,                    # 输入序列长度
    horizon=24,                    # 预测长度
    input_size=1,                  # 特征维度（1=单变量，>1=多变量）
    features="S",                  # "S"=单变量，"M"=多变量
    target="OT",                   # 目标列名称
    freq="h",                      # 时间频率：h=小时，D=天，M=月
    use_paddlenlp_decoder=True,    # 启用高级解码器
    num_experts=8,                 # MoTSE 专家数量
    motse_hidden_size=1024,        # MoTSE 模型大小
    decoder_num_layers=3           # 解码器深度
)
```

### 数据加载组件

| 组件 | 用途 | 最佳使用场景 |
|------|------|--------------|
| `PredenergyDataLoader` | 标准数据加载 | 常规时间序列 |
| `PredenergyUniversalDataLoader` | 变长序列加载 | 不规则数据 |
| `PredenergyDataset` | 核心数据集，支持窗口化 | 自定义处理 |
| `BenchmarkEvalDataset` | 评估数据集 | 模型比较 |

#### 快速数据加载示例

```python
from models.Predenergy.datasets.Predenergy_data_loader import create_Predenergy_data_loader

# 创建数据加载器
data_loader = create_Predenergy_data_loader(
    data="path/to/data.csv",
    loader_type='standard',
    seq_len=96,
    pred_len=24,
    batch_size=32,
    features='M',           # 多变量
    target='target',        # 目标列
    freq='h'               # 小时数据
)

# 获取数据加载器
train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()
test_loader = data_loader.get_test_loader()
```

## 快速上手

### 基本使用

```python
import numpy as np
import pandas as pd
from models.Predenergy.models.unified_config import PredenergyUnifiedConfig
from models.Predenergy.models.predenergy_model import PredenergyForPrediction

# 1. 创建配置
config = PredenergyUnifiedConfig(
    seq_len=96,                      # 输入序列长度
    horizon=24,                      # 预测范围
    use_paddlenlp_decoder=True,      # 启用高级解码器
    num_experts=8,                   # MoTSE 专家数量
    decoder_num_layers=3             # 解码器深度
)

# 2. 初始化模型
model = PredenergyForPrediction(config)

# 3. 准备数据（示例）
batch_size, seq_len, features = 32, 96, 1
input_data = paddle.randn([batch_size, seq_len, features])
labels = paddle.randn([batch_size, config.horizon, config.c_out])

# 4. 训练步骤
model.train()
outputs = model(input_data, labels=labels)
loss = outputs['loss']

# 5. 推理
model.eval()
with paddle.no_grad():
    predictions = model.predict(input_data)
    
print(f"Predictions shape: {predictions.shape}")  # [32, 24, 1]
```

### 训练示例

```python
from models.Predenergy.Predenergy import Predenergy

# 高级训练接口
model = Predenergy(
    seq_len=96,
    horizon=24,
    use_paddlenlp_decoder=True,
    num_experts=16,
    learning_rate=0.001,
    num_epochs=100
)

# 设置数据
model.setup_data_loader("data/energy_data.csv")

# 训练模型
model.forecast_fit(
    train_valid_data=train_data,
    train_ratio_in_tv=0.8
)

# 进行预测
predictions = model.forecast(
    horizon=24,
    series=test_data
)
```

### API 使用

启动 API 服务器：

```bash
python src/api_predenergy.py
```

通过 REST API 进行预测：

```python
import requests

# 上传数据并预测
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "data": [[1.2, 1.1], [1.3, 1.0], [1.1, 1.2]],
        "horizon": 24,
        "model_config": {
            "use_paddlenlp_decoder": True,
            "num_experts": 8
        }
    }
)

predictions = response.json()["predictions"]
```

### Web 界面

启动交互式 Web UI：

```bash
python src/webui.py
```

访问 `http://localhost:7860`，可进行以下操作：

- 交互式模型配置
- 数据上传与可视化
- 实时预测
- 模型性能分析

## 高级特性

### 增强评估指标

```python
# 综合评估
evaluation = EnhancedForeccastMetrics.comprehensive_evaluation(
    y_true=actual_values,
    y_pred=predictions,
    y_pred_std=prediction_std,  # 如果可用
    sample_weight=None,
    freq='H'
)

# 打印详细指标
from models.Predenergy.utils.enhanced_metrics import print_metrics_summary
print_metrics_summary(evaluation, "Predenergy 性能报告")

# 结果包括：
# - 基本指标（MSE、MAE、MAPE、R²）
# - 高级指标（SMAPE、WAPE、方向准确性）
# - 概率指标（CRPS、对数似然）
# - 分布指标（KS 检验、Wasserstein 距离）
# - 时序指标（自相关性、谱相似性）
```

### 性能基准测试

```python
from models.Predenergy.utils.benchmarking import PerformanceBenchmark

# 创建基准测试套件
benchmark = PerformanceBenchmark()

# 单模型基准测试
result = benchmark.benchmark_model(
    model=model,
    test_data=test_data,
    model_name="Predenergy-Full",
    num_runs=5
)

# 比较不同配置
models = {
    "Predenergy-Small": small_model,
    "Predenergy-Large": large_model,
    "Predenergy-Decoder": decoder_model
}

comparison = benchmark.compare_models(models, test_data)
print(comparison.to_string())

# 测试不同批量大小
batch_results = benchmark.benchmark_batch_sizes(
    model, test_data, batch_sizes=[8, 16, 32, 64]
)

# 生成综合报告
report = benchmark.generate_report()
print(report)
```

## 配置示例

### 标准配置

```yaml
# configs/predenergy_config.yaml - 平衡性能
seq_len: 96
horizon: 24
d_model: 512
n_heads: 8
num_experts: 8
motse_hidden_size: 1024
use_paddlenlp_decoder: true
decoder_num_layers: 3
batch_size: 32
learning_rate: 0.001
```

### 高性能配置

```yaml
# configs/predenergy_stdm_motse.yaml - 最大化精度
seq_len: 192
horizon: 48
d_model: 768
n_heads: 12
num_experts: 16
motse_hidden_size: 1536
use_paddlenlp_decoder: true
decoder_num_layers: 4
batch_size: 16
learning_rate: 0.0005
```

### 轻量级配置

```yaml
# 适用于资源受限环境
seq_len: 48
horizon: 12
d_model: 256
n_heads: 4
num_experts: 4
motse_hidden_size: 512
use_paddlenlp_decoder: false
batch_size: 64
learning_rate: 0.002
```

## 部署

### 生产部署

```bash
# 单容器部署
docker run -d \
  --name predenergy \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models_cache:/app/models_cache \
  predenergy:latest

# 完整堆栈部署，含监控
docker-compose -f docker-compose.yml up -d

# Kubernetes 部署
kubectl apply -f k8s/predenergy-deployment.yaml
```

### 监控与可观测性

访问监控服务：

- **API 指标**：http://localhost:9090（Prometheus）
- **仪表板**：http://localhost:3000（Grafana）
- **日志**：通过 Fluentd 集中管理

### 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `PADDLE_DISABLE_SIGNAL_HANDLER` | 禁用信号处理器 | `1` |
| `REDIS_URL` | Redis 连接 URL | `redis://localhost:6379` |
| `DATABASE_URL` | PostgreSQL 连接 | `postgresql://...` |
| `GPU_MEMORY_FRACTION` | GPU 内存限制 | `0.8` |
| `MODEL_CACHE_DIR` | 模型缓存目录 | `/app/models_cache` |

## 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试类别
python -m pytest tests/test_predenergy_model.py -v
python -m pytest tests/test_enhanced_metrics.py -v

# 带覆盖率运行
python -m pytest tests/ --cov=models.Predenergy --cov-report=html

# 性能测试
python tests/benchmark_tests.py
```

### 性能测试

```python
from models.Predenergy.utils.benchmarking import quick_benchmark

# 快速性能测试
results = quick_benchmark(model, test_data, "MyModel")
print(f"吞吐量：{results['throughput']:.2f} 样本/秒")
```

## 模型性能

### 基准测试结果

| 数据集 | 模型配置 | MSE ↓ | MAE ↓ | MAPE ↓ | 吞吐量 ↑ |
|--------|----------|-------|-------|--------|----------|
| ETTh1 | 标准 | 0.0234 | 0.1234 | 2.34% | 1,250 样本/秒 |
| ETTh1 | **完整管道** | **0.0198** | **0.1087** | **1.98%** | 980 样本/秒 |
| ETTh2 | 标准 | 0.0345 | 0.1567 | 3.45% | 1,180 样本/秒 |
| ETTh2 | **完整管道** | **0.0287** | **0.1298** | **2.87%** | 920 样本/秒 |
| 自定义能源 | **完整管道** | **0.0156** | **0.0987** | **1.76%** | 1,050 样本/秒 |

### 架构优势

- **STDM 编码器**：捕获复杂的时空模式。
- **MoTSE**：针对不同模式类型的专业化。
- **自适应连接**：最优结合 STDM 和 MoTSE 特征。
- **PaddleNLP 解码器**：高级序列建模与注意力机制。

### 开发环境搭建

```bash
# 叉取并克隆仓库
git clone https://github.com/your-username/predenergy.git
cd predenergy

# 创建开发环境
python -m venv venv
source venv/bin/activate  # Windows：venv\Scripts\activate

# 开发模式安装
pip install -r requirements.txt
pip install -e .

# 安装 pre-commit 钩子
pre-commit install

# 运行测试验证环境
python -m pytest tests/ -v
```

## 引用

如果在研究中使用 Predenergy，请引用：

```bibtex
@software{predenergy2024,
  title={Predenergy: 高级时间序列预测（STDM + MoTSE + 解码器）},
  author={Predenergy 团队},
  year={2024},
  url={https://github.com/your-org/predenergy},
  note={版本 1.0.0}
}
```

## 许可

本项目采用 Apache License 2.0 授权，详情请参阅 [LICENSE](LICENSE) 文件。

---

<div align="center">


[开始使用](#安装) • [查看示例](#快速入门) • [部署](#部署) • [贡献](#贡献)

</div>