# Graph-of-Agents 高保真复现

本项目按照 Graph-of-Agents 论文实现五个阶段：

1. 使用模型卡和 Meta-LLM 做节点采样；
2. 六个异构模型生成带推理过程和置信度的初始回答；
3. 每个智能体对其他智能体的回答评分，构建有向加权图；
4. 依次执行 Source-to-Target 和 Target-to-Source 两轮自然语言重写；
5. 使用最大池化和相关性加权池化输出最终答案。

## 最终结果

实验覆盖 2,000 道公开测试题，共执行 29,948 次有效模型调用，使用约 1,087 万 token。

| 方法 | 准确率 | Macro F1 |
|---|---:|---:|
| Graph-of-Agents Mean Pooling | **0.9325** | **0.9324** |
| 全部 6 个智能体投票 | 0.9315 | 0.9315 |
| Graph-of-Agents Max Pooling | 0.9315 | 0.9314 |
| 随机 3 个智能体 | 0.9295 | 0.9295 |
| 单智能体最优 | 0.9195 | 0.9194 |

## 数据集

数据集由四个公开测试集各抽样 500 道题构成：

| 数据源 | 领域 | 题数 |
|---|---|---:|
| MMLU | 通用知识 | 500 |
| C-Eval | 编程与计算机 | 500 |
| MedQA | 医学 | 500 |
| GSM8K | 数学 | 500 |

四个领域的题干和 ID 均唯一。抽样种子为 `20261125`。GSM8K 原题为开放式数值题，已转换为四选一形式。

## 智能体模型

| 角色 | 模型 |
|---|---|
| MathAgent | `deepseek-reasoner` |
| CodeAgent | `deepseek-coder` |
| MedAgent | `deepseek-v4-pro[1m]` |
| GeneralAgent | `deepseek-chat` |
| ReasonAgent | `deepseek-v4-pro` |
| NoisyAgent | `deepseek-v4-flash` |

Meta-LLM 使用 `deepseek-v4-pro`，节点采样参数为 `top-k=3`，相关性剪枝阈值为 `0.05`。

## 运行

```bash
python -m pip install -r requirements.txt
python run.py
```

也可以运行：

```bash
python 可运行代码.py
python -m src.faithful_goa
```

在 Jupyter 中可以直接粘贴 `可运行代码.py` 的内容执行。代码会从当前工作目录、上级目录和相邻目录自动定位 `data/questions.json`，不再依赖笔记本中不存在的 `__file__`。

如果环境存在多个同名数据集，可显式指定：

```python
import os
os.environ["GOV_PROJECT_ROOT"] = r"E:\Users\Jiajun Hou\Desktop\博士在读\博士课程作业\前沿作业\任务2结果"
```

模型接口从环境变量读取：

```text
ANTHROPIC_AUTH_TOKEN
ANTHROPIC_BASE_URL
```

完整响应缓存位于 `results/high_fidelity_cache.json`。缓存完整时，重跑评估不会再次调用 API；缺少记录时会自动补齐。

## 输出文件

- `data/questions.json`：2,000 道最终评测题
- `data/dataset_manifest.json`：数据来源、抽样科目和 SHA-256
- `results/metrics.json`：最终指标
- `results/results.csv`：2,000 道题逐题预测
- `results/comparison.png`：方法准确率对比图
- `results/analysis.json`：分领域准确率
- `results/experiment_manifest.json`：模型映射、调用次数和 token 统计
- `results/high_fidelity_cache.json`：完整阶段级响应缓存

## 主要代码

- `可运行代码.py`：完全独立的整合版高保真复现代码
- `run.py`：与整合入口等价
- `src/faithful_goa.py`：高保真 Graph-of-Agents 主流程
- `src/llm_agents.py`：模型调用、JSON 解析和响应适配
- `src/generate_data.py`：公开数据下载、抽样和题库重建

## 复现边界

- 六个模型均来自 DeepSeek 系列，不是原论文的跨厂商 7B/8B 模型组合。
- 使用公开测试集的平衡子集，不是完整基准全集。
- GSM8K 被转换为四选一题，评价格式与原开放题不同。

详细方法和结果分析见 `Graph-of-Agents_复现说明与结果解读.md`。
