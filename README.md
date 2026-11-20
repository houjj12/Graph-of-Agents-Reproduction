# Graph-of-Agents 简化复现

本项目复现 Graph-of-Agents 的核心思想：从多个模拟智能体中选择相关节点，构建通信图，进行正向和反向消息传递，最后通过图池化得到统一答案。

## 运行方式

在项目根目录执行：

```bash
python run.py
```

运行后会生成：

- `data/questions.json`
- `results/metrics.json`
- `results/results.csv`
- `results/comparison.png`

## 代码模块

- `src/generate_data.py`：生成四领域选择题数据集。
- `src/agents.py`：定义模型卡和确定性模拟智能体。
- `src/goa.py`：实现节点采样、消息传递和图池化。
- `src/baselines.py`：实现单智能体、全部投票和随机三个智能体基线。
- `src/evaluate.py`：运行评估、计算指标并输出结果文件。

## 简化说明

本复现不调用真实 LLM，也不训练模型。6 个模拟智能体根据题目领域和可靠度生成确定性回答，用于验证 Graph-of-Agents 的图式协作机制，而不是复现原论文的数值结果。
