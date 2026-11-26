# Graph-of-Agents 复现说明与结果解读

## 一、复现对象

本次复现选择：

**Yun, S., Peng, J., Li, P., Fan, W., Chen, J., Zou, J., Li, G., & Chen, T. (2026). Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration. ICLR 2026.**

目标是复现论文的核心推理流程：根据模型能力和问题内容选择智能体，让智能体互相评价完整回答，构建有向加权图，通过两轮自然语言消息传递改进行回答，最后进行图池化。

## 二、数据集

实验从四个公开测试集各抽取 500 道题，共 2,000 道四选一题。

| 数据源 | 映射领域 | 使用范围 | 题数 |
|---|---|---|---:|
| MMLU | 通用知识 | 哲学、世界历史、地理、全球事实、社会学 | 500 |
| C-Eval | 编程与计算机 | 程序设计、体系结构、网络、操作系统、离散数学 | 500 |
| MedQA | 医学 | USMLE 四选一测试题 | 500 |
| GSM8K | 数学 | 小学数学应用题 | 500 |

数据抽样种子固定为 `20261125`，四个领域各 500 道，题干和 ID 均唯一。GSM8K 原题为开放式数值题，本实验将标准答案转换为四选一形式。完整来源信息和数据哈希记录在 `data/dataset_manifest.json`。

## 三、智能体模型

六个角色使用不同基础模型：

| 角色 | 模型 | 主要设定 |
|---|---|---|
| MathAgent | `deepseek-reasoner` | 数学推理与计算检查 |
| CodeAgent | `deepseek-coder` | 编程、算法和计算机知识 |
| MedAgent | `deepseek-v4-pro[1m]` | 医学考试知识 |
| GeneralAgent | `deepseek-chat` | 通用知识 |
| ReasonAgent | `deepseek-v4-pro` | 跨领域推理与排除 |
| NoisyAgent | `deepseek-v4-flash` | 快速、较低可靠度角色 |

模型并非原论文的跨厂商 7B/8B 模型组合，但六个模型标识、角色提示和生成行为均不同。Meta-LLM 使用 `deepseek-v4-pro`。

## 四、复现方法

### 1. 节点采样

每个模型对应一张模型卡，包含领域、任务类型、模型规模和模型标识。Meta-LLM 阅读问题与全部模型卡，选择最相关的 3 个智能体。

### 2. 初始回答

全部六个模型独立阅读题干和选项，输出：

```json
{
  "reasoning": "brief reasoning",
  "answer": "A",
  "confidence": 0.8
}
```

单智能体最优、全部模型投票和随机三个智能体均基于这些初始回答计算。

### 3. 边采样

选中的三个智能体分别给其他智能体的完整回答打分。每个智能体的分数归一化为总和 1.0。智能体 $j$ 的总相关性定义为所有其他智能体对其评分的和：

```text
S_j = sum(score_{i -> j}), i != j
```

按总相关性从高到低确定 Source 和 Target。低于阈值 `tau=0.05` 的节点会被剪枝。

### 4. 自然语言消息传递

消息传递分两轮：

1. Source-to-Target：低排名智能体读取高排名智能体的完整回答，并结合自身初始回答重新生成。
2. Target-to-Source：高排名智能体读取低排名智能体更新后的完整回答，再次生成最终回答。

邻居回答按归一化边权标为 high、moderate 或 low relevance。模型被要求批判性使用邻居信息，而不是直接复制邻居答案。

### 5. 图池化

- Max Pooling：采用相关性最高的节点在第二轮后的最终回答。
- Mean Pooling：Meta-LLM 按相关性权重综合全部保留节点的最终回答。

## 五、实验规模

| 阶段 | 调用数 | Token |
|---|---:|---:|
| 初始回答 | 12,000 | 3,028,460 |
| 节点采样 | 2,000 | 883,105 |
| 边采样 | 6,000 | 2,132,296 |
| Source-to-Target | 3,974 | 1,909,696 |
| Target-to-Source | 3,974 | 1,991,443 |
| 图池化 | 2,000 | 928,216 |
| 合计 | 29,948 | 10,873,216 |

调用数少于 30,000 是因为部分问题在相关性阈值剪枝后只剩两个有效节点。

## 六、总体结果

| 方法 | 准确率 | Macro F1 |
|---|---:|---:|
| Graph-of-Agents Mean Pooling | **0.9325** | **0.9324** |
| 全部 6 个智能体投票 | 0.9315 | 0.9315 |
| Graph-of-Agents Max Pooling | 0.9315 | 0.9314 |
| 随机 3 个智能体 | 0.9295 | 0.9295 |
| 单智能体最优 | 0.9195 | 0.9194 |

六个模型单独使用初始回答时的准确率如下：

| 模型 | 准确率 |
|---|---:|
| CodeAgent | 0.9200 |
| MathAgent | 0.9190 |
| ReasonAgent | 0.9190 |
| MedAgent | 0.9155 |
| GeneralAgent | 0.9090 |
| NoisyAgent | 0.9060 |

## 七、分领域结果

| 领域 | 单智能体最优 | 全部 6 个投票 | 随机 3 个 | GoA Max | GoA Mean |
|---|---:|---:|---:|---:|---:|
| 通用 | 0.882 | **0.900** | 0.896 | 0.884 | 0.890 |
| 编程 | 0.900 | 0.904 | 0.900 | **0.916** | 0.910 |
| 医学 | 0.902 | 0.924 | 0.924 | 0.930 | **0.934** |
| 数学 | 0.994 | **0.998** | **0.998** | 0.996 | 0.996 |

## 八、结果解读

第一，完整高保真流程成功复现了论文的主要方向。GoA Mean 的总体准确率为 0.9325，是所有方法中最高的，比单智能体最优高 0.0130。

第二，GoA 相对于强基线的优势很小。GoA Mean 只比全部 6 个智能体投票高 0.0010，比随机 3 个智能体高 0.0030。因此，当前实验结果支持“GoA 有轻微收益”，但不足以支持“显著优于全部基线”。

第三，GoA 的收益具有领域差异。Max Pooling 在编程题上表现最好，Mean Pooling 在医学题上表现最好；通用和数学题仍由全部模型投票表现最好或持平。这说明图结构不是对所有任务都稳定增益。

第四，简单投票已经非常强。六个模型的初始准确率均在 0.906 到 0.920 之间，模型之间差异有限，因此后续消息传递能带来的额外信息较少。这也是各种聚合方法结果接近的重要原因。

第五，高保真消息传递不是简单覆盖答案或置信度，而是让模型读取邻居完整回答后重新生成。这一机制保留了论文中自然语言协商的核心特征，也使 GoA Mean 成为总体最优方法。

## 九、与原论文的差异

- 原论文使用 7B/8B 跨厂商模型池；本实验的六个模型均来自 DeepSeek 系列。
- 原论文使用 MMLU、MMLU-Pro、GPQA、MATH、HumanEval、MedMCQA；本实验使用 MMLU、C-Eval、MedQA、GSM8K 的平衡子集。
- 原论文的 MoA、Debate、SC 等基线未全部重新实现；本实验使用单智能体、全部六模型加权投票和随机三模型投票作为主要对照。
- GSM8K 被转换为四选一，干扰项由确定性数值规则生成。
- Meta-LLM 不是原论文使用的 Qwen2.5-7B-Instruct，而是 `deepseek-v4-pro`。

## 十、结论

本次复现完整实现了 Graph-of-Agents 的节点采样、边采样、双向自然语言消息传递和图池化，并在 2,000 道公开测试题上完成了近 30,000 次真实模型调用。

实验结果表明，高保真 GoA Mean Pooling 的总体准确率最高，但相对全部模型投票的优势仅为 0.0010。因此，本次复现可以支持论文方法的可行性和方向性，但不能在统计上证明其显著优于简单集成。最稳妥的论文结论应表述为：

> 在六个 DeepSeek 异构模型和四个公开测试集的平衡子集上，完整实现的 Graph-of-Agents Mean Pooling 取得最高总体准确率，但相对强基线只有较小提升。

## 十一、数据来源

- Hendrycks, D., et al. Measuring Massive Multitask Language Understanding. ICLR 2021. <https://huggingface.co/datasets/cais/mmlu>
- Huang, Y., et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. NeurIPS 2023. <https://huggingface.co/datasets/ceval/ceval-exam>
- Jin, D., et al. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. 2021. <https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options>
- Cobbe, K., et al. Training Verifiers to Solve Math Word Problems. 2021. <https://github.com/openai/grade-school-math>
