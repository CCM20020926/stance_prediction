# 针对开源数据集的认知立场检测实验

## 一、MITweet 多维意识形态检测数据集

（代码目录：[MITweet/](MITweet/)，数据集链接：[https://github.com/LST1836/MITweet](https://github.com/LST1836/MITweet)）

### （一）数据集说明

**检测维度：** 12 （详见[MITweet/detection_dim.md](MITweet/detection_dim.md)）

**标签种类数：** 4 （预处理后种类标号：  0 -  未涉及 1 - 左派 2 - 中立 3 - 右派）

### （二）检测方案与效果

#### 1. BERT 微调模型

- 训练回合数: 10

- 2 种训练方案：全参微调、增量微调

- 评估策略：每训练完一回合后在验证集上评估一次

- 实验效果：

  - **（1）全量微调方案:**

    （运行命令： cd MITweet && python finetune_bert.py）

    **指标效果（mACC - 各个维度的检测准确率平均值, mMICRO-F1 - 各个维度的检测 MICRO-F1 分数平均值， mMACRO-F1 各个维度的 MACRO-F1 分数平均值）**

    | 训练回合数 | mACC(%) | mMICRO-F1(%) | mMACRO-F1(%) |
    | ---------- | ------- | ------------ | ------------ |
    | 4          | 92.0    | 92.0         | 38.6         |
    | 10         | 91.5    | 91.5         | 39.9         |

    **时空开销：** 总共训练时长大概15分钟，显存消耗 4-5 GB

  - **（2）增量微调方案：**

    （运行命令：cd MITweet && python finetune_bert.py  --frozen_base=True ）

    **指标效果**

    | 训练回合数 | mACC(%) | mMICRO-F1(%) | mMACRO-F1(%) |
    | ---------- | ------- | ------------ | ------------ |
    | 10         | 88.7    | 88.7         | 23.4         |

    **时空开销：** 总共训练时长大概5分钟，显存消耗 <1 GB

#### 2. LLM Zero-Shot 检测方案

运行命令：cd MITweet && python llm_agent_test.py

需要配置环境变量：OPENAI_API_KEY, OPENAI_BASE_URL

**测试效果**

| 模型        | mACC(%) | mMICRO-F1(%) | mMACRO-F1(%) |
| ----------- | ------- | ------------ | ------------ |
| GPT-5       | 83.7    | 83.7         | 36.8         |
| DeepSeek-V3 | 85.5    | 85.5         | 34.1         |
| DeepSeek-R1 | 80.8    | 80.8         | 33.8         |
