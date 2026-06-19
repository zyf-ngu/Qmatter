# Qmatter - 跟着问题学——AI大模型入门系列（笔记&代码）

视频链接  

B站：不如语冰  

https://space.bilibili.com/70431433?spm_id_from=333.1007.0.0。  

小红书：不如语冰（AI大模型版）26183735367


https://www.xiaohongshu.com/user/profile/6846bd9c000000001e03dbc5?m_source=mingfenghuohu



## 系列定位
本系列以"跟着问题学"为核心，通过「笔记 PDF + 可运行 Python 代码」的形式，为 AI 大模型初学者提供清晰的学习路径。内容从 Python 基础语言起步，经过 PyTorch、经典 CV/NLP 模型、Transformer 原理，最终落地到 RAG 与 Agent 等实战项目。学完即可具备大模型入门所需的理论认知与工程实践能力。


## 项目目录结构
仓库按学习模块组织，每个模块通常包含 `docs/`（笔记 PDF）、`models/`（模型定义）、`train/`（训练/运行脚本）三类内容，方便对照学习。

```
Qmatter/
├── python基础/                        # Python 核心语法入门
│   ├── Python基础v2.0.pdf
│   ├── learn_class.py                 # 类与面向对象
│   ├── learn_data_structure.py        # 列表/字典/元组等数据结构
│   ├── learn_function.py              # 函数定义与调用
│   └── 数据结构/
│
├── python进阶/                        # 适配大型项目的 Python 工程能力
│   ├── Python进阶v3.1.pdf
│   ├── config.py
│   ├── learn_argparse.py              # 命令行参数
│   ├── learn_common_function.py
│   ├── learn_contextmanager.py        # with 语句 / 上下文管理器
│   ├── learn_fancy_function.py        # 装饰器等高级函数
│   ├── learn_file_path.py             # 文件与路径管理
│   ├── learn_iterable.py              # 迭代器 / 生成器
│   ├── learn_log.py                   # 日志模块
│   ├── learn_tips.py
│   └── learn_try.py                   # 异常处理
│
├── 多进程多线程异步编程/              # 并发编程核心
│   ├── 异步编程和多进程多线程V3.0docx.pdf
│   ├── learn_asyncio.py               # asyncio 异步编程
│   ├── learn_multi_process.py         # 多进程
│   └── learn_multi_thread.py          # 多线程
│
├── pytorch基础/                       # PyTorch 框架入门
│   ├── pytorch核心笔记-张量v2.pdf
│   ├── learn_tensor.py                # 张量创建与运算
│   └── learn_tensor2.py               # 张量变换/常用 API
│
├── CV/                                # 计算机视觉经典模型
│   ├── docs/                          # "跟着问题学"系列笔记（LeNet/ResNet/VGG 等）
│   ├── models/                        # 模型实现（LeNet, AlexNet, VGG, GoogLenet, ResNet）
│   └── train/                         # 训练脚本（以 ResNet 为代表，含完整训练循环）
│
├── NLP经典模型/                       # 自然语言处理经典模型
│   ├── docs/                          # "跟着问题学"系列笔记（词向量/Seq2Seq/Attention 等）
│   ├── models/                        # 模型实现（Word2Vec, RNN, LSTM, Seq2Seq, Seq2Seq-Attention）
│   └── train/                         # 训练脚本（Word2Vec, RNN, LSTM, GRU, Seq2Seq）
│
├── transformer大模型/                 # Transformer 原理与大模型
│   ├── 跟着问题学18——transformer详解及代码实战.pdf
│   ├── 大模型微调和训练v2.0.pdf
│   ├── 大模型部署应用实例v2.pdf
│   └── transformers.py                # Transformer 核心代码
│
├── RAG实践/                           # RAG（检索增强生成）实战
│   ├── 1.RAG概述v2.0.pdf ~ 6.RAG-向量化与向量知识库v4.0.pdf
│   ├── fastapi v2.0.pdf
│   ├── streamlit v2.0.pdf
│   ├── simple_rag/                    # 最简 RAG 实现（含 parser / spliter / retriver / knowledge_base）
│   │   ├── main.py                    # SimpleRAG 入口，可直接运行演示
│   │   ├── sample_docs/               # 示例文档（PDF/Word/Excel/Markdown/TXT）
│   │   ├── parser/                    # PDF / DOCX / Excel / TXT 解析器
│   │   ├── spliter/                   # 文本分块（递归/字符/语义分块）
│   │   ├── retriver/                  # BM25 / 向量 / 混合检索 + Reranker
│   │   ├── knowledge_base/            # FAISS 向量索引（EmbeddingModel + VectorStore）
│   │   └── README.md
│   └── Qrag/                          # 增强版 RAG（在 simple_rag 基础上扩展）
│       ├── main.py                    # RAGPipeline 入口，内置混合检索 + 重排
│       ├── parser/                    # 支持 PDF（pypdf/pdfplumber/PyMuPDF）/ DOCX / Excel
│       ├── spliter/                   # 文本分块
│       ├── retriver/                  # BM25Retriever / VectorRetriever / HybridRetriever / Reranker
│       └── knowledge_base/            # FAISS 向量索引
│
└── Agent实践/                          # Agent 实践
    ├── 1 agent概述v2.0.pdf
    └── 2 agent基本组件简介v2.0.pdf
```


## 学习框架
当前分为 **八大模块**（后续持续更新），按从基础到实战的顺序组织：

### 1. Python 基础
[python基础/](file:///d:/Qmatter/python基础/) —— AI 大模型开发的核心语言。从"0.5 基础"起步，侧重实战应用：
- 数据结构、函数定义与调用、类与面向对象编程
- 配套代码：[learn_data_structure.py](file:///d:/Qmatter/python基础/learn_data_structure.py)、[learn_function.py](file:///d:/Qmatter/python基础/learn_function.py)、[learn_class.py](file:///d:/Qmatter/python基础/learn_class.py)

### 2. Python 进阶
[python进阶/](file:///d:/Qmatter/python进阶/) —— 适配大型工程的 Python 能力：
- 日志系统、参数传递、文件路径管理、上下文管理器、装饰器、迭代器、异常处理等
- 配套代码：[learn_log.py](file:///d:/Qmatter/python进阶/learn_log.py)、[learn_argparse.py](file:///d:/Qmatter/python进阶/learn_argparse.py)、[learn_contextmanager.py](file:///d:/Qmatter/python进阶/learn_contextmanager.py) 等

### 3. 多进程 / 多线程 / 异步编程
[多进程多线程异步编程/](file:///d:/Qmatter/多进程多线程异步编程/) —— 并发编程与高性能数据处理基础：
- 配套代码：[learn_multi_process.py](file:///d:/Qmatter/多进程多线程异步编程/learn_multi_process.py)、[learn_multi_thread.py](file:///d:/Qmatter/多进程多线程异步编程/learn_multi_thread.py)、[learn_asyncio.py](file:///d:/Qmatter/多进程多线程异步编程/learn_asyncio.py)

### 4. PyTorch 框架学习
[pytorch基础/](file:///d:/Qmatter/pytorch基础/) —— AI 模型开发核心框架，聚焦"张量"这一核心概念：
- 张量的创建、运算与变换；模型输入输出的张量维度变化逻辑
- 常用 API 实战（如 `torch.nn` 模块）
- 配套代码：[learn_tensor.py](file:///d:/Qmatter/pytorch基础/learn_tensor.py)、[learn_tensor2.py](file:///d:/Qmatter/pytorch基础/learn_tensor2.py)

### 5. 计算机视觉（CV）经典模型
[CV/](file:///d:/Qmatter/CV/) —— 建立 CV 基础认知，从 LeNet 到 ResNet：
- 经典模型：LeNet、AlexNet、VGG、GoogLeNet、ResNet（含 v1 / v2 / v3 变体）
- 配套知识：数据处理、损失函数、优化算法、反向传播原理
- 模型定义：[models/ResNet.py](file:///d:/Qmatter/CV/models/ResNet.py) 等
- 训练脚本：[train/ResNet.py](file:///d:/Qmatter/CV/train/ResNet.py)（含完整训练循环、学习率调度、最佳模型保存）

### 6. NLP 经典模型
[NLP经典模型/](file:///d:/Qmatter/NLP经典模型/) —— Transformer 出现前的核心 NLP 模型：
- 词向量、Word2Vec、RNN、LSTM、GRU、Seq2Seq、Seq2Seq + Attention
- 模型定义：[models/LSTM.py](file:///d:/Qmatter/NLP经典模型/models/LSTM.py)、[models/seq2seq.py](file:///d:/Qmatter/NLP经典模型/models/seq2seq.py)、[models/word2vec.py](file:///d:/Qmatter/NLP经典模型/models/word2vec.py) 等
- 训练脚本：[train/GRU.py](file:///d:/Qmatter/NLP经典模型/train/GRU.py)、[train/seq2seq.py](file:///d:/Qmatter/NLP经典模型/train/seq2seq.py) 等

### 7. Transformer 大模型
[transformer大模型/](file:///d:/Qmatter/transformer大模型/) —— 大模型时代的核心架构：
- Transformer 原理与代码实现（[transformers.py](file:///d:/Qmatter/transformer大模型/transformers.py)）
- 大模型部署与微调（在线 API 调用 + 本地部署，调参技巧）

### 8. RAG 与 Agent 实战
[RAG实践/](file:///d:/Qmatter/RAG实践/) / [Agent实践/](file:///d:/Qmatter/Agent实践/) —— 将所学综合落地到真实项目：
- **simple_rag**：最简可运行 RAG 系统，包含文档解析、文本分块、向量索引、混合检索（BM25 + 向量）、Reranker 重排；直接运行 [simple_rag/main.py](file:///d:/Qmatter/RAG实践/simple_rag/main.py) 即可体验
- **Qrag**：增强版 RAG，在 simple_rag 基础上扩展解析能力（PDF 支持 pypdf / pdfplumber / PyMuPDF 三种方案）；见 [Qrag/main.py](file:///d:/Qmatter/RAG实践/Qrag/main.py) 与 [Qrag/parser/pdf_parser.py](file:///d:/Qmatter/RAG实践/Qrag/parser/pdf_parser.py)
- **Agent**：Agent 概述与核心组件简介，为后续 LLM Agent 开发打基础


## 学习思路
遵循以下 5 点原则，提升学习效率：

1. **跟着问题学**  
   聚焦学习中的真实问题（如"Transformer 为什么要用自注意力？""文档为什么要做分块？"），从问题拆解知识点，避免"知识诅咒"。欢迎在 Issues / 评论区提问题，共同完善内容。

2. **多动手敲代码**  
   仓库提供配套代码，建议逐行手动编写（勿复制粘贴），在实操中暴露并解决问题，才能真正掌握。

3. **温故而知新的重复**  
   对核心知识点（如 Transformer、张量运算、RAG 的检索流程）反复回顾，用"苏轼八面受敌读书法"逐步加深理解。

4. **定期做知识总结**  
   将零散知识点梳理成系统结构（如思维导图、笔记 PDF），便于记忆、应用与扩展。

5. **多输出检验成果**  
   输出是最好的学习：向他人讲解知识点、写补充笔记、提交代码 PR，梳理知识的同时深化理解。


## 关于本系列的想法
1. **相信 AI 的价值**  
   AI 将成为人类的强大助手：提升信息处理效率、替代重复事务，让人类聚焦创造性工作。

2. **助力后来者，共同学习**  
   本系列不仅是分享，更是互助——你的问题与反馈，会帮助完善内容，形成良性循环。

3. **交流 AI 未来发展**  
   创造力源于思维碰撞，欢迎在评论区 / 仓库交流大模型应用场景与未来方向。


## 结语
本系列致力于创造"有价值、有意义"的学习内容，如有建议（补充知识点、优化代码等），欢迎通过以下方式反馈：
- 视频评论区留言  
- GitHub 仓库提交 Issues 或 PR  

一起开启 AI 大模型学习之旅！
