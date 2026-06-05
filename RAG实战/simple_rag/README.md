# 简单 RAG 系统演示

## 使用方法

### 1. 准备文档
在 `sample_docs` 目录下放入您的文档，支持以下格式：
- PDF (.pdf)
- Word (.docx, .doc)
- Excel (.xlsx, .xls)
- 文本 (.txt, .md)

### 2. 生成示例文档（可选）
如果需要生成测试用的示例文档，可以运行：
```bash
# 生成Word文档
cd sample_docs
python create_docx.py

# 生成Excel文档
python create_excel.py

# 生成PDF文档
python create_pdf.py
```

### 3. 运行演示
```bash
cd e:\Qagent\simple_rag
python main.py
```

### 4. 自定义使用
```python
from main import SimpleRAG

# 初始化
rag = SimpleRAG(
    embed_model_path=r"E:\Qagent\models\allMiniLML6v2",
    chunk_size=200,
    overlap=30
)

# 从目录构建索引
rag.build_index_from_directory("sample_docs")

# 查询
result = rag.query("您的问题")
print(result["answer"])
```

## 目录结构
```
simple_rag/
├── main.py                 # 主程序入口
├── sample_docs/            # 示例文档目录
│   ├── 公司介绍.txt
│   ├── 产品手册.md
│   ├── create_docx.py      # Word生成脚本
│   ├── create_excel.py     # Excel生成脚本
│   └── create_pdf.py       # PDF生成脚本
├── parser/                 # 文档解析包
├── knowledge_base/         # 向量索引
├── spliter/                # 文本分块
└── retriver/               # 检索器
```
