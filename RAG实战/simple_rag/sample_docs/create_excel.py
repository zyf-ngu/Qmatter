import os
import pandas as pd

sample_docs_dir = os.path.dirname(__file__)

try:
    data = {
        "产品名称": ["智能问答系统", "文档处理平台", "数据分析工具", "图像识别引擎"],
        "版本": ["v2.1", "v1.5", "v3.0", "v1.2"],
        "发布日期": ["2024-01", "2024-03", "2024-02", "2024-04"],
        "用户数": [1200, 850, 620, 380]
    }
    df = pd.DataFrame(data)
    output_path = os.path.join(sample_docs_dir, "产品列表.xlsx")
    df.to_excel(output_path, index=False, sheet_name="产品信息")
    print(f"✓ Excel文件创建成功: {output_path}")
except Exception as e:
    print(f"创建Excel文件失败: {e}")
