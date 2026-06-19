import pypdf


def parse_with_pypdf(file_path: str):
    """使用 pypdf 读取 PDF 元数据、文本和页面尺寸"""
    reader = pypdf.PdfReader(file_path)

    # 1. 文档基本信息
    print("PDF版本:", reader.stream.read(8).decode(errors='ignore'))
    print("文件尾 Trailer:", reader.trailer)

    # 2. 元数据
    if reader.metadata:
        print("标题:", reader.metadata.get("/Title"))
        print("作者:", reader.metadata.get("/Author"))
        print("主题:", reader.metadata.get("/Subject"))
        print("创建者:", reader.metadata.get("/Creator"))

    # 3. 逐页提取文本及页面尺寸
    for i, page in enumerate(reader.pages):
        print(f"\n--- 第 {i+1} 页 ---")
        # 页面尺寸 (MediaBox)
        mb = page.mediabox
        print(f"页面尺寸: {mb.width} x {mb.height}")
        # 提取文本
        text = page.extract_text()
        print(f"文本内容(前200字): {text[:200]}...")




import pdfplumber


def parse_with_pdfplumber(file_path: str):
    """使用 pdfplumber 提取文本、表格、页面尺寸及图像"""
    with pdfplumber.open(file_path) as pdf:
        # 1. 文档元数据
        print("元数据:", pdf.metadata)

        for i, page in enumerate(pdf.pages):
            print(f"\n--- 第 {i+1} 页 ---")
            # 2. 页面尺寸
            print(f"页面尺寸: {page.width} x {page.height}")

            # 3. 提取文本
            text = page.extract_text()
            print(f"文本内容(前200字): {text[:200]}...")

            # 4. 提取表格 (自动识别线条和文本对齐)
            tables = page.extract_tables()
            if tables:
                print(f"发现 {len(tables)} 个表格")
                for t_idx, table in enumerate(tables):
                    print(f"  表格 {t_idx+1}:")
                    for row in table[:3]:  # 只打印前3行
                        print("    ", row)

            # 5. 提取图像 (简单方式)
            if page.images:
                print(f"发现 {len(page.images)} 个图像对象")
                # 保存第一个图像为例
                img = page.images[0]
                # 裁剪图像区域并保存
                im = page.within_bbox(img["bbox"]).to_image()
                im.save(f"page_{i+1}_img_0.png", format="PNG")



import fitz  # PyMuPDF


def parse_with_pymupdf(file_path: str):
    """使用 PyMuPDF 提取文本、图像、页面渲染及目录"""
    doc = fitz.open(file_path)

    # 1. 文档基本信息
    print(f"页数: {doc.page_count}")
    print("元数据:", doc.metadata)

    # 2. 目录 (如果存在)
    toc = doc.get_toc()
    if toc:
        print("目录结构:")
        for item in toc:
            print(f"  层级{item[0]}: {item[1]} -> 第{item[2]}页")

    # 3. 逐页处理
    for page_num in range(doc.page_count):
        page = doc[page_num]
        print(f"\n--- 第 {page_num+1} 页 ---")

        # 页面尺寸
        rect = page.rect
        print(f"页面尺寸: {rect.width} x {rect.height}")

        # 提取文本 (纯文本)
        text = page.get_text("text")
        print(f"文本内容(前300字): {text[:300]}...")

        # 提取文本块 (带坐标)
        blocks = page.get_text("blocks")
        print(f"文本块数量: {len(blocks)}")

        # 提取图像
        image_list = page.get_images(full=True)
        if image_list:
            print(f"发现 {len(image_list)} 个图像")
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                ext = base_img["ext"]
                # 保存图像文件
                with open(f"page_{page_num+1}_img_{img_idx}.{ext}", "wb") as f:
                    f.write(img_bytes)
                print(f"  保存图像: page_{page_num+1}_img_{img_idx}.{ext}")

        # 将页面渲染为 PNG (仅第一页示例)
        if page_num == 0:
            pix = page.get_pixmap(dpi=150)
            pix.save("first_page.png")
            print("第一页已渲染为 first_page.png")

    doc.close()


if __name__ == "__main__":
    parse_with_pypdf("example.pdf")
    parse_with_pdfplumber("example.pdf")
    parse_with_pymupdf("example.pdf")