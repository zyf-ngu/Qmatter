import os

sample_docs_dir = os.path.dirname(__file__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm

    pdf_path = os.path.join(sample_docs_dir, "公司年报.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
    )

    story.append(Paragraph("2024年度公司报告", title_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>一、公司概况</b>", styles['Heading2']))
    story.append(Paragraph("公司成立于2010年，专注于人工智能技术研发与创新。", styles['BodyText']))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("<b>二、业务发展</b>", styles['Heading2']))
    story.append(Paragraph("• RAG系统产品线营收增长120%", styles['BodyText']))
    story.append(Paragraph("• 新增客户200+家", styles['BodyText']))
    story.append(Paragraph("• 研发团队扩展至50人", styles['BodyText']))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("<b>三、未来规划</b>", styles['Heading2']))
    story.append(Paragraph("继续深耕RAG技术，拓展更多应用场景，预计2025年实现营收翻倍。", styles['BodyText']))

    doc.build(story)
    print(f"✓ PDF文件创建成功: {pdf_path}")
except Exception as e:
    print(f"创建PDF文件失败: {e}")
