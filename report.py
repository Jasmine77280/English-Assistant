"""学习报告生成模块：支持导出 Markdown 和 PDF。"""
import os
from typing import Dict

# 尝试导入 fpdf2
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# 系统中文字体路径（优先 .ttf，fpdf2 对 .ttc 支持有限）
_CN_FONT_CANDIDATES = [
    # Windows
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    # Linux（部署时安装 fonts-wqy-zenhei 或 fonts-noto-cjk）
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def _find_cn_font() -> str:
    for p in _CN_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return ""


def generate_markdown(record: Dict) -> str:
    """生成 Markdown 格式学习报告。"""
    lines = []
    lines.append(f"# 英入眼帘 · 学习报告")
    lines.append("")
    lines.append(f"**生成时间**：{record.get('created_at', '')}")
    lines.append(f"**模式**：{'Demo 演示' if record.get('demo_mode') else 'AI 生成'}")
    lines.append("")
    lines.append("## 目标词汇")
    lines.append("")
    lines.append(f"- 目标单词（{len(record.get('words', []))} 个）：{', '.join(record.get('words', []))}")
    lines.append(f"- Vocabulary Coverage：**{record.get('coverage', 0)}%**")
    lines.append(f"- 已覆盖：{', '.join(record.get('matched_words', [])) or '无'}")
    lines.append(f"- 未覆盖：{', '.join(record.get('missing_words', [])) or '无'}")
    lines.append("")

    articles = record.get("articles", [])
    translations = record.get("translations", [])
    for i, art in enumerate(articles, 1):
        lines.append(f"## Article {i}")
        lines.append("")
        lines.append(art)
        lines.append("")
        if i - 1 < len(translations):
            lines.append(f"**中文翻译**：{translations[i - 1]}")
            lines.append("")

    questions = record.get("questions", [])
    if questions:
        lines.append("## 阅读理解")
        lines.append("")
        for i, q in enumerate(questions, 1):
            lines.append(f"### {i}. {q.get('question', '')}")
            lines.append("")
            for j, opt in enumerate(q.get("options", [])):
                mark = " ✅" if j == q.get("answer") else ""
                lines.append(f"- {chr(65 + j)}. {opt}{mark}")
            lines.append("")
            lines.append(f"**解析**：{q.get('explanation', '')}")
            lines.append("")

    lines.append("---")
    lines.append("*由 英入眼帘 (AI English Learning Agent) 生成*")
    return "\n".join(lines)


def generate_pdf(record: Dict, output_path: str) -> bool:
    """生成 PDF 格式学习报告。使用 fpdf2 + 中文字体。"""
    if not HAS_FPDF:
        return False
    font_path = _find_cn_font()
    if not font_path:
        return False

    try:
        pdf = FPDF()
        pdf.add_page()
        # 注册中文字体
        pdf.add_font("CN", "", font_path)

        def _mc(w, h, txt):
            """multi_cell 后重置 x 到左边距，避免 fpdf2 连续 multi_cell 宽度不足。"""
            pdf.multi_cell(w, h, txt)
            pdf.set_x(pdf.l_margin)

        pdf.set_font("CN", size=16)
        pdf.cell(0, 12, "English Insight - Learning Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("CN", size=10)
        pdf.cell(0, 8, f"Generated: {record.get('created_at', '')}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        pdf.set_font("CN", size=12)
        words = record.get("words", [])
        _mc(0, 8, f"Target Words ({len(words)}): {', '.join(words)}")
        _mc(0, 8, f"Coverage: {record.get('coverage', 0)}%")
        _mc(0, 8, f"Matched: {', '.join(record.get('matched_words', [])) or 'None'}")
        _mc(0, 8, f"Missing: {', '.join(record.get('missing_words', [])) or 'None'}")
        pdf.ln(4)

        articles = record.get("articles", [])
        translations = record.get("translations", [])
        for i, art in enumerate(articles, 1):
            pdf.set_font("CN", size=13)
            pdf.cell(0, 10, f"Article {i}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("CN", size=11)
            _mc(0, 7, art)
            if i - 1 < len(translations):
                pdf.set_font("CN", size=10)
                pdf.set_text_color(100, 100, 100)
                _mc(0, 7, translations[i - 1])
                pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

        questions = record.get("questions", [])
        if questions:
            pdf.set_font("CN", size=13)
            pdf.cell(0, 10, "Reading Comprehension", new_x="LMARGIN", new_y="NEXT")
            for i, q in enumerate(questions, 1):
                pdf.set_font("CN", size=11)
                _mc(0, 7, f"{i}. {q.get('question', '')}")
                pdf.set_font("CN", size=10)
                for j, opt in enumerate(q.get("options", [])):
                    mark = " <-- Correct" if j == q.get("answer") else ""
                    _mc(0, 6, f"   {chr(65 + j)}. {opt}{mark}")
                pdf.set_text_color(80, 80, 80)
                _mc(0, 6, f"   Explanation: {q.get('explanation', '')}")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)

        pdf.output(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("PDF generation failed: %s", e)
        return False


def generate_report_html(record: Dict) -> str:
    """生成可打印的 HTML 报告（浏览器可另存为 PDF）。"""
    articles = record.get("articles", [])
    translations = record.get("translations", [])
    questions = record.get("questions", [])
    words = record.get("words", [])

    articles_html = ""
    for i, art in enumerate(articles, 1):
        cn = translations[i - 1] if i - 1 < len(translations) else ""
        articles_html += f"""
        <h3>Article {i}</h3>
        <p class="en">{art}</p>
        <p class="cn">{cn}</p>
        """

    q_html = ""
    for i, q in enumerate(questions, 1):
        opts = ""
        for j, opt in enumerate(q.get("options", [])):
            mark = " ✅" if j == q.get("answer") else ""
            opts += f"<li>{chr(65 + j)}. {opt}{mark}</li>"
        q_html += f"""
        <div class="q">
          <p><b>{i}.</b> {q.get('question', '')}</p>
          <ul>{opts}</ul>
          <p class="explain"><b>解析：</b>{q.get('explanation', '')}</p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Learning Report</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1e293b; }}
  h1 {{ text-align: center; color: #4f46e5; }}
  .meta {{ text-align: center; color: #64748b; margin-bottom: 30px; }}
  h3 {{ color: #4f46e5; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
  .en {{ font-size: 15px; line-height: 1.8; }}
  .cn {{ color: #64748b; font-size: 14px; }}
  .q {{ margin-bottom: 20px; padding: 12px; background: #f8fafc; border-radius: 8px; }}
  .explain {{ color: #64748b; font-size: 13px; }}
  ul {{ list-style: none; padding-left: 20px; }}
  @media print {{ body {{ margin: 0; }} }}
</style></head>
<body>
  <h1>English Insight · Learning Report</h1>
  <div class="meta">Generated: {record.get('created_at', '')} | Mode: {'Demo' if record.get('demo_mode') else 'AI'}</div>
  <p><b>Target Words ({len(words)}):</b> {', '.join(words)}</p>
  <p><b>Coverage:</b> {record.get('coverage', 0)}% | <b>Matched:</b> {', '.join(record.get('matched_words', [])) or 'None'} | <b>Missing:</b> {', '.join(record.get('missing_words', [])) or 'None'}</p>
  {articles_html}
  <h3>Reading Comprehension</h3>
  {q_html}
</body></html>"""
