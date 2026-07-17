"""
Convert methodology_and_architecture.md to a styled PDF.
Uses markdown → HTML → PDF (via browser print).
"""
import os
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.codehilite import CodeHiliteExtension

INPUT = r"C:\Users\yunus\.gemini\antigravity\brain\ff7e888b-ba2a-425f-8fe8-54e154a7f255\methodology_and_architecture.md"
OUTPUT_HTML = os.path.join(os.path.dirname(INPUT), "methodology_and_architecture.html")
OUTPUT_DESKTOP = os.path.expanduser(r"~\OneDrive\Desktop\methodology_and_architecture.html")

with open(INPUT, "r", encoding="utf-8") as f:
    md_text = f.read()

# Strip mermaid blocks (they won't render in static HTML) — replace with placeholder
import re
def replace_mermaid(match):
    content = match.group(1).strip()
    # Extract a summary from the mermaid content
    lines = content.split('\n')
    diagram_type = lines[0] if lines else "diagram"
    return f'<div style="background:#1e293b;color:#94a3b8;padding:20px;border-radius:8px;margin:16px 0;text-align:center;font-style:italic;border:1px dashed #475569;">[Mermaid {diagram_type} — renders in Markdown viewers, GitHub, VS Code]</div>'

md_text = re.sub(r'```mermaid\n(.*?)```', replace_mermaid, md_text, flags=re.DOTALL)

# Convert MD → HTML
html_body = markdown.markdown(
    md_text,
    extensions=[
        'tables',
        'fenced_code',
        'codehilite',
        'nl2br',
    ],
    extension_configs={
        'codehilite': {
            'css_class': 'highlight',
            'guess_lang': False,
        }
    }
)

# Full styled HTML document
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI + Quantum Material Discovery — Methodology & Architecture</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
    line-height: 1.7;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 50px;
    background: #ffffff;
    font-size: 13px;
}}

h1 {{
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
    margin: 30px 0 8px;
    padding-bottom: 8px;
    border-bottom: 3px solid #3b82f6;
}}

h2 {{
    font-size: 20px;
    font-weight: 700;
    color: #1e40af;
    margin: 28px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #dbeafe;
}}

h3 {{
    font-size: 16px;
    font-weight: 600;
    color: #1e3a5f;
    margin: 20px 0 8px;
}}

h4 {{
    font-size: 14px;
    font-weight: 600;
    color: #334155;
    margin: 16px 0 8px;
}}

p {{
    margin: 8px 0;
    text-align: justify;
}}

strong {{
    color: #0f172a;
}}

hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 24px 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 12px;
}}

th {{
    background: #1e40af;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

td {{
    padding: 7px 12px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}}

tr:nth-child(even) td {{
    background: #f8fafc;
}}

tr:hover td {{
    background: #eff6ff;
}}

code {{
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 11.5px;
    color: #7c3aed;
}}

pre {{
    background: #1e293b;
    color: #e2e8f0;
    padding: 16px 20px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 12px 0;
    font-size: 12px;
    line-height: 1.5;
}}

pre code {{
    background: none;
    color: #e2e8f0;
    padding: 0;
}}

blockquote {{
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 12px;
}}

blockquote strong {{
    color: #1e40af;
}}

ul, ol {{
    margin: 8px 0;
    padding-left: 24px;
}}

li {{
    margin: 4px 0;
}}

/* Page break hints for print */
h2 {{
    page-break-before: auto;
}}

@media print {{
    body {{
        padding: 20px 30px;
        font-size: 11px;
    }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 17px; page-break-after: avoid; }}
    h3 {{ font-size: 14px; page-break-after: avoid; }}
    table {{ page-break-inside: avoid; font-size: 10px; }}
    pre {{ page-break-inside: avoid; }}
    blockquote {{ page-break-inside: avoid; }}
}}

/* Cover-like heading */
h1:first-of-type {{
    text-align: center;
    font-size: 32px;
    margin-top: 60px;
    margin-bottom: 4px;
}}

h2:first-of-type {{
    text-align: center;
    border-bottom: none;
    color: #64748b;
    font-weight: 400;
    font-size: 16px;
    margin-bottom: 40px;
}}
</style>
</head>
<body>
{html_body}
<footer style="margin-top:40px;padding-top:16px;border-top:1px solid #e2e8f0;text-align:center;color:#94a3b8;font-size:11px;">
AI + Quantum Material Discovery Platform &mdash; Project Documentation &mdash; February 2026
</footer>
</body>
</html>
"""

# Save HTML
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

# Also copy to Desktop for easy access
with open(OUTPUT_DESKTOP, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML saved to: {OUTPUT_HTML}")
print(f"Desktop copy:  {OUTPUT_DESKTOP}")
print(f"\nTo convert to PDF:")
print(f"  1. Open the HTML file on your Desktop")
print(f"  2. Press Ctrl+P → Save as PDF")
print(f"  (The HTML is already styled for print)")
