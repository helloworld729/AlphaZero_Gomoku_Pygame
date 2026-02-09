#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将笔记.txt转换为PDF的脚本
使用标准库 + weasyprint (如果可用) 或 HTML 中间文件
"""

import os
import sys

def markdown_to_html(content):
    """简单的 Markdown 到 HTML 转换"""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AlphaZero 五子棋项目核心概念笔记</title>
    <style>
        body {
            font-family: "PingFang SC", "Microsoft YaHei", "SimHei", Arial, sans-serif;
            line-height: 1.8;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background-color: #ffffff;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 28px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 35px;
            font-size: 24px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
            font-size: 20px;
        }
        h4 {
            color: #95a5a6;
            margin-top: 20px;
            font-size: 18px;
        }
        pre {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 14px;
            line-height: 1.6;
        }
        code {
            background-color: #f1f3f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
            font-size: 13px;
            color: #e74c3c;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
        }
        ul, ol {
            margin-left: 20px;
            margin-top: 10px;
        }
        li {
            margin-bottom: 8px;
        }
        strong {
            color: #e74c3c;
            font-weight: bold;
        }
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }
        .page-break {
            page-break-after: always;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }
        @media print {
            body {
                margin: 0;
                padding: 20px;
            }
            h1, h2, h3 {
                page-break-after: avoid;
            }
            pre, blockquote {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
"""

    lines = content.split('\n')
    in_code_block = False
    code_buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # 代码块处理
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_buffer = []
            else:
                # 结束代码块
                html += '<pre><code>'
                html += '\n'.join(code_buffer)
                html += '</code></pre>\n'
                in_code_block = False
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue

        # 标题处理
        if line.startswith('# '):
            html += f'<h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            html += f'<h2>{line[3:]}</h2>\n'
        elif line.startswith('### '):
            html += f'<h3>{line[4:]}</h3>\n'
        elif line.startswith('#### '):
            html += f'<h4>{line[5:]}</h4>\n'
        elif line.strip() == '---':
            html += '<hr>\n'
        elif line.startswith('- '):
            # 列表项
            html += '<ul>\n'
            while i < len(lines) and lines[i].startswith('- '):
                item = lines[i][2:]
                # 处理行内代码
                item = process_inline_code(item)
                # 处理加粗
                item = process_bold(item)
                html += f'<li>{item}</li>\n'
                i += 1
            html += '</ul>\n'
            continue
        elif line.strip() == '':
            html += '<br>\n'
        else:
            # 普通段落
            processed_line = line
            # 处理行内代码
            processed_line = process_inline_code(processed_line)
            # 处理加粗
            processed_line = process_bold(processed_line)
            html += f'<p>{processed_line}</p>\n'

        i += 1

    html += """
</body>
</html>
"""
    return html

def process_inline_code(text):
    """处理行内代码"""
    import re
    # 替换 `code` 为 <code>code</code>
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

def process_bold(text):
    """处理加粗文本"""
    import re
    # 替换 **text** 为 <strong>text</strong>
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

def main():
    # 读取笔记文件
    note_file = '笔记.txt'
    if not os.path.exists(note_file):
        print(f"错误: 找不到文件 {note_file}")
        return 1

    with open(note_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 转换为 HTML
    html_content = markdown_to_html(content)

    # 保存 HTML 文件
    html_file = 'AlphaZero核心概念笔记.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML 文件已生成: {html_file}")

    # 尝试使用 weasyprint 生成 PDF
    try:
        from weasyprint import HTML
        pdf_file = 'AlphaZero核心概念笔记.pdf'
        HTML(html_file).write_pdf(pdf_file)
        print(f"✅ PDF 文件已生成: {pdf_file}")
        return 0
    except ImportError:
        print("\n⚠️  weasyprint 未安装，仅生成了 HTML 文件")
        print("\n您可以:")
        print("1. 安装 weasyprint: pip install weasyprint")
        print("2. 或者在浏览器中打开 HTML 文件，然后使用浏览器的打印功能保存为 PDF")
        print(f"   文件路径: {os.path.abspath(html_file)}")
        return 0
    except Exception as e:
        print(f"\n⚠️  PDF 生成失败: {e}")
        print(f"但 HTML 文件已成功生成: {html_file}")
        print("您可以在浏览器中打开 HTML 文件，然后使用打印功能保存为 PDF")
        return 0

if __name__ == '__main__':
    sys.exit(main())

