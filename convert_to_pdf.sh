#!/bin/bash
# 将 HTML 转换为 PDF 的脚本

HTML_FILE="AlphaZero核心概念笔记.html"
PDF_FILE="AlphaZero核心概念笔记.pdf"

echo "🔄 正在将 HTML 转换为 PDF..."

# 检查 HTML 文件是否存在
if [ ! -f "$HTML_FILE" ]; then
    echo "❌ 错误: 找不到文件 $HTML_FILE"
    exit 1
fi

# 方法1: 尝试使用 Chrome/Chromium (Headless)
if command -v google-chrome &> /dev/null; then
    echo "使用 Google Chrome..."
    google-chrome --headless --disable-gpu --print-to-pdf="$PDF_FILE" "$HTML_FILE"
    if [ -f "$PDF_FILE" ]; then
        echo "✅ PDF 生成成功: $PDF_FILE"
        exit 0
    fi
elif command -v chromium &> /dev/null; then
    echo "使用 Chromium..."
    chromium --headless --disable-gpu --print-to-pdf="$PDF_FILE" "$HTML_FILE"
    if [ -f "$PDF_FILE" ]; then
        echo "✅ PDF 生成成功: $PDF_FILE"
        exit 0
    fi
elif command -v "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" &> /dev/null; then
    echo "使用 Google Chrome (macOS)..."
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --print-to-pdf="$PDF_FILE" "file://$(pwd)/$HTML_FILE"
    if [ -f "$PDF_FILE" ]; then
        echo "✅ PDF 生成成功: $PDF_FILE"
        exit 0
    fi
fi

# 方法2: 使用 cupsfilter (macOS 自带)
if command -v cupsfilter &> /dev/null; then
    echo "尝试使用 cupsfilter..."
    cupsfilter "$HTML_FILE" > "$PDF_FILE" 2>/dev/null
    if [ -f "$PDF_FILE" ] && [ -s "$PDF_FILE" ]; then
        echo "✅ PDF 生成成功: $PDF_FILE"
        exit 0
    fi
fi

# 方法3: 使用 textutil 和 cupsfilter
if command -v textutil &> /dev/null && command -v cupsfilter &> /dev/null; then
    echo "尝试使用 textutil + cupsfilter..."
    TMP_RTF="temp_note.rtf"
    textutil -convert rtf -output "$TMP_RTF" "$HTML_FILE" 2>/dev/null
    if [ -f "$TMP_RTF" ]; then
        cupsfilter "$TMP_RTF" > "$PDF_FILE" 2>/dev/null
        rm -f "$TMP_RTF"
        if [ -f "$PDF_FILE" ] && [ -s "$PDF_FILE" ]; then
            echo "✅ PDF 生成成功: $PDF_FILE"
            exit 0
        fi
    fi
fi

# 如果所有方法都失败
echo ""
echo "⚠️  无法自动生成 PDF"
echo ""
echo "📝 但是，HTML 文件已成功生成！"
echo ""
echo "请使用以下方法之一手动转换为 PDF:"
echo ""
echo "方法1: 使用浏览器打印"
echo "  1. 打开文件: file://$(pwd)/$HTML_FILE"
echo "  2. 按 Cmd+P 打开打印对话框"
echo "  3. 选择'存储为 PDF'"
echo "  4. 保存为: $PDF_FILE"
echo ""
echo "方法2: 安装转换工具"
echo "  brew install wkhtmltopdf"
echo "  wkhtmltopdf $HTML_FILE $PDF_FILE"
echo ""
echo "方法3: 安装 Python 库"
echo "  pip install weasyprint"
echo "  python3 generate_pdf.py"
echo ""

exit 1

