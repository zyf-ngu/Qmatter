#!/usr/bin/env python3
import sys
import json


def main():
    # 从命令行参数获取文本内容（由Agent传入）
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No content provided"}))
        return

    content = sys.argv[1]
    lines = content.split('\n')
    words = content.split()

    result = {
        "line_count": len(lines),
        "word_count": len(words),
        "char_count": len(content)
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()