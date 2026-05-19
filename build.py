#!/usr/bin/env python3
"""
build.py — reads thesis/ directory, parses frontmatter, generates dashboard.html
Usage: python3 build.py
"""
import os
import re
import json
import glob

THESIS_DIR = os.path.join(os.path.dirname(__file__), "thesis")
OUTPUT = os.path.join(os.path.dirname(__file__), "dashboard.html")
TEMPLATE = os.path.join(os.path.dirname(__file__), "template.html")


def parse_frontmatter(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.startswith("---"):
        return None, content
    end = content.index("---", 3)
    fm_text = content[3:end].strip()
    body = content[end + 3:].strip()
    meta = {}
    for line in fm_text.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, body


def discover_stocks():
    stocks = {}
    for path in sorted(glob.glob(os.path.join(THESIS_DIR, "*-thesis.md"))):
        filename = os.path.basename(path)
        stock_id = filename.replace("-thesis.md", "")
        meta, body = parse_frontmatter(path)
        if not meta:
            continue
        stocks[stock_id] = {
            "id": stock_id,
            **meta,
            "files": {"thesis": body},
        }
    for path in glob.glob(os.path.join(THESIS_DIR, "*-monitor.md")):
        filename = os.path.basename(path)
        stock_id = filename.replace("-monitor.md", "")
        if stock_id in stocks:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.startswith("---"):
                _, content = parse_frontmatter(path)
            stocks[stock_id]["files"]["monitor"] = content
    for path in glob.glob(os.path.join(THESIS_DIR, "*-thesis-archive.md")):
        filename = os.path.basename(path)
        stock_id = filename.replace("-thesis-archive.md", "")
        if stock_id in stocks:
            with open(path, "r", encoding="utf-8") as f:
                stocks[stock_id]["files"]["archive"] = f.read()
    for path in glob.glob(os.path.join(THESIS_DIR, "*-model.yaml")):
        filename = os.path.basename(path)
        stock_id = filename.replace("-model.yaml", "")
        if stock_id in stocks:
            with open(path, "r", encoding="utf-8") as f:
                stocks[stock_id]["files"]["model"] = f.read()
    return list(stocks.values())


def build():
    stocks = discover_stocks()
    data_json = json.dumps(stocks, ensure_ascii=False)
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__STOCK_DATA__*/", f"const STOCKS = {data_json};", 1)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Built {OUTPUT}")
    print(f"  {len(stocks)} stocks, {sum(len(s['files']) for s in stocks)} files embedded")


if __name__ == "__main__":
    build()
