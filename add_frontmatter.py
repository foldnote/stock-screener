#!/usr/bin/env python3
"""One-time script: prepend YAML frontmatter to thesis files."""
import os, re
from datetime import datetime

THESIS_DIR = os.path.expanduser("~/Projects/stock-screener/thesis")

STOCKS = {
    "popmart-09992":          {"name": "泡泡玛特 Pop Mart",       "ticker": "09992.HK", "market": "港股", "sector": "IP/潮流消费",   "confidence": "core",    "verdict": "核心跟踪，最深的 thesis"},
    "sanrio-8136":            {"name": "三丽鸥 Sanrio",           "ticker": "8136.T",   "market": "日本", "sector": "IP/潮流消费",   "confidence": "bullish", "verdict": "全球最纯 IP 授权平台，确定性最高"},
    "duolingo-DUOL":          {"name": "Duolingo",               "ticker": "DUOL",     "market": "美股", "sector": "SaaS/平台",     "confidence": "core",    "verdict": "可以建仓，不宜重仓"},
    "berkshire-hathaway-BRK.B":{"name": "Berkshire Hathaway",    "ticker": "BRK.B",    "market": "美股", "sector": "综合",          "confidence": "core",    "verdict": "合理可入，$400-430 更好"},
    "moutai-600519":          {"name": "贵州茅台",                "ticker": "600519.SS","market": "A股",  "sector": "消费",          "confidence": "bullish", "verdict": "PE 历史底部，适合分批建仓"},
    "伊利-600887":             {"name": "伊利股份",                "ticker": "600887.SS","market": "A股",  "sector": "消费",          "confidence": "bullish", "verdict": "逐步建仓可以，¥20-22 更好"},
    "wuxi-apptec-603259":     {"name": "药明康德",                "ticker": "603259.SS","market": "A股",  "sector": "医药",          "confidence": "bullish", "verdict": "绝对值得持有，BIOSECURE 12月前不宜重仓"},
    "servicenow-NOW":         {"name": "ServiceNow",             "ticker": "NOW",      "market": "美股", "sector": "SaaS/平台",     "confidence": "bullish", "verdict": "SaaS 杀估值创造买入窗口"},
    "mcdonalds-MCD":          {"name": "麦当劳 McDonald's",       "ticker": "MCD",      "market": "美股", "sector": "消费",          "confidence": "bullish", "verdict": "核心资产，等 $240-260"},
    "luckin-LKNCY":           {"name": "瑞幸咖啡",                "ticker": "LKNCY",    "market": "美股", "sector": "消费",          "confidence": "bullish", "verdict": "规模+高增长+零负债"},
    "starbucks-SBUX":         {"name": "星巴克 Starbucks",        "ticker": "SBUX",     "market": "美股", "sector": "消费",          "confidence": "bullish", "verdict": "好生意+对的人，等利润率恢复验证"},
    "lvmh-MC":                {"name": "LVMH",                   "ticker": "MC.PA",    "market": "欧洲", "sector": "消费",          "confidence": "bullish", "verdict": "全球最优质消费品公司，等 €380-400"},
    "tripcom-TCOM":           {"name": "携程 Trip.com",           "ticker": "TCOM",     "market": "美股", "sector": "消费服务",      "confidence": "bullish", "verdict": "反垄断罚款落地后可加仓"},
    "nongfuspring-09633":     {"name": "农夫山泉",                "ticker": "09633.HK", "market": "港股", "sector": "消费",          "confidence": "wait",    "verdict": "一线品质，等 HK$35-40"},
    "mixue-02097":            {"name": "蜜雪冰城",                "ticker": "02097.HK", "market": "港股", "sector": "消费",          "confidence": "wait",    "verdict": "好生意但增速放缓，等 HKD 160-200"},
    "anta-02020":             {"name": "安踏体育",                "ticker": "02020.HK", "market": "港股", "sector": "消费",          "confidence": "wait",    "verdict": "当前不宜追入，等 HK$65-75"},
    "xiaomi-01810":           {"name": "小米集团",                "ticker": "01810.HK", "market": "港股", "sector": "科技硬件",      "confidence": "wait",    "verdict": "生态有吸引力但护城河脆弱"},
    "geely-0175":             {"name": "吉利汽车",                "ticker": "0175.HK",  "market": "港股", "sector": "新能源/汽车",   "confidence": "wait",    "verdict": "有条件看多，等 HK$16-19"},
    "cnooc-0883":             {"name": "中国海洋石油",             "ticker": "0883.HK",  "market": "港股", "sector": "能源",          "confidence": "wait",    "verdict": "三桶油最优质，当前 price in 乐观假设"},
    "anker-300866":           {"name": "安克创新",                "ticker": "300866.SZ","market": "A股",  "sector": "科技硬件",      "confidence": "wait",    "verdict": "A股消费电子一梯队，等 ¥85-100"},
    "byd-002594":             {"name": "比亚迪",                  "ticker": "002594.SZ","market": "A股",  "sector": "新能源/汽车",   "confidence": "wait",    "verdict": "安全边际不足，等 ¥70-85"},
    "catl-300750":            {"name": "宁德时代",                "ticker": "300750.SZ","market": "A股",  "sector": "新能源/汽车",   "confidence": "wait",    "verdict": "定价合理但无安全边际，等 ¥300-350"},
    "viking-VIK":             {"name": "Viking Holdings",        "ticker": "VIK",      "market": "美股", "sector": "消费服务",      "confidence": "wait",    "verdict": "优秀生意，$82 太贵，等 $58-65"},
    "nike-NKE":               {"name": "Nike",                   "ticker": "NKE",      "market": "美股", "sector": "消费",          "confidence": "wait",    "verdict": "安全边际不足，最多轻仓试探"},
    "gartner-IT":             {"name": "Gartner",                "ticker": "IT",       "market": "美股", "sector": "SaaS/平台",     "confidence": "wait",    "verdict": "等 $120-130"},
    "nio-NIO":                {"name": "蔚来 NIO",               "ticker": "NIO",      "market": "美股", "sector": "新能源/汽车",   "confidence": "wait",    "verdict": "谨慎看多，GIC 诉讼悬而未决"},
    "chagee-CHA":             {"name": "霸王茶姬 CHAGEE",         "ticker": "CHA",      "market": "美股", "sector": "消费",          "confidence": "wait",    "verdict": "逆向机会但同店恶化需验证"},
    "adobe-ADBE":             {"name": "Adobe",                  "ticker": "ADBE",     "market": "美股", "sector": "SaaS/平台",     "confidence": "wait",    "verdict": "CEO 换帅观察中"},
    "coreweave-CRWV":         {"name": "CoreWeave",              "ticker": "CRWV",     "market": "美股", "sector": "AI基础设施",    "confidence": "pass",    "verdict": "高杠杆 AI 赌注，不适合长期持仓"},
    "tesla-TSLA":             {"name": "Tesla",                  "ticker": "TSLA",     "market": "美股", "sector": "新能源/汽车",   "confidence": "pass",    "verdict": "风险收益比严重不对称"},
    "iren-IREN":              {"name": "IREN",                   "ticker": "IREN",     "market": "美股", "sector": "AI基础设施",    "confidence": "pass",    "verdict": "Insider selling 警示"},
    "micron-MU":              {"name": "Micron",                 "ticker": "MU",       "market": "美股", "sector": "半导体",        "confidence": "pass",    "verdict": "当前不适合建仓，等存储寒冬"},
    "boe-000725":             {"name": "京东方A",                 "ticker": "000725.SZ","market": "A股",  "sector": "半导体/面板",   "confidence": "pass",    "verdict": "ROE 4.39% 毁灭价值，赌 OLED 转型"},
}

def get_updated_date(filepath):
    """Extract date from thesis if present, else use file mtime."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"分析日期[：:]\s*(\d{4}-\d{2}-\d{2})", content)
    if m:
        return m.group(1)
    mtime = os.path.getmtime(filepath)
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

def add_frontmatter(filepath, meta):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if content.startswith("---"):
        print(f"  SKIP (already has frontmatter): {filepath}")
        return
    updated = get_updated_date(filepath)
    fm = f"""---
name: "{meta['name']}"
ticker: "{meta['ticker']}"
market: "{meta['market']}"
sector: "{meta['sector']}"
confidence: "{meta['confidence']}"
verdict: "{meta['verdict']}"
updated: "{updated}"
---

"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(fm + content)
    print(f"  DONE: {filepath}")

def main():
    for stock_id, meta in STOCKS.items():
        thesis = os.path.join(THESIS_DIR, f"{stock_id}-thesis.md")
        if os.path.exists(thesis):
            print(f"[thesis] {stock_id}")
            add_frontmatter(thesis, meta)
        else:
            print(f"  MISSING: {thesis}")

if __name__ == "__main__":
    main()
