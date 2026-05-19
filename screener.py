"""
7因子股票筛选器
=============
因子：P/E, P/FCF, ROE, 毛利率, 资产负债率, 12个月动量, 回购力度
数据源：Yahoo Finance（免费）
用法：python3 screener.py [--top 20] [--universe sp500|custom] [--save]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# 1. 股票池
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """从 Wikipedia 拉取标普500成分股列表"""
    import io
    import urllib.request

    print("正在获取标普500成分股列表...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode()
    table = pd.read_html(io.StringIO(html))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"  共 {len(tickers)} 只股票")
    return tickers


DEMO_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "BRK-B", "JNJ", "JPM", "V", "PG",
    "UNH", "HD", "MA", "NVDA", "XOM",
    "PFE", "KO", "PEP", "MRK", "ABBV",
    "CVX", "COST", "WMT", "BAC", "TMO",
    "CSCO", "ABT", "CRM", "MCD", "NKE",
]


# ---------------------------------------------------------------------------
# 2. 数据获取
# ---------------------------------------------------------------------------

def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """获取每只股票的基本面数据"""
    records = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        pct = i / total * 100
        print(f"\r  [{i}/{total}] ({pct:.0f}%) 正在获取 {ticker:<6}", end="", flush=True)

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("quoteType") not in ("EQUITY", None):
                continue

            record = {
                "ticker": ticker,
                "name": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                # 价值因子
                "pe": info.get("trailingPE"),
                "p_fcf": _calc_p_fcf(info),
                # 质量因子
                "roe": info.get("returnOnEquity"),
                "gross_margin": info.get("grossMargins"),
                "debt_to_asset": _calc_debt_to_asset(info),
                # 行为因子
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
            }
            records.append(record)

        except Exception:
            pass

        if i % 10 == 0:
            time.sleep(0.5)

    print()
    return pd.DataFrame(records)


def _calc_p_fcf(info: dict) -> float | None:
    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    if mcap and fcf and fcf > 0:
        return mcap / fcf
    return None


def _calc_debt_to_asset(info: dict) -> float | None:
    debt = info.get("totalDebt", 0)
    equity = info.get("totalStockholderEquity")
    assets_direct = info.get("totalAssets")
    if assets_direct and assets_direct > 0:
        return (debt or 0) / assets_direct
    if equity is not None and debt is not None:
        total = debt + equity
        if total > 0:
            return debt / total
    return None


def fetch_momentum(tickers: list[str]) -> pd.Series:
    """批量获取12个月动量（去掉最近1个月）"""
    print("正在计算12个月动量...")
    end = datetime.now()
    start = end - timedelta(days=395)

    data = yf.download(tickers, start=start, end=end, progress=False, threads=True)

    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]

    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    price_12m_ago = close.iloc[:25].mean()
    price_1m_ago = close.iloc[-25:].mean()
    momentum = (price_1m_ago - price_12m_ago) / price_12m_ago

    print(f"  动量计算完成，覆盖 {momentum.dropna().shape[0]} 只股票")
    return momentum


def fetch_buyback_signal(tickers: list[str]) -> pd.Series:
    """通过股数变化估算回购力度（股数减少 = 在回购 = 正面信号）"""
    print("正在估算回购力度...")
    signals = {}

    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"\r  [{i}/{len(tickers)}]", end="", flush=True)
        try:
            stock = yf.Ticker(ticker)
            bs = stock.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                shares_row = None
                for label in ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]:
                    if label in bs.index:
                        shares_row = bs.loc[label]
                        break
                if shares_row is not None and len(shares_row.dropna()) >= 2:
                    latest = shares_row.dropna().iloc[0]
                    oldest = shares_row.dropna().iloc[-1]
                    if oldest > 0:
                        signals[ticker] = (oldest - latest) / oldest
        except Exception:
            pass

    print()
    return pd.Series(signals)


# ---------------------------------------------------------------------------
# 3. 评分
# ---------------------------------------------------------------------------

def percentile_score(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """将原始值转换为 0-100 的百分位得分"""
    ranked = series.rank(pct=True, na_option="keep")
    if lower_is_better:
        ranked = 1 - ranked
    return ranked * 100


def score_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """计算综合得分"""

    df = df.copy()

    # 过滤：只保留有基本数据的股票
    required = ["pe", "roe", "gross_margin"]
    df = df.dropna(subset=required, how="all")

    # 过滤：P/E 为负说明亏损
    df = df[df["pe"] > 0]

    # 过滤：市值太小的不要（< 20亿美元）
    if "market_cap" in df.columns:
        df = df[df["market_cap"] >= 2e9]

    # --- 各因子评分 ---
    #
    # 因子               权重    方向
    # P/E                30%    越低越好
    # P/FCF              15%    越低越好
    # ROE                15%    越高越好
    # 毛利率             10%    越高越好
    # 资产负债率         10%    越低越好
    # 12个月动量         10%    越高越好
    # 回购力度           10%    越高越好

    weights = {
        "pe_score":           0.30,
        "p_fcf_score":        0.15,
        "roe_score":          0.15,
        "gross_margin_score": 0.10,
        "debt_score":         0.10,
        "momentum_score":     0.10,
        "buyback_score":      0.10,
    }

    df["pe_score"]           = percentile_score(df["pe"], lower_is_better=True)
    df["p_fcf_score"]        = percentile_score(df["p_fcf"], lower_is_better=True)
    df["roe_score"]          = percentile_score(df["roe"])
    df["gross_margin_score"] = percentile_score(df["gross_margin"])
    df["debt_score"]         = percentile_score(df["debt_to_asset"], lower_is_better=True)
    df["momentum_score"]     = percentile_score(df["momentum"])
    df["buyback_score"]      = percentile_score(df["buyback"])

    # 综合得分：跳过全空的因子列，权重重新分配
    score_cols = list(weights.keys())
    active_weights = {}
    for col in score_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mean())
            active_weights[col] = weights[col]

    total_w = sum(active_weights.values())
    df["total_score"] = sum(df[col] * (w / total_w) for col, w in active_weights.items())
    df["total_score"] = df["total_score"].round(1)

    return df.sort_values("total_score", ascending=False)


# ---------------------------------------------------------------------------
# 4. 输出
# ---------------------------------------------------------------------------

def print_results(df: pd.DataFrame, top_n: int):
    top = df.head(top_n)

    print(f"\n{'='*90}")
    print(f"  7因子筛选结果 TOP {top_n}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}")

    print(f"\n{'排名':>4}  {'代码':<7} {'公司名称':<24} {'行业':<18} {'综合分':>6}")
    print(f"{'':>4}  {'P/E':>7} {'P/FCF':>7} {'ROE':>7} {'毛利率':>7} {'负债率':>7} {'动量':>8} {'回购':>7}")
    print("-" * 90)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        name = (row["name"] or "")[:22]
        sector = (row["sector"] or "")[:16]
        print(f"\n{rank:>4}. {row['ticker']:<7} {name:<24} {sector:<18} {row['total_score']:>5.1f}")

        pe_str = f"{row['pe']:.1f}" if pd.notna(row['pe']) else "N/A"
        pfcf_str = f"{row['p_fcf']:.1f}" if pd.notna(row['p_fcf']) else "N/A"
        roe_str = f"{row['roe']*100:.1f}%" if pd.notna(row['roe']) else "N/A"
        gm_str = f"{row['gross_margin']*100:.1f}%" if pd.notna(row['gross_margin']) else "N/A"
        da_str = f"{row['debt_to_asset']*100:.1f}%" if pd.notna(row.get('debt_to_asset')) else "N/A"
        mom_str = f"{row['momentum']*100:+.1f}%" if pd.notna(row.get('momentum')) else "N/A"
        bb_str = f"{row['buyback']*100:.2f}%" if pd.notna(row.get('buyback')) else "N/A"

        print(f"      {pe_str:>7} {pfcf_str:>7} {roe_str:>7} {gm_str:>7} {da_str:>7} {mom_str:>8} {bb_str:>7}")

    print(f"\n{'='*90}")

    # 各因子得分分布
    print(f"\n分因子得分（满分100）:")
    print(f"{'排名':>4}  {'代码':<7} {'价值PE':>7} {'价值FCF':>8} {'质量ROE':>8} {'毛利率':>7} {'低负债':>7} {'动量':>6} {'回购':>6}")
    print("-" * 70)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        print(f"{rank:>4}. {row['ticker']:<7}"
              f" {row['pe_score']:>6.0f}"
              f" {row['p_fcf_score']:>7.0f}"
              f" {row['roe_score']:>7.0f}"
              f" {row['gross_margin_score']:>6.0f}"
              f" {row['debt_score']:>6.0f}"
              f" {row['momentum_score']:>5.0f}"
              f" {row['buyback_score']:>5.0f}")


def save_results(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"\n完整结果已保存到: {path}")


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="7因子股票筛选器")
    parser.add_argument("--top", type=int, default=20, help="显示排名前N的股票（默认20）")
    parser.add_argument("--universe", choices=["sp500", "demo"], default="demo",
                        help="股票池：sp500=标普500全部（慢），demo=30只代表股（快）")
    parser.add_argument("--save", action="store_true", help="保存完整结果到CSV")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  7因子股票筛选器")
    print("=" * 50)

    # 获取股票池
    if args.universe == "sp500":
        tickers = get_sp500_tickers()
    else:
        tickers = DEMO_TICKERS
        print(f"使用演示股票池（{len(tickers)} 只）。用 --universe sp500 筛选全部标普500。")

    # 获取数据
    print("\n[1/4] 获取基本面数据...")
    df = fetch_fundamentals(tickers)

    if df.empty:
        print("错误：无法获取任何股票数据。请检查网络连接。")
        sys.exit(1)

    print(f"  成功获取 {len(df)} 只股票的基本面数据")

    print("\n[2/4] 计算动量...")
    momentum = fetch_momentum(tickers)
    df["momentum"] = df["ticker"].map(momentum)

    print("\n[3/4] 估算回购力度...")
    buyback = fetch_buyback_signal(tickers)
    df["buyback"] = df["ticker"].map(buyback)

    print(f"\n[4/4] 计算综合得分...")
    scored = score_stocks(df)
    print(f"  最终纳入评分的股票：{len(scored)} 只")

    # 输出
    print_results(scored, min(args.top, len(scored)))

    if args.save:
        out_path = Path(__file__).parent / f"results_{datetime.now():%Y%m%d_%H%M}.csv"
        save_results(scored, out_path)

    print(f"\n提示：这是基于公开财报数据的量化筛选，不构成投资建议。")
    print(f"      筛选结果应作为进一步研究的起点，而非买入信号。\n")


if __name__ == "__main__":
    main()
