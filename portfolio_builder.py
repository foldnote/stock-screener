"""
投资组合构建器
==============
完成剩余流程：
  4. 深度研究（定性）— 同行对比、竞争壁垒、管理层质量
  5. 组合构建 — 行业分散、因子分散、仓位控制
  6. 交易纪律 — 买入价、止损线、目标价、卖出条件

用法：python3 portfolio_builder.py
"""

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# 候选股（DCF 结果）
# ---------------------------------------------------------------------------

CANDIDATES = {
    "ADBE": {"tier": "买入", "dcf_conservative": 307, "dcf_base": 528, "dcf_optimistic": 891, "sector": "Technology"},
    "IT":   {"tier": "买入", "dcf_conservative": 176, "dcf_base": 289, "dcf_optimistic": 470, "sector": "Technology"},
    "T":    {"tier": "买入", "dcf_conservative": 28, "dcf_base": 92, "dcf_optimistic": 212, "sector": "Communication Services"},
    "BMY":  {"tier": "可考虑", "dcf_conservative": 45, "dcf_base": 69, "dcf_optimistic": 107, "sector": "Healthcare"},
    "DLTR": {"tier": "可考虑", "dcf_conservative": 83, "dcf_base": 208, "dcf_optimistic": 445, "sector": "Consumer Defensive"},
}


# ---------------------------------------------------------------------------
# 1. 深度研究（定性）
# ---------------------------------------------------------------------------

@dataclass
class QualitativeReport:
    ticker: str
    name: str
    sector: str
    industry: str
    # 竞争壁垒
    gross_margin: float = 0
    peer_avg_margin: float = 0
    margin_premium: float = 0      # 相对同行的毛利率溢价
    # 管理层
    insider_ownership: float = 0
    institutional_ownership: float = 0
    # 增长质量
    revenue_growth_3y: float = 0
    fcf_growth_3y: float = 0
    revenue_consistency: str = ""   # 连续增长/波动/下降
    # 估值交叉验证
    pe: float = 0
    peer_avg_pe: float = 0
    pb: float = 0
    dividend_yield: float = 0
    # 风险
    beta: float = 0
    debt_to_equity: float = 0
    # 最终评分
    moat_score: int = 0        # 0-100 护城河
    management_score: int = 0  # 0-100 管理层
    growth_score: int = 0      # 0-100 增长质量
    valuation_score: int = 0   # 0-100 估值吸引力
    risk_score: int = 0        # 0-100 风险（越高越安全）
    total_score: float = 0
    issues: list = field(default_factory=list)
    strengths: list = field(default_factory=list)


PEER_GROUPS = {
    "ADBE": ["CRM", "INTU", "NOW", "CDNS", "SNPS"],
    "IT":   ["ACN", "CTSH", "EPAM", "LDOS", "SAIC"],
    "T":    ["VZ", "TMUS", "CMCSA", "CHTR"],
    "BMY":  ["PFE", "MRK", "ABBV", "GILD", "JNJ"],
    "DLTR": ["DG", "COST", "WMT", "TGT", "ROST"],
}


def fetch_peer_margins(peers: list[str]) -> float:
    margins = []
    for p in peers:
        try:
            info = yf.Ticker(p).info
            gm = info.get("grossMargins")
            if gm and gm > 0:
                margins.append(gm)
        except Exception:
            pass
    return np.mean(margins) if margins else 0


def fetch_peer_pe(peers: list[str]) -> float:
    pes = []
    for p in peers:
        try:
            info = yf.Ticker(p).info
            pe = info.get("trailingPE")
            if pe and 0 < pe < 200:
                pes.append(pe)
        except Exception:
            pass
    return np.mean(pes) if pes else 0


def safe_row_vals(df, labels, n=4):
    if df is None or df.empty:
        return []
    for label in labels:
        if label in df.index:
            return df.loc[label].dropna().values[:n].astype(float).tolist()
    return []


def qualitative_analysis(ticker: str) -> QualitativeReport:
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    peers = PEER_GROUPS.get(ticker, [])

    r = QualitativeReport(
        ticker=ticker,
        name=info.get("shortName", ""),
        sector=info.get("sector", ""),
        industry=info.get("industry", ""),
    )

    # --- 竞争壁垒（护城河）---
    r.gross_margin = info.get("grossMargins", 0) or 0
    r.peer_avg_margin = fetch_peer_margins(peers) if peers else 0
    r.margin_premium = r.gross_margin - r.peer_avg_margin

    if r.margin_premium > 0.10:
        r.moat_score = 90
        r.strengths.append(f"毛利率 {r.gross_margin:.1%} 远超同行均值 {r.peer_avg_margin:.1%}，定价权强")
    elif r.margin_premium > 0.03:
        r.moat_score = 70
        r.strengths.append(f"毛利率 {r.gross_margin:.1%} 略高于同行 {r.peer_avg_margin:.1%}")
    elif r.margin_premium > -0.03:
        r.moat_score = 50
    else:
        r.moat_score = 30
        r.issues.append(f"毛利率 {r.gross_margin:.1%} 低于同行均值 {r.peer_avg_margin:.1%}，可能缺乏护城河")

    # --- 管理层质量 ---
    r.insider_ownership = info.get("heldPercentInsiders", 0) or 0
    r.institutional_ownership = info.get("heldPercentInstitutions", 0) or 0

    if r.insider_ownership > 0.05:
        r.management_score = 90
        r.strengths.append(f"内部人持股 {r.insider_ownership:.1%}，管理层利益与股东高度绑定")
    elif r.insider_ownership > 0.01:
        r.management_score = 70
    else:
        r.management_score = 50

    if r.institutional_ownership > 0.80:
        r.management_score = min(r.management_score + 10, 100)
        r.strengths.append(f"机构持股 {r.institutional_ownership:.1%}，受专业投资者认可")

    # --- 增长质量 ---
    try:
        inc = stock.financials
        revs = safe_row_vals(inc, ["Total Revenue", "Revenue"])
        if len(revs) >= 4:
            cagr = (revs[0] / revs[3]) ** (1/3) - 1
            r.revenue_growth_3y = cagr

            diffs = [revs[i] - revs[i+1] for i in range(len(revs)-1)]
            if all(d > 0 for d in diffs):
                r.revenue_consistency = "连续增长"
            elif all(d < 0 for d in diffs):
                r.revenue_consistency = "连续下降"
            else:
                r.revenue_consistency = "波动"
    except Exception:
        pass

    try:
        cf = stock.cashflow
        cfo_vals = safe_row_vals(cf, ["Operating Cash Flow"])
        capex_vals = safe_row_vals(cf, ["Capital Expenditure"])
        if len(cfo_vals) >= 4 and len(capex_vals) >= 4:
            fcfs = [cfo_vals[i] - abs(capex_vals[i]) for i in range(min(len(cfo_vals), len(capex_vals)))]
            if fcfs[3] > 0 and fcfs[0] > 0:
                r.fcf_growth_3y = (fcfs[0] / fcfs[3]) ** (1/3) - 1
    except Exception:
        pass

    if r.revenue_consistency == "连续增长" and r.revenue_growth_3y > 0.05:
        r.growth_score = 90
        r.strengths.append(f"营收连续3年增长，CAGR {r.revenue_growth_3y:.1%}")
    elif r.revenue_growth_3y > 0.03:
        r.growth_score = 70
    elif r.revenue_growth_3y > 0:
        r.growth_score = 50
    else:
        r.growth_score = 30
        r.issues.append(f"营收3年CAGR {r.revenue_growth_3y:.1%}，增长乏力")

    # --- 估值交叉验证 ---
    r.pe = info.get("trailingPE", 0) or 0
    r.peer_avg_pe = fetch_peer_pe(peers) if peers else 0
    r.pb = info.get("priceToBook", 0) or 0
    r.dividend_yield = info.get("dividendYield", 0) or 0

    dcf_info = CANDIDATES.get(ticker, {})
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    dcf_conservative = dcf_info.get("dcf_conservative", 0)

    if price > 0 and dcf_conservative > 0:
        mos = (dcf_conservative - price) / dcf_conservative
        if mos > 0.15:
            r.valuation_score = 90
            r.strengths.append(f"DCF保守估值${dcf_conservative:.0f}，当前${price:.0f}，安全边际{mos:.0%}")
        elif mos > 0:
            r.valuation_score = 70
            r.strengths.append(f"价格略低于保守估值，安全边际{mos:.0%}")
        elif mos > -0.20:
            r.valuation_score = 50
            r.issues.append(f"价格${price:.0f}略高于保守估值${dcf_conservative:.0f}")
        else:
            r.valuation_score = 30
            r.issues.append(f"价格${price:.0f}显著高于保守估值${dcf_conservative:.0f}，安全边际不足")

    if r.pe > 0 and r.peer_avg_pe > 0:
        pe_discount = (r.peer_avg_pe - r.pe) / r.peer_avg_pe
        if pe_discount > 0.2:
            r.strengths.append(f"PE {r.pe:.1f}x 比同行均值 {r.peer_avg_pe:.1f}x 便宜 {pe_discount:.0%}")
        elif pe_discount < -0.2:
            r.issues.append(f"PE {r.pe:.1f}x 比同行均值 {r.peer_avg_pe:.1f}x 贵 {-pe_discount:.0%}")

    # --- 风险评估 ---
    r.beta = info.get("beta", 1) or 1
    de = info.get("debtToEquity")
    r.debt_to_equity = de / 100 if de else 0

    risk = 70
    if r.beta > 1.5:
        risk -= 20
        r.issues.append(f"Beta {r.beta:.1f}，波动大于大盘50%以上")
    elif r.beta < 0.8:
        risk += 10
        r.strengths.append(f"Beta {r.beta:.1f}，波动小于大盘，防御性好")

    if r.debt_to_equity > 2:
        risk -= 20
        r.issues.append(f"D/E {r.debt_to_equity:.1f}x，杠杆偏高")
    elif r.debt_to_equity < 0.5:
        risk += 10

    r.risk_score = max(0, min(100, risk))

    # --- 综合得分 ---
    r.total_score = (
        r.moat_score * 0.25 +
        r.management_score * 0.15 +
        r.growth_score * 0.20 +
        r.valuation_score * 0.25 +
        r.risk_score * 0.15
    )

    return r


# ---------------------------------------------------------------------------
# 2. 组合构建
# ---------------------------------------------------------------------------

@dataclass
class PortfolioPosition:
    ticker: str
    name: str
    sector: str
    weight: float
    current_price: float
    buy_below: float          # 建仓价上限
    stop_loss: float          # 止损价
    target_price: float       # 目标价（基准估值）
    ceiling_price: float      # 乐观估值
    expected_return: float    # 到目标价的预期收益率
    risk_reward: float        # 风险收益比
    qualitative_score: float
    conviction: str           # 高/中/低


@dataclass
class Portfolio:
    positions: list[PortfolioPosition] = field(default_factory=list)
    cash_reserve: float = 0.10
    total_invested: float = 0.90
    rebalance_interval: str = "每季度（财报后）"


def build_portfolio(reports: list[QualitativeReport]) -> Portfolio:
    portfolio = Portfolio()

    qualified = [r for r in reports if r.total_score >= 55]

    if not qualified:
        return portfolio

    qualified.sort(key=lambda x: x.total_score, reverse=True)

    # 行业去重：同行业最多2只
    sector_count = {}
    filtered = []
    for r in qualified:
        s = r.sector
        if sector_count.get(s, 0) < 2:
            filtered.append(r)
            sector_count[s] = sector_count.get(s, 0) + 1

    # 仓位分配：按得分加权
    total_score = sum(r.total_score for r in filtered)
    base_weights = {r.ticker: r.total_score / total_score for r in filtered}

    # 单只上限 35%，下限 10%
    for ticker in base_weights:
        base_weights[ticker] = max(0.10, min(0.35, base_weights[ticker]))

    # 归一化到 90%（10% 留现金）
    w_sum = sum(base_weights.values())
    for ticker in base_weights:
        base_weights[ticker] = base_weights[ticker] / w_sum * 0.90

    # 构建仓位
    for r in filtered:
        dcf = CANDIDATES.get(r.ticker, {})
        price_info = yf.Ticker(r.ticker).info
        price = price_info.get("currentPrice") or price_info.get("regularMarketPrice") or 0

        conservative = dcf.get("dcf_conservative", price)
        base_val = dcf.get("dcf_base", price)
        optimistic = dcf.get("dcf_optimistic", price)

        buy_below = conservative
        stop_loss = price * 0.75    # 25% 止损
        target = base_val
        expected_return = (target - price) / price if price > 0 else 0
        risk = price - stop_loss
        reward = target - price
        risk_reward = reward / risk if risk > 0 else 0

        if r.total_score >= 75:
            conviction = "高"
        elif r.total_score >= 60:
            conviction = "中"
        else:
            conviction = "低"

        pos = PortfolioPosition(
            ticker=r.ticker,
            name=r.name,
            sector=r.sector,
            weight=base_weights.get(r.ticker, 0),
            current_price=price,
            buy_below=buy_below,
            stop_loss=round(stop_loss, 2),
            target_price=target,
            ceiling_price=optimistic,
            expected_return=expected_return,
            risk_reward=risk_reward,
            qualitative_score=r.total_score,
            conviction=conviction,
        )
        portfolio.positions.append(pos)

    return portfolio


# ---------------------------------------------------------------------------
# 3. 输出
# ---------------------------------------------------------------------------

def print_qualitative(reports: list[QualitativeReport]):
    print(f"\n{'=' * 85}")
    print(f"  第四步：深度定性研究")
    print(f"{'=' * 85}")

    for r in reports:
        print(f"\n{'━' * 85}")
        print(f"  {r.ticker} | {r.name} | {r.industry}")
        print(f"{'━' * 85}")

        print(f"\n  各维度评分（满分100）：")
        print(f"    护城河    {'█' * (r.moat_score // 5):<20} {r.moat_score}")
        print(f"    管理层    {'█' * (r.management_score // 5):<20} {r.management_score}")
        print(f"    增长质量  {'█' * (r.growth_score // 5):<20} {r.growth_score}")
        print(f"    估值吸引  {'█' * (r.valuation_score // 5):<20} {r.valuation_score}")
        print(f"    风险控制  {'█' * (r.risk_score // 5):<20} {r.risk_score}")
        print(f"    ──────────────────────────────────────")
        print(f"    综合得分  {'█' * (int(r.total_score) // 5):<20} {r.total_score:.0f}")

        if r.strengths:
            print(f"\n  ✅ 优势：")
            for s in r.strengths:
                print(f"     • {s}")

        if r.issues:
            print(f"\n  ⚠️ 关注点：")
            for s in r.issues:
                print(f"     • {s}")

        # 关键数据
        print(f"\n  📊 关键比较：")
        print(f"     毛利率: {r.gross_margin:.1%} vs 同行 {r.peer_avg_margin:.1%}")
        print(f"     PE: {r.pe:.1f}x vs 同行 {r.peer_avg_pe:.1f}x")
        print(f"     营收3年CAGR: {r.revenue_growth_3y:.1%}")
        print(f"     FCF 3年CAGR: {r.fcf_growth_3y:.1%}")
        print(f"     Beta: {r.beta:.2f}")
        print(f"     D/E: {r.debt_to_equity:.1f}x")
        if r.dividend_yield > 0:
            print(f"     股息率: {r.dividend_yield:.1%}")

    # 定性筛选汇总
    print(f"\n{'─' * 85}")
    print(f"  定性研究汇总：")
    print(f"  {'代码':<7} {'护城河':>6} {'管理层':>6} {'增长':>6} {'估值':>6} {'风险':>6} {'综合':>6} {'结论'}")
    print(f"  {'─' * 70}")
    for r in sorted(reports, key=lambda x: x.total_score, reverse=True):
        verdict = "✅ 入选" if r.total_score >= 55 else "❌ 淘汰"
        print(f"  {r.ticker:<7} {r.moat_score:>5} {r.management_score:>5} {r.growth_score:>5} "
              f"{r.valuation_score:>5} {r.risk_score:>5} {r.total_score:>5.0f}  {verdict}")


def print_portfolio(portfolio: Portfolio):
    print(f"\n\n{'=' * 85}")
    print(f"  第五步：最终投资组合")
    print(f"  生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 85}")

    if not portfolio.positions:
        print("  无合格标的")
        return

    print(f"\n  资金分配：")
    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    for p in portfolio.positions:
        bar = '█' * int(p.weight * 100)
        print(f"  │ {p.ticker:<6} {bar:<35} {p.weight:>5.1%} │")
    print(f"  │ {'现金':<6} {'░' * int(portfolio.cash_reserve * 100):<35} {portfolio.cash_reserve:>5.1%} │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    print(f"\n  持仓明细：")
    print(f"  ┌{'─' * 83}┐")
    print(f"  │ {'代码':<6} │ {'行业':<16} │ {'仓位':>5} │ {'现价':>7} │ {'建仓上限':>8} │"
          f" {'目标价':>7} │ {'止损':>7} │ {'预期收益':>8} │")
    print(f"  ├{'─' * 83}┤")

    for p in portfolio.positions:
        sector_short = p.sector[:14]
        print(f"  │ {p.ticker:<6} │ {sector_short:<16} │ {p.weight:>4.0%} │"
              f" ${p.current_price:>5.0f} │ ${p.buy_below:>7.0f} │"
              f" ${p.target_price:>5.0f} │ ${p.stop_loss:>5.0f} │"
              f" {p.expected_return:>+7.0%} │")
    print(f"  └{'─' * 83}┘")

    # 组合特征
    avg_return = np.mean([p.expected_return for p in portfolio.positions])
    avg_rr = np.mean([p.risk_reward for p in portfolio.positions])
    sectors = set(p.sector for p in portfolio.positions)

    print(f"\n  组合特征：")
    print(f"    持仓数量        {len(portfolio.positions)} 只")
    print(f"    行业覆盖        {len(sectors)} 个行业")
    print(f"    平均预期收益    {avg_return:+.0%}")
    print(f"    平均风险收益比  {avg_rr:.1f} : 1")
    print(f"    现金储备        {portfolio.cash_reserve:.0%}")
    print(f"    再平衡周期      {portfolio.rebalance_interval}")


def print_trading_rules(portfolio: Portfolio):
    print(f"\n\n{'=' * 85}")
    print(f"  第六步：交易纪律")
    print(f"{'=' * 85}")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         买  入  规  则                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. 只在「建仓上限」以下买入，不追高                                │
  │  2. 分批建仓：首次买入目标仓位的 50%，跌 5% 再加 30%，跌 10% 补齐  │
  │  3. 单日不买入超过目标仓位的 50%                                    │
  │  4. 大盘暴跌日（SPY 日跌 > 3%）暂停买入，等 2 个交易日再决定       │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                         卖  出  规  则                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  触发任何一条即卖出：                                               │
  │                                                                     │
  │  🔴 止损：跌破止损价 → 无条件卖出全部仓位                          │""")

    for p in portfolio.positions:
        print(f"  │     {p.ticker}: 当前 ${p.current_price:.0f} → 止损 ${p.stop_loss:.0f}（-25%）{' ' * (26 - len(p.ticker))}│")

    print(f"""  │                                                                     │
  │  🟢 止盈：涨到目标价 → 卖出 50%，上移止损到成本价                  │""")

    for p in portfolio.positions:
        print(f"  │     {p.ticker}: 目标 ${p.target_price:.0f}{' ' * (48 - len(p.ticker) - len(f'{p.target_price:.0f}'))}│")

    print(f"""  │                                                                     │
  │  ⚠️ 基本面恶化（触发任一条）：                                     │
  │     • 连续 2 个季度 FCF 同比下降 > 20%                             │
  │     • 毛利率单季下降超过 3 个百分点                                 │
  │     • 管理层意外变动（CEO/CFO辞职）                                 │
  │     • 审计师更换或审计意见有保留                                     │
  │     • 出现重大诉讼/监管调查                                         │
  │                                                                     │
  │  📅 定期审查：                                                      │
  │     • 每季度财报后重跑筛选器 + 排雷器 + DCF                        │
  │     • 如果某只股票排名跌出 Top 50 → 降低仓位到 5%                  │
  │     • 如果排雷器新增 🔴 → 立即卖出                                 │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘""")

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         仓  位  管  理                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  • 单只股票最大仓位    35%（到达后不再加仓）                        │
  │  • 单个行业最大仓位    50%                                          │
  │  • 现金最低保留        10%（永远留子弹）                            │
  │  • 再平衡：每季度末检查，偏离目标权重 > 5% 时调整                  │
  │  • 新机会：有更好的标的出现时，替换得分最低的持仓                   │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘""")


def print_action_plan(portfolio: Portfolio):
    print(f"\n\n{'=' * 85}")
    print(f"  立即行动清单")
    print(f"{'=' * 85}\n")

    for i, p in enumerate(portfolio.positions, 1):
        if p.current_price <= p.buy_below:
            action = "可立即建仓（价格在建仓区间内）"
            urgency = "🟢"
        elif p.current_price <= p.buy_below * 1.05:
            action = "接近建仓区间，设置价格提醒"
            urgency = "🟡"
        else:
            action = f"等待回调到 ${p.buy_below:.0f} 以下"
            urgency = "⏳"

        print(f"  {urgency} {i}. {p.ticker}（{p.name}）")
        print(f"     目标仓位 {p.weight:.0%} | 当前价 ${p.current_price:.0f} | 建仓上限 ${p.buy_below:.0f}")
        print(f"     → {action}")
        if p.current_price <= p.buy_below:
            first_batch = p.weight * 0.5
            print(f"     → 第一批买入 {first_batch:.0%} 仓位（总资金的 {first_batch:.0%}）")
        print()

    print(f"  ⏰ 设置以下提醒：")
    for p in portfolio.positions:
        if p.current_price > p.buy_below:
            print(f"     • {p.ticker} 跌到 ${p.buy_below:.0f} 时提醒买入")
        print(f"     • {p.ticker} 跌到 ${p.stop_loss:.0f} 时提醒止损")
        print(f"     • {p.ticker} 涨到 ${p.target_price:.0f} 时提醒止盈")


def print_final_summary(reports: list[QualitativeReport], portfolio: Portfolio):
    print(f"\n\n{'═' * 85}")
    print(f"  ╔═══════════════════════════════════════════════════════════════╗")
    print(f"  ║               完 整 流 程 执 行 完 毕                       ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝")
    print(f"{'═' * 85}")

    print(f"""
  执行路径：

  标普500（503只）
    │  7因子量化筛选
    ▼
  候选名单（30只）
    │  行业专属排雷（11项检查 × 行业差异化阈值）
    ▼
  排雷通过（8只）
    │  三情景DCF估值
    ▼
  估值合理（5只）
    │  定性研究（护城河/管理层/增长/风险）
    ▼
  最终组合（{len(portfolio.positions)}只）
""")

    if portfolio.positions:
        print(f"  最终入选：")
        for p in portfolio.positions:
            print(f"    {'🟢' if p.current_price <= p.buy_below else '🟡'} {p.ticker:<6} "
                  f"仓位{p.weight:.0%}  买入≤${p.buy_below:.0f}  "
                  f"目标${p.target_price:.0f}  止损${p.stop_loss:.0f}  "
                  f"信心:{p.conviction}")

        print(f"\n  ⚠️ 此组合基于公开财务数据的量化分析 + 系统化定性评估。")
        print(f"     不构成投资建议。实际投资前请：")
        print(f"     1. 阅读每家公司最近的 10-K 和 Earnings Call")
        print(f"     2. 了解你自己的风险承受能力")
        print(f"     3. 咨询持牌投资顾问")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'=' * 50}")
    print(f"  投资组合构建器")
    print(f"  流程：定性研究 → 组合构建 → 交易纪律")
    print(f"{'=' * 50}")

    # 第四步：定性研究
    print(f"\n正在进行深度定性分析...")
    reports = []
    for i, ticker in enumerate(CANDIDATES.keys(), 1):
        print(f"\r  [{i}/{len(CANDIDATES)}] 分析 {ticker:<7}（含同行对比）", end="", flush=True)
        try:
            r = qualitative_analysis(ticker)
            reports.append(r)
        except Exception as e:
            print(f"\n  ⚠️ {ticker} 分析失败: {e}")
        time.sleep(0.5)
    print()

    print_qualitative(reports)

    # 第五步：组合构建
    portfolio = build_portfolio(reports)
    print_portfolio(portfolio)

    # 第六步：交易纪律
    print_trading_rules(portfolio)

    # 行动清单
    print_action_plan(portfolio)

    # 最终汇总
    print_final_summary(reports, portfolio)


if __name__ == "__main__":
    main()
