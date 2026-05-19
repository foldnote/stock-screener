"""
DCF 估值器
==========
对排雷后的候选股进行三情景 DCF 估值。
按行业区分估值方法：
  - 标准 FCF 折现：科技、消费、医疗、通信、工业
  - 正常化利润 FCF：能源、基础材料（周期股）
  - 超额收益/DDM：金融

用法：
  python3 dcf_valuation.py ADBE DVN TROW BMY T DLTR APA IT
  python3 dcf_valuation.py --survivors   # 直接跑排雷存活的8只
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


SURVIVORS = ["ADBE", "DVN", "TROW", "BMY", "T", "DLTR", "APA", "IT"]


# ---------------------------------------------------------------------------
# 1. 数据获取
# ---------------------------------------------------------------------------

@dataclass
class FinData:
    ticker: str
    info: dict = field(default_factory=dict)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)


def fetch_data(ticker: str) -> FinData:
    stock = yf.Ticker(ticker)
    d = FinData(ticker=ticker, info=stock.info or {})
    try:
        d.income_stmt = stock.financials
    except Exception:
        pass
    try:
        d.balance_sheet = stock.balance_sheet
    except Exception:
        pass
    try:
        d.cashflow = stock.cashflow
    except Exception:
        pass
    return d


def safe_row(df: pd.DataFrame, labels: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def row_values(df: pd.DataFrame, labels: list[str], n: int = 4) -> list[float]:
    row = safe_row(df, labels)
    if row is None:
        return []
    vals = row.dropna().values[:n].astype(float).tolist()
    return vals


# ---------------------------------------------------------------------------
# 2. 行业分类与参数
# ---------------------------------------------------------------------------

@dataclass
class DCFParams:
    method: str  # "standard", "normalized", "ddm"
    discount_rate_low: float = 0.08
    discount_rate_mid: float = 0.10
    discount_rate_high: float = 0.12
    terminal_growth: float = 0.025
    projection_years: int = 10
    # 情景调整
    growth_haircut_conservative: float = 0.5   # 保守情景用历史增速的几折
    growth_haircut_optimistic: float = 1.3     # 乐观情景用历史增速的几倍
    terminal_multiple_override: float | None = None


SECTOR_PARAMS = {
    "Technology": DCFParams(
        method="standard",
        discount_rate_low=0.08,
        discount_rate_mid=0.10,
        discount_rate_high=0.12,
    ),
    "Energy": DCFParams(
        method="normalized",
        discount_rate_low=0.09,
        discount_rate_mid=0.11,
        discount_rate_high=0.13,
        growth_haircut_conservative=0.3,
        terminal_growth=0.02,
    ),
    "Financial Services": DCFParams(
        method="ddm",
        discount_rate_low=0.08,
        discount_rate_mid=0.10,
        discount_rate_high=0.12,
        terminal_growth=0.03,
    ),
    "Healthcare": DCFParams(
        method="standard",
        discount_rate_low=0.08,
        discount_rate_mid=0.10,
        discount_rate_high=0.12,
        growth_haircut_conservative=0.4,
    ),
    "Communication Services": DCFParams(
        method="standard",
        discount_rate_low=0.07,
        discount_rate_mid=0.09,
        discount_rate_high=0.11,
        terminal_growth=0.02,
    ),
    "Consumer Defensive": DCFParams(
        method="standard",
        discount_rate_low=0.07,
        discount_rate_mid=0.09,
        discount_rate_high=0.11,
        terminal_growth=0.025,
    ),
    "Consumer Cyclical": DCFParams(
        method="standard",
        discount_rate_low=0.08,
        discount_rate_mid=0.10,
        discount_rate_high=0.12,
    ),
}

DEFAULT_PARAMS = DCFParams(method="standard")


def get_params(sector: str) -> DCFParams:
    return SECTOR_PARAMS.get(sector, DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# 3. 历史财务数据提取
# ---------------------------------------------------------------------------

@dataclass
class HistoricalMetrics:
    fcf_values: list[float] = field(default_factory=list)
    revenue_values: list[float] = field(default_factory=list)
    net_income_values: list[float] = field(default_factory=list)
    dividend_values: list[float] = field(default_factory=list)
    fcf_growth_avg: float | None = None
    revenue_growth_avg: float | None = None
    fcf_margin_avg: float | None = None
    latest_fcf: float | None = None
    latest_revenue: float | None = None
    latest_net_income: float | None = None
    shares_outstanding: float | None = None
    current_price: float | None = None
    market_cap: float | None = None
    cash: float | None = None
    total_debt: float | None = None
    net_debt: float | None = None
    book_value_per_share: float | None = None
    dividend_per_share: float | None = None
    payout_ratio: float | None = None


def extract_metrics(data: FinData) -> HistoricalMetrics:
    m = HistoricalMetrics()

    # FCF
    cfo_vals = row_values(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex_vals = row_values(data.cashflow, ["Capital Expenditure", "Capital Expenditures"])

    if cfo_vals and capex_vals:
        n = min(len(cfo_vals), len(capex_vals))
        m.fcf_values = [cfo_vals[i] - abs(capex_vals[i]) for i in range(n)]
        m.latest_fcf = m.fcf_values[0] if m.fcf_values else None

    # Revenue
    m.revenue_values = row_values(data.income_stmt, ["Total Revenue", "Revenue"])
    m.latest_revenue = m.revenue_values[0] if m.revenue_values else None

    # Net Income
    m.net_income_values = row_values(data.income_stmt, ["Net Income", "Net Income Common Stockholders"])
    m.latest_net_income = m.net_income_values[0] if m.net_income_values else None

    # Dividends
    div_vals = row_values(data.cashflow, [
        "Common Stock Dividend Paid", "Cash Dividends Paid",
        "Payment Of Dividends And Other Cash Distributions",
    ])
    m.dividend_values = [abs(v) for v in div_vals] if div_vals else []

    # Growth rates
    if len(m.fcf_values) >= 2:
        growths = []
        for i in range(len(m.fcf_values) - 1):
            prev = m.fcf_values[i + 1]
            if prev > 0:
                growths.append((m.fcf_values[i] - prev) / prev)
        m.fcf_growth_avg = np.mean(growths) if growths else None

    if len(m.revenue_values) >= 2:
        growths = []
        for i in range(len(m.revenue_values) - 1):
            prev = m.revenue_values[i + 1]
            if prev > 0:
                growths.append((m.revenue_values[i] - prev) / prev)
        m.revenue_growth_avg = np.mean(growths) if growths else None

    # FCF Margin
    if m.fcf_values and m.revenue_values:
        margins = []
        for i in range(min(len(m.fcf_values), len(m.revenue_values))):
            if m.revenue_values[i] > 0:
                margins.append(m.fcf_values[i] / m.revenue_values[i])
        m.fcf_margin_avg = np.mean(margins) if margins else None

    # Info-based metrics
    m.shares_outstanding = data.info.get("sharesOutstanding")
    m.current_price = data.info.get("currentPrice") or data.info.get("regularMarketPrice")
    m.market_cap = data.info.get("marketCap")
    m.cash = data.info.get("totalCash", 0)
    m.total_debt = data.info.get("totalDebt", 0)
    m.net_debt = (m.total_debt or 0) - (m.cash or 0)
    m.book_value_per_share = data.info.get("bookValue")
    m.dividend_per_share = data.info.get("dividendRate")

    if m.dividend_per_share and m.latest_net_income and m.shares_outstanding:
        eps = m.latest_net_income / m.shares_outstanding
        if eps > 0:
            m.payout_ratio = m.dividend_per_share / eps

    return m


# ---------------------------------------------------------------------------
# 4. DCF 计算引擎
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    growth_rates: list[float]    # 每年的增长率
    discount_rate: float
    terminal_growth: float
    projected_fcfs: list[float] = field(default_factory=list)
    terminal_value: float = 0
    pv_fcfs: float = 0
    pv_terminal: float = 0
    enterprise_value: float = 0
    equity_value: float = 0
    value_per_share: float = 0
    margin_of_safety: float = 0


@dataclass
class ValuationResult:
    ticker: str
    name: str
    sector: str
    method: str
    current_price: float
    shares: float
    net_debt: float
    metrics: HistoricalMetrics = None
    conservative: Scenario = None
    base: Scenario = None
    optimistic: Scenario = None
    buy_price: float = 0     # 保守情景的内在价值（买入价）
    fair_price: float = 0    # 基准情景的内在价值（合理价）
    upside_price: float = 0  # 乐观情景
    recommendation: str = ""
    error: str = ""


def run_standard_dcf(metrics: HistoricalMetrics, params: DCFParams) -> tuple[Scenario, Scenario, Scenario]:
    """标准 FCF 折现法：科技、消费、医疗、通信"""
    base_fcf = metrics.latest_fcf
    if not base_fcf or base_fcf <= 0:
        if metrics.latest_net_income and metrics.latest_net_income > 0:
            base_fcf = metrics.latest_net_income * 0.8
        else:
            return None, None, None

    hist_growth = metrics.fcf_growth_avg or metrics.revenue_growth_avg or 0.05
    hist_growth = max(min(hist_growth, 0.30), -0.05)

    scenarios = []
    for name, growth_mult, disc_rate in [
        ("保守", params.growth_haircut_conservative, params.discount_rate_high),
        ("基准", 1.0, params.discount_rate_mid),
        ("乐观", params.growth_haircut_optimistic, params.discount_rate_low),
    ]:
        base_growth = hist_growth * growth_mult
        base_growth = max(base_growth, 0.0)

        growth_rates = []
        for year in range(params.projection_years):
            fade = 1 - (year / params.projection_years) * 0.6
            g = base_growth * fade + params.terminal_growth * (1 - fade)
            growth_rates.append(g)

        s = Scenario(
            name=name,
            growth_rates=growth_rates,
            discount_rate=disc_rate,
            terminal_growth=params.terminal_growth,
        )

        fcf = base_fcf
        for g in growth_rates:
            fcf = fcf * (1 + g)
            s.projected_fcfs.append(fcf)

        final_fcf = s.projected_fcfs[-1]
        s.terminal_value = final_fcf * (1 + params.terminal_growth) / (disc_rate - params.terminal_growth)

        s.pv_fcfs = sum(
            f / (1 + disc_rate) ** (i + 1)
            for i, f in enumerate(s.projected_fcfs)
        )
        s.pv_terminal = s.terminal_value / (1 + disc_rate) ** params.projection_years

        s.enterprise_value = s.pv_fcfs + s.pv_terminal
        scenarios.append(s)

    return tuple(scenarios)


def run_normalized_dcf(metrics: HistoricalMetrics, params: DCFParams) -> tuple[Scenario, Scenario, Scenario]:
    """正常化利润 DCF：能源/周期股，用历史平均 FCF 而非最新"""
    if not metrics.fcf_values:
        return None, None, None

    positive_fcfs = [f for f in metrics.fcf_values if f > 0]
    if not positive_fcfs:
        return None, None, None

    normalized_fcf = np.mean(positive_fcfs)

    hist_growth = metrics.revenue_growth_avg or 0.02
    hist_growth = max(min(hist_growth, 0.15), -0.02)

    scenarios = []
    for name, growth_mult, disc_rate in [
        ("保守", params.growth_haircut_conservative, params.discount_rate_high),
        ("基准", 0.7, params.discount_rate_mid),
        ("乐观", 1.0, params.discount_rate_low),
    ]:
        base_growth = hist_growth * growth_mult
        base_growth = max(base_growth, 0.0)

        growth_rates = []
        for year in range(params.projection_years):
            fade = 1 - (year / params.projection_years) * 0.7
            g = base_growth * fade + params.terminal_growth * (1 - fade)
            growth_rates.append(g)

        s = Scenario(
            name=name,
            growth_rates=growth_rates,
            discount_rate=disc_rate,
            terminal_growth=params.terminal_growth,
        )

        fcf = normalized_fcf
        for g in growth_rates:
            fcf = fcf * (1 + g)
            s.projected_fcfs.append(fcf)

        final_fcf = s.projected_fcfs[-1]
        s.terminal_value = final_fcf * (1 + params.terminal_growth) / (disc_rate - params.terminal_growth)

        s.pv_fcfs = sum(
            f / (1 + disc_rate) ** (i + 1)
            for i, f in enumerate(s.projected_fcfs)
        )
        s.pv_terminal = s.terminal_value / (1 + disc_rate) ** params.projection_years

        s.enterprise_value = s.pv_fcfs + s.pv_terminal
        scenarios.append(s)

    return tuple(scenarios)


def run_ddm(metrics: HistoricalMetrics, params: DCFParams) -> tuple[Scenario, Scenario, Scenario]:
    """股息折现 + 超额收益：金融公司"""
    div = metrics.dividend_values[0] if metrics.dividend_values else None
    ni = metrics.latest_net_income

    if not ni or ni <= 0:
        return run_standard_dcf(metrics, params)

    if not div or div <= 0:
        payout = 0.3
        div = ni * payout
    else:
        payout = div / ni
        payout = min(payout, 0.9)

    retained = ni * (1 - payout)

    hist_growth = metrics.revenue_growth_avg or 0.05
    hist_growth = max(min(hist_growth, 0.15), 0.0)

    scenarios = []
    for name, growth_mult, disc_rate in [
        ("保守", params.growth_haircut_conservative, params.discount_rate_high),
        ("基准", 0.8, params.discount_rate_mid),
        ("乐观", 1.2, params.discount_rate_low),
    ]:
        base_growth = hist_growth * growth_mult
        base_growth = max(base_growth, 0.01)

        growth_rates = []
        for year in range(params.projection_years):
            fade = 1 - (year / params.projection_years) * 0.5
            g = base_growth * fade + params.terminal_growth * (1 - fade)
            growth_rates.append(g)

        s = Scenario(
            name=name,
            growth_rates=growth_rates,
            discount_rate=disc_rate,
            terminal_growth=params.terminal_growth,
        )

        current_earnings = ni
        for g in growth_rates:
            current_earnings = current_earnings * (1 + g)
            distributable = current_earnings * payout + retained * 0.1
            s.projected_fcfs.append(distributable)

        final_dist = s.projected_fcfs[-1]
        s.terminal_value = final_dist * (1 + params.terminal_growth) / (disc_rate - params.terminal_growth)

        s.pv_fcfs = sum(
            f / (1 + disc_rate) ** (i + 1)
            for i, f in enumerate(s.projected_fcfs)
        )
        s.pv_terminal = s.terminal_value / (1 + disc_rate) ** params.projection_years

        s.enterprise_value = s.pv_fcfs + s.pv_terminal
        scenarios.append(s)

    return tuple(scenarios)


# ---------------------------------------------------------------------------
# 5. 估值主逻辑
# ---------------------------------------------------------------------------

def valuate(ticker: str) -> ValuationResult:
    data = fetch_data(ticker)
    sector = data.info.get("sector", "")
    params = get_params(sector)
    metrics = extract_metrics(data)

    result = ValuationResult(
        ticker=ticker,
        name=data.info.get("shortName", ""),
        sector=sector,
        method=params.method,
        current_price=metrics.current_price or 0,
        shares=metrics.shares_outstanding or 0,
        net_debt=metrics.net_debt or 0,
        metrics=metrics,
    )

    if not metrics.current_price or not metrics.shares_outstanding:
        result.error = "缺少股价或股数数据"
        return result

    # 选择估值方法
    if params.method == "normalized":
        scenarios = run_normalized_dcf(metrics, params)
    elif params.method == "ddm":
        scenarios = run_ddm(metrics, params)
    else:
        scenarios = run_standard_dcf(metrics, params)

    if scenarios[0] is None:
        result.error = "无法获取足够的财务数据进行估值"
        return result

    result.conservative, result.base, result.optimistic = scenarios

    for s in [result.conservative, result.base, result.optimistic]:
        if params.method == "ddm":
            s.equity_value = s.enterprise_value
        else:
            s.equity_value = s.enterprise_value - (metrics.net_debt or 0)

        s.equity_value = max(s.equity_value, 0)
        s.value_per_share = s.equity_value / metrics.shares_outstanding
        if metrics.current_price > 0:
            s.margin_of_safety = (s.value_per_share - metrics.current_price) / s.value_per_share

    result.buy_price = result.conservative.value_per_share
    result.fair_price = result.base.value_per_share
    result.upside_price = result.optimistic.value_per_share

    # 推荐
    price = metrics.current_price
    if price <= result.buy_price * 0.9:
        result.recommendation = "🟢 强烈买入 — 保守估值下仍有显著安全边际"
    elif price <= result.buy_price:
        result.recommendation = "🟢 买入 — 价格低于保守估值"
    elif price <= result.fair_price * 0.95:
        result.recommendation = "🟡 可考虑 — 价格接近合理但未到保守底线"
    elif price <= result.fair_price:
        result.recommendation = "🟡 持有/观望 — 价格接近合理估值"
    elif price <= result.upside_price:
        result.recommendation = "🟠 偏贵 — 需要乐观假设才能支撑当前价格"
    else:
        result.recommendation = "🔴 过贵 — 超出乐观估值，无安全边际"

    return result


# ---------------------------------------------------------------------------
# 6. 输出
# ---------------------------------------------------------------------------

def print_valuation(v: ValuationResult):
    print(f"\n{'━' * 85}")
    print(f"  {v.ticker} | {v.name} | {v.sector}")
    print(f"  估值方法：{_method_name(v.method)}")
    print(f"{'━' * 85}")

    if v.error:
        print(f"  ❌ {v.error}")
        return

    m = v.metrics
    print(f"\n  📊 基础数据：")
    print(f"     当前股价        ${v.current_price:.2f}")
    print(f"     市值            ${(v.current_price * v.shares) / 1e9:.1f}B")
    if m.latest_fcf:
        print(f"     最近年FCF       ${m.latest_fcf / 1e9:.1f}B")
    if m.fcf_values and len(m.fcf_values) > 1:
        avg = np.mean(m.fcf_values)
        print(f"     历史平均FCF     ${avg / 1e9:.1f}B（{len(m.fcf_values)}年）")
    if m.fcf_growth_avg is not None:
        print(f"     FCF历史增速     {m.fcf_growth_avg:+.1%}")
    if m.revenue_growth_avg is not None:
        print(f"     营收历史增速    {m.revenue_growth_avg:+.1%}")
    if m.fcf_margin_avg is not None:
        print(f"     FCF利润率均值   {m.fcf_margin_avg:.1%}")
    print(f"     净负债          ${v.net_debt / 1e9:+.1f}B（{'净现金' if v.net_debt < 0 else '净负债'}）")

    # 三情景对比
    print(f"\n  📈 三情景估值：")
    print(f"     {'':>16} {'保守':>12} {'基准':>12} {'乐观':>12}")
    print(f"     {'─' * 52}")

    c, b, o = v.conservative, v.base, v.optimistic
    print(f"     {'折现率':>16} {c.discount_rate:>11.0%} {b.discount_rate:>11.0%} {o.discount_rate:>11.0%}")
    print(f"     {'终值增长率':>16} {c.terminal_growth:>11.1%} {b.terminal_growth:>11.1%} {o.terminal_growth:>11.1%}")

    if c.growth_rates:
        print(f"     {'第1年增速':>16} {c.growth_rates[0]:>11.1%} {b.growth_rates[0]:>11.1%} {o.growth_rates[0]:>11.1%}")
        mid = len(c.growth_rates) // 2
        print(f"     {'第{0}年增速'.format(mid+1):>16} {c.growth_rates[mid]:>11.1%} {b.growth_rates[mid]:>11.1%} {o.growth_rates[mid]:>11.1%}")
        print(f"     {'第10年增速':>16} {c.growth_rates[-1]:>11.1%} {b.growth_rates[-1]:>11.1%} {o.growth_rates[-1]:>11.1%}")

    print(f"     {'─' * 52}")
    print(f"     {'PV(FCFs)':>16} ${c.pv_fcfs/1e9:>10.1f}B ${b.pv_fcfs/1e9:>10.1f}B ${o.pv_fcfs/1e9:>10.1f}B")
    print(f"     {'PV(终值)':>16} ${c.pv_terminal/1e9:>10.1f}B ${b.pv_terminal/1e9:>10.1f}B ${o.pv_terminal/1e9:>10.1f}B")
    print(f"     {'企业价值':>16} ${c.enterprise_value/1e9:>10.1f}B ${b.enterprise_value/1e9:>10.1f}B ${o.enterprise_value/1e9:>10.1f}B")
    print(f"     {'股权价值':>16} ${c.equity_value/1e9:>10.1f}B ${b.equity_value/1e9:>10.1f}B ${o.equity_value/1e9:>10.1f}B")
    print(f"     {'─' * 52}")
    print(f"     {'每股内在价值':>16} ${c.value_per_share:>10.0f}  ${b.value_per_share:>10.0f}  ${o.value_per_share:>10.0f}")
    print(f"     {'安全边际':>16} {c.margin_of_safety:>10.0%}  {b.margin_of_safety:>10.0%}  {o.margin_of_safety:>10.0%}")

    # 价格标尺
    print(f"\n  💰 价格标尺：")
    prices = [
        ("强烈买入区", v.buy_price * 0.9),
        ("买入价（保守估值）", v.buy_price),
        (">>> 当前股价 <<<", v.current_price),
        ("合理价（基准估值）", v.fair_price),
        ("乐观估值", v.upside_price),
    ]
    prices.sort(key=lambda x: x[1])

    for label, price in prices:
        marker = " 👈" if "当前" in label else ""
        print(f"     ${price:>8.0f}  {label}{marker}")

    # 推荐
    print(f"\n  🎯 {v.recommendation}")


def print_summary(results: list[ValuationResult]):
    print(f"\n\n{'=' * 85}")
    print(f"  DCF 估值汇总  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 85}")

    valid = [r for r in results if not r.error]

    print(f"\n  {'代码':<7} {'行业':<8} {'方法':<6} {'股价':>8} {'保守估值':>9} {'基准估值':>9} {'安全边际':>8} {'建议'}")
    print(f"  {'─' * 80}")

    for r in sorted(valid, key=lambda x: x.conservative.margin_of_safety, reverse=True):
        method_short = {"standard": "FCF", "normalized": "正常化", "ddm": "DDM"}[r.method]
        print(f"  {r.ticker:<7} {r.sector[:6]:<8} {method_short:<6} "
              f"${r.current_price:>7.0f} ${r.buy_price:>8.0f} ${r.fair_price:>8.0f} "
              f"{r.conservative.margin_of_safety:>7.0%}  "
              f"{r.recommendation.split('—')[0].strip()}")

    # 买入建议排序
    buys = [r for r in valid if "买入" in r.recommendation]
    considers = [r for r in valid if "可考虑" in r.recommendation]
    holds = [r for r in valid if "持有" in r.recommendation or "偏贵" in r.recommendation]
    avoids = [r for r in valid if "过贵" in r.recommendation]

    print(f"\n  按行动分组：")
    if buys:
        print(f"  🟢 买入：{', '.join(r.ticker for r in buys)}")
    if considers:
        print(f"  🟡 可考虑：{', '.join(r.ticker for r in considers)}")
    if holds:
        print(f"  🟠 观望/偏贵：{', '.join(r.ticker for r in holds)}")
    if avoids:
        print(f"  🔴 过贵：{', '.join(r.ticker for r in avoids)}")

    if buys:
        print(f"\n  如果构建组合（等权重分配），建议：")
        weight = 1.0 / len(buys)
        for r in buys:
            mos = r.conservative.margin_of_safety
            print(f"    {r.ticker:<7} 仓位 {weight:.0%}  |  "
                  f"买入价 ≤ ${r.buy_price:.0f}  |  "
                  f"目标价 ${r.fair_price:.0f}  |  "
                  f"安全边际 {mos:.0%}")

    print(f"\n  ⚠️ 重要提醒：")
    print(f"     1. DCF 对假设极其敏感——增长率差2%，估值可能差50%")
    print(f"     2. 保守情景 = 买入依据，乐观情景只用来评估上行空间")
    print(f"     3. 这些是定量筛选结果，不替代对商业模式的定性判断")
    print(f"     4. 此工具不构成投资建议\n")


def _method_name(method: str) -> str:
    return {
        "standard": "标准FCF折现（自由现金流 → 折现 → 减去净负债）",
        "normalized": "正常化FCF折现（用历史平均FCF，避免周期高/低点扭曲）",
        "ddm": "股息折现+超额收益（金融公司用可分配利润替代FCF）",
    }.get(method, method)


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DCF 估值器")
    parser.add_argument("tickers", nargs="*", help="要估值的股票代码")
    parser.add_argument("--survivors", action="store_true",
                        help="直接估值排雷存活的8只股票")
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.survivors:
        tickers = SURVIVORS

    if not tickers:
        print("用法：python3 dcf_valuation.py ADBE DVN TROW")
        print("  或：python3 dcf_valuation.py --survivors")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print(f"  DCF 估值器 | {len(tickers)} 只股票")
    print(f"{'=' * 50}")

    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\r  [{i}/{len(tickers)}] 正在估值 {ticker:<7}", end="", flush=True)
        try:
            result = valuate(ticker)
            results.append(result)
        except Exception as e:
            results.append(ValuationResult(
                ticker=ticker, name="", sector="", method="",
                current_price=0, shares=0, net_debt=0, error=str(e),
            ))
        if i % 3 == 0:
            time.sleep(0.5)

    print()

    for r in results:
        print_valuation(r)

    print_summary(results)


if __name__ == "__main__":
    main()
