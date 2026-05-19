"""
排雷器（Mine Sweeper）
=====================
对筛选器输出的候选股进行深度排雷检查。
按行业分别设计检测逻辑，不同产业有不同阈值和专属检查项。

用法：
  python3 minesweeper.py AAPL MSFT MRK          # 检查指定股票
  python3 minesweeper.py --from-csv results.csv  # 从筛选器结果读取
  python3 minesweeper.py --from-csv results.csv --top 10  # 只查前10
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# 1. 行业配置：每个行业的阈值和专属检查项
# ---------------------------------------------------------------------------

@dataclass
class IndustryConfig:
    """某个行业的排雷参数"""
    name: str
    # --- 基础检查阈值（不同行业标准不同）---
    cash_profit_ratio_min: float = 0.7      # 经营现金流/净利润 最低要求
    receivable_growth_ratio: float = 1.5    # 应收增速/营收增速 最高容忍倍数
    inventory_growth_ratio: float = 1.5     # 存货增速/营收增速 最高容忍倍数
    gaap_vs_nongaap_max: float = 0.20       # GAAP vs Non-GAAP 利润差距上限
    insider_net_sell_warn: bool = True       # 内部人净卖出是否告警
    max_customer_concentration: float = 0.30  # 前几大客户占收入上限
    buyback_debt_funded_warn: bool = True    # 举债回购是否告警
    # --- 是否跳过某些不适用的检查 ---
    skip_inventory_check: bool = False       # 金融/服务业没有存货
    skip_cash_profit_check: bool = False     # 金融业现金流结构不同
    # --- 行业专属检查 ---
    extra_checks: list = field(default_factory=list)


# 各行业配置
INDUSTRY_CONFIGS = {
    "Energy": IndustryConfig(
        name="能源",
        cash_profit_ratio_min=0.6,       # 能源行业资本支出大，标准放宽
        inventory_growth_ratio=2.0,      # 能源存货受油价影响波动大
        extra_checks=["cycle_position", "reserve_depletion", "capex_discipline"],
    ),
    "Basic Materials": IndustryConfig(
        name="基础材料",
        cash_profit_ratio_min=0.6,
        inventory_growth_ratio=2.0,
        extra_checks=["cycle_position", "commodity_dependency"],
    ),
    "Technology": IndustryConfig(
        name="科技",
        skip_inventory_check=True,       # 软件公司没啥存货
        cash_profit_ratio_min=0.8,       # 科技公司应该现金流很好
        extra_checks=["sbc_dilution", "rd_trend", "deferred_revenue_trend"],
    ),
    "Healthcare": IndustryConfig(
        name="医疗健康",
        cash_profit_ratio_min=0.7,
        extra_checks=["revenue_concentration", "pipeline_dependency", "patent_cliff"],
    ),
    "Financial Services": IndustryConfig(
        name="金融",
        skip_inventory_check=True,
        skip_cash_profit_check=True,     # 银行现金流量表结构完全不同
        extra_checks=["book_value_trend", "credit_quality_proxy", "capital_ratio"],
    ),
    "Consumer Cyclical": IndustryConfig(
        name="可选消费",
        cash_profit_ratio_min=0.7,
        inventory_growth_ratio=1.3,      # 零售业存货管理很重要，标准更严
        extra_checks=["same_store_signal", "consumer_leverage"],
    ),
    "Consumer Defensive": IndustryConfig(
        name="必选消费",
        cash_profit_ratio_min=0.8,       # 消费品现金流应该很稳定
        inventory_growth_ratio=1.3,
        extra_checks=["margin_stability", "brand_power"],
    ),
    "Communication Services": IndustryConfig(
        name="通信服务",
        skip_inventory_check=True,
        extra_checks=["subscriber_trend", "content_spend_ratio"],
    ),
    "Industrials": IndustryConfig(
        name="工业",
        extra_checks=["backlog_trend", "capex_cycle"],
    ),
    "Real Estate": IndustryConfig(
        name="房地产/REITs",
        skip_cash_profit_check=True,     # REITs用FFO而非净利润
        skip_inventory_check=True,
        extra_checks=["ffo_trend", "debt_maturity", "occupancy_proxy"],
    ),
    "Utilities": IndustryConfig(
        name="公用事业",
        cash_profit_ratio_min=0.6,       # 公用事业资本支出大
        skip_inventory_check=True,
        extra_checks=["regulatory_rate_base", "dividend_coverage"],
    ),
}

DEFAULT_CONFIG = IndustryConfig(name="其他", extra_checks=[])


def get_config(sector: str) -> IndustryConfig:
    return INDUSTRY_CONFIGS.get(sector, DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# 2. 检查结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str           # 检查项名称
    passed: bool        # 是否通过
    severity: str       # "🔴" 严重 / "🟡" 警告 / "⚪" 信息
    detail: str         # 具体说明
    value: str = ""     # 关键数值


@dataclass
class StockReport:
    ticker: str
    name: str
    sector: str
    industry: str
    config_name: str
    checks: list = field(default_factory=list)
    error: str = ""

    @property
    def red_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "🔴")

    @property
    def yellow_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "🟡")

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def verdict(self) -> str:
        if self.error:
            return "❓ 数据不足"
        if self.red_count >= 2:
            return "🚫 淘汰"
        if self.red_count == 1 and self.yellow_count >= 2:
            return "🚫 淘汰"
        if self.red_count == 1:
            return "⚠️ 高风险"
        if self.yellow_count >= 3:
            return "⚠️ 需关注"
        if self.yellow_count >= 1:
            return "🔍 可研究"
        return "✅ 进入深度研究"


# ---------------------------------------------------------------------------
# 3. 数据获取层
# ---------------------------------------------------------------------------

@dataclass
class StockData:
    """一只股票的全部原始数据"""
    ticker: str
    info: dict = field(default_factory=dict)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    q_income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    q_balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    q_cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    insider_txns: pd.DataFrame = field(default_factory=pd.DataFrame)


def fetch_stock_data(ticker: str) -> StockData:
    stock = yf.Ticker(ticker)
    data = StockData(ticker=ticker)

    data.info = stock.info or {}

    try:
        data.income_stmt = stock.financials
    except Exception:
        pass
    try:
        data.balance_sheet = stock.balance_sheet
    except Exception:
        pass
    try:
        data.cashflow = stock.cashflow
    except Exception:
        pass
    try:
        data.q_income_stmt = stock.quarterly_financials
    except Exception:
        pass
    try:
        data.q_balance_sheet = stock.quarterly_balance_sheet
    except Exception:
        pass
    try:
        data.q_cashflow = stock.quarterly_cashflow
    except Exception:
        pass
    try:
        data.insider_txns = stock.insider_transactions
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------------
# 4. 工具函数
# ---------------------------------------------------------------------------

def safe_get_row(df: pd.DataFrame, labels: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            return df.loc[label]
    return None


def yoy_growth(series: pd.Series) -> float | None:
    """计算最近一年的同比增长率。财报列从新到旧排列。"""
    vals = series.dropna()
    if len(vals) < 2:
        return None
    latest, previous = vals.iloc[0], vals.iloc[1]
    if previous == 0:
        return None
    return (latest - previous) / abs(previous)


def multi_year_trend(series: pd.Series, years: int = 3) -> str | None:
    """判断多年趋势：连续增长/下降/波动"""
    vals = series.dropna()
    if len(vals) < years:
        return None
    recent = vals.iloc[:years].values  # 从新到旧
    diffs = np.diff(recent)  # 新-旧的差值（注意方向是反的）
    if all(d < 0 for d in diffs):  # 每年都比前一年大
        return "连续增长"
    if all(d > 0 for d in diffs):  # 每年都比前一年小
        return "连续下降"
    return "波动"


def fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:+.1f}%"


def fmt_ratio(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


# ---------------------------------------------------------------------------
# 5. 通用检查（所有行业都跑，但阈值按行业调整）
# ---------------------------------------------------------------------------

def check_cash_profit_ratio(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查1：经营现金流 / 净利润"""
    if config.skip_cash_profit_check:
        return CheckResult(
            name="现金利润比",
            passed=True, severity="⚪",
            detail=f"{config.name}行业不适用此检查（现金流结构不同）",
        )

    cfo_row = safe_get_row(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    ni_row = safe_get_row(data.income_stmt, ["Net Income", "Net Income Common Stockholders"])

    if cfo_row is None or ni_row is None:
        return CheckResult(name="现金利润比", passed=True, severity="⚪",
                           detail="数据不足，跳过", value="N/A")

    cfo_latest = cfo_row.dropna().iloc[0] if len(cfo_row.dropna()) > 0 else None
    ni_latest = ni_row.dropna().iloc[0] if len(ni_row.dropna()) > 0 else None

    if cfo_latest is None or ni_latest is None or ni_latest <= 0:
        return CheckResult(name="现金利润比", passed=True, severity="⚪",
                           detail="净利润为负或数据缺失", value="N/A")

    ratio = cfo_latest / ni_latest
    passed = ratio >= config.cash_profit_ratio_min
    severity = "🔴" if ratio < 0.5 else ("🟡" if not passed else "⚪")

    return CheckResult(
        name="现金利润比",
        passed=passed,
        severity=severity,
        detail=f"经营现金流/净利润 = {ratio:.2f}（{config.name}行业要求 ≥ {config.cash_profit_ratio_min}）"
               + ("" if passed else " → 利润含水量高，可能有虚增"),
        value=fmt_ratio(ratio),
    )


def check_receivable_growth(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查2：应收账款增速 vs 营收增速"""
    ar_row = safe_get_row(data.balance_sheet, ["Net Receivables", "Accounts Receivable"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if ar_row is None or rev_row is None:
        return CheckResult(name="应收账款增速", passed=True, severity="⚪",
                           detail="数据不足，跳过", value="N/A")

    ar_growth = yoy_growth(ar_row)
    rev_growth = yoy_growth(rev_row)

    if ar_growth is None or rev_growth is None:
        return CheckResult(name="应收账款增速", passed=True, severity="⚪",
                           detail="历史数据不足", value="N/A")

    if rev_growth <= 0:
        if ar_growth > 0.1:
            return CheckResult(
                name="应收账款增速", passed=False, severity="🔴",
                detail=f"营收下降({fmt_pct(rev_growth)})但应收在涨({fmt_pct(ar_growth)}) → 赊账冲业绩或收款困难",
                value=f"AR {fmt_pct(ar_growth)} vs Rev {fmt_pct(rev_growth)}",
            )
        return CheckResult(name="应收账款增速", passed=True, severity="⚪",
                           detail=f"营收 {fmt_pct(rev_growth)}，应收 {fmt_pct(ar_growth)}")

    ratio = ar_growth / rev_growth if rev_growth > 0 else 0
    passed = ratio <= config.receivable_growth_ratio

    severity = "🔴" if ratio > 2.0 else ("🟡" if not passed else "⚪")

    return CheckResult(
        name="应收账款增速",
        passed=passed,
        severity=severity,
        detail=f"应收增速 {fmt_pct(ar_growth)} vs 营收增速 {fmt_pct(rev_growth)}，"
               f"倍数 {ratio:.1f}x（上限 {config.receivable_growth_ratio}x）"
               + ("" if passed else " → 可能在赊账冲业绩"),
        value=f"{ratio:.1f}x",
    )


def check_inventory_growth(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查3：存货增速 vs 营收增速"""
    if config.skip_inventory_check:
        return CheckResult(name="存货异常", passed=True, severity="⚪",
                           detail=f"{config.name}行业不适用存货检查")

    inv_row = safe_get_row(data.balance_sheet, ["Inventory", "Net Inventory"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if inv_row is None or rev_row is None:
        return CheckResult(name="存货异常", passed=True, severity="⚪",
                           detail="无存货数据（可能是服务型公司）", value="N/A")

    inv_growth = yoy_growth(inv_row)
    rev_growth = yoy_growth(rev_row)

    if inv_growth is None or rev_growth is None:
        return CheckResult(name="存货异常", passed=True, severity="⚪",
                           detail="历史数据不足", value="N/A")

    if rev_growth <= 0 and inv_growth > 0.15:
        return CheckResult(
            name="存货异常", passed=False, severity="🔴",
            detail=f"营收下降({fmt_pct(rev_growth)})但存货在涨({fmt_pct(inv_growth)}) → 东西卖不动",
            value=f"Inv {fmt_pct(inv_growth)} vs Rev {fmt_pct(rev_growth)}",
        )

    if rev_growth > 0:
        ratio = inv_growth / rev_growth
        passed = ratio <= config.inventory_growth_ratio
        severity = "🔴" if ratio > 2.5 else ("🟡" if not passed else "⚪")
    else:
        passed = inv_growth < 0.1
        ratio = None
        severity = "⚪" if passed else "🟡"

    return CheckResult(
        name="存货异常",
        passed=passed,
        severity=severity,
        detail=f"存货增速 {fmt_pct(inv_growth)} vs 营收增速 {fmt_pct(rev_growth)}"
               + (f"，倍数 {ratio:.1f}x" if ratio else "")
               + ("" if passed else f" → {config.name}行业存货增速过快，可能滞销"),
        value=f"Inv {fmt_pct(inv_growth)}",
    )


def check_gaap_vs_actual(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查4：一次性收益占比（用营业利润 vs 净利润的差距近似）"""
    oi_row = safe_get_row(data.income_stmt, ["Operating Income", "EBIT"])
    ni_row = safe_get_row(data.income_stmt, ["Net Income", "Net Income Common Stockholders"])

    if oi_row is None or ni_row is None:
        return CheckResult(name="一次性收益", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    oi = oi_row.dropna().iloc[0] if len(oi_row.dropna()) > 0 else None
    ni = ni_row.dropna().iloc[0] if len(ni_row.dropna()) > 0 else None

    if oi is None or ni is None or oi == 0:
        return CheckResult(name="一次性收益", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    diff_ratio = abs(ni - oi) / abs(oi)

    if ni > oi * 1.5:
        return CheckResult(
            name="一次性收益", passed=False, severity="🔴",
            detail=f"净利润(${ni/1e9:.1f}B)远超营业利润(${oi/1e9:.1f}B)，差距{diff_ratio:.0%} → 可能含大额非经常性收益",
            value=f"{diff_ratio:.0%}",
        )

    passed = diff_ratio <= config.gaap_vs_nongaap_max
    severity = "🟡" if not passed else "⚪"

    return CheckResult(
        name="一次性收益",
        passed=passed,
        severity=severity,
        detail=f"营业利润 ${oi/1e9:.1f}B vs 净利润 ${ni/1e9:.1f}B，差距 {diff_ratio:.0%}"
               + ("" if passed else " → 非经常性项目占比偏高"),
        value=f"{diff_ratio:.0%}",
    )


def check_insider_activity(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查5：内部人买卖"""
    if not config.insider_net_sell_warn:
        return CheckResult(name="内部人交易", passed=True, severity="⚪",
                           detail="此行业不重点关注内部人交易")

    txns = data.insider_txns
    if txns is None or txns.empty:
        return CheckResult(name="内部人交易", passed=True, severity="⚪",
                           detail="无近期内部人交易数据", value="N/A")

    text_col = None
    for col in ["Text", "Transaction", "text", "transaction"]:
        if col in txns.columns:
            text_col = col
            break

    if text_col is None:
        return CheckResult(name="内部人交易", passed=True, severity="⚪",
                           detail="交易数据格式无法解析", value="N/A")

    recent = txns.head(20)
    txn_texts = recent[text_col].astype(str).str.lower()
    sales = txn_texts.str.contains("sale|sold|sell", na=False).sum()
    buys = txn_texts.str.contains("purchase|bought|buy|acquisition", na=False).sum()

    if sales > 0 and buys == 0 and sales >= 3:
        return CheckResult(
            name="内部人交易", passed=False, severity="🔴",
            detail=f"近期 {sales} 笔卖出、{buys} 笔买入 → 内部人在大量抛售，无人买入",
            value=f"卖{sales}/买{buys}",
        )

    if sales > buys * 3 and sales >= 4:
        return CheckResult(
            name="内部人交易", passed=False, severity="🟡",
            detail=f"近期 {sales} 笔卖出、{buys} 笔买入 → 卖出明显多于买入",
            value=f"卖{sales}/买{buys}",
        )

    return CheckResult(
        name="内部人交易", passed=True, severity="⚪",
        detail=f"近期 {sales} 笔卖出、{buys} 笔买入，无异常",
        value=f"卖{sales}/买{buys}",
    )


def check_buyback_quality(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查6：回购是否健康（用自由现金流还是借钱）"""
    cfo_row = safe_get_row(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex_row = safe_get_row(data.cashflow, ["Capital Expenditure", "Capital Expenditures"])
    buyback_row = safe_get_row(data.cashflow, [
        "Repurchase Of Capital Stock", "Common Stock Repurchased",
        "Repurchase of Common and Preferred Stock",
    ])

    if cfo_row is None or buyback_row is None:
        return CheckResult(name="回购质量", passed=True, severity="⚪",
                           detail="无回购或现金流数据", value="N/A")

    cfo = cfo_row.dropna().iloc[0] if len(cfo_row.dropna()) > 0 else 0
    capex = abs(capex_row.dropna().iloc[0]) if capex_row is not None and len(capex_row.dropna()) > 0 else 0
    buyback = abs(buyback_row.dropna().iloc[0]) if len(buyback_row.dropna()) > 0 else 0

    if buyback < 1e6:
        return CheckResult(name="回购质量", passed=True, severity="⚪",
                           detail="近期无显著回购", value="无回购")

    fcf = cfo - capex

    if fcf <= 0:
        return CheckResult(
            name="回购质量", passed=False, severity="🔴",
            detail=f"自由现金流为负(${fcf/1e9:.1f}B)但仍在回购(${buyback/1e9:.1f}B) → 在借钱回购，不可持续",
            value=f"FCF ${fcf/1e9:.1f}B, 回购 ${buyback/1e9:.1f}B",
        )

    if buyback > fcf * 1.2:
        return CheckResult(
            name="回购质量", passed=False, severity="🟡",
            detail=f"回购金额(${buyback/1e9:.1f}B)超过自由现金流(${fcf/1e9:.1f}B) → 部分靠举债",
            value=f"回购/FCF = {buyback/fcf:.1f}x",
        )

    return CheckResult(
        name="回购质量", passed=True, severity="⚪",
        detail=f"回购 ${buyback/1e9:.1f}B，自由现金流 ${fcf/1e9:.1f}B，回购资金健康",
        value=f"回购/FCF = {buyback/fcf:.1%}",
    )


def check_revenue_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查7：营收是否在萎缩（行业衰退信号）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if rev_row is None:
        return CheckResult(name="营收趋势", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    trend = multi_year_trend(rev_row, years=3)
    growth = yoy_growth(rev_row)

    if trend == "连续下降":
        return CheckResult(
            name="营收趋势", passed=False, severity="🔴",
            detail=f"营收连续多年下降（最近一年 {fmt_pct(growth)}）→ 可能面临行业结构性衰退",
            value=f"{trend} {fmt_pct(growth)}",
        )

    if growth is not None and growth < -0.05:
        return CheckResult(
            name="营收趋势", passed=False, severity="🟡",
            detail=f"最近一年营收下降 {fmt_pct(growth)}",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="营收趋势", passed=True, severity="⚪",
        detail=f"营收趋势：{trend or 'N/A'}，最近一年 {fmt_pct(growth)}",
        value=fmt_pct(growth),
    )


def check_debt_level(data: StockData, config: IndustryConfig) -> CheckResult:
    """检查8：负债水平"""
    debt = data.info.get("totalDebt")
    equity = data.info.get("totalStockholderEquity")
    ebitda = data.info.get("ebitda")

    if debt is None:
        return CheckResult(name="负债水平", passed=True, severity="⚪",
                           detail="无负债数据", value="N/A")

    results = []

    # Debt/Equity
    if equity and equity > 0:
        de_ratio = debt / equity
        results.append(f"D/E={de_ratio:.1f}")
        if de_ratio > 5:
            return CheckResult(
                name="负债水平", passed=False, severity="🔴",
                detail=f"债务/股东权益 = {de_ratio:.1f}x → 杠杆极高",
                value=f"D/E {de_ratio:.1f}x",
            )
    elif equity and equity < 0:
        results.append("股东权益为负")

    # Debt/EBITDA
    if ebitda and ebitda > 0:
        de_ebitda = debt / ebitda
        results.append(f"Debt/EBITDA={de_ebitda:.1f}")
        if de_ebitda > 5:
            return CheckResult(
                name="负债水平", passed=False, severity="🔴",
                detail=f"债务/EBITDA = {de_ebitda:.1f}x → 5年利润才能还清债务",
                value=f"Debt/EBITDA {de_ebitda:.1f}x",
            )
        if de_ebitda > 3:
            return CheckResult(
                name="负债水平", passed=False, severity="🟡",
                detail=f"债务/EBITDA = {de_ebitda:.1f}x → 偏高",
                value=f"Debt/EBITDA {de_ebitda:.1f}x",
            )

    return CheckResult(
        name="负债水平", passed=True, severity="⚪",
        detail=f"负债指标：{', '.join(results) if results else '数据有限'}",
        value=", ".join(results),
    )


# ---------------------------------------------------------------------------
# 6. 行业专属检查
# ---------------------------------------------------------------------------

# --- 能源 / 基础材料：周期股专属 ---

def check_cycle_position(data: StockData, config: IndustryConfig) -> CheckResult:
    """能源/材料：当前处于景气周期什么位置"""
    margin_row = safe_get_row(data.income_stmt, ["Operating Income", "EBIT"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if margin_row is None or rev_row is None or len(margin_row.dropna()) < 3:
        return CheckResult(name="⚡周期位置", passed=True, severity="⚪",
                           detail="历史数据不足以判断周期位置", value="N/A")

    margins = []
    for i in range(min(len(margin_row.dropna()), len(rev_row.dropna()))):
        r = rev_row.dropna().iloc[i]
        m = margin_row.dropna().iloc[i]
        if r > 0:
            margins.append(m / r)

    if len(margins) < 3:
        return CheckResult(name="⚡周期位置", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    current_margin = margins[0]
    avg_margin = np.mean(margins)
    max_margin = max(margins)

    if current_margin > avg_margin * 1.5:
        position = "接近顶部"
        return CheckResult(
            name="⚡周期位置", passed=False, severity="🔴",
            detail=f"当前利润率 {current_margin:.1%} 远超历史均值 {avg_margin:.1%} → "
                   f"⚠️ 周期股陷阱：利润在高位 = PE在低位 ≠ 真的便宜",
            value=position,
        )

    if current_margin > avg_margin * 1.2:
        return CheckResult(
            name="⚡周期位置", passed=False, severity="🟡",
            detail=f"当前利润率 {current_margin:.1%} 高于历史均值 {avg_margin:.1%} → 可能偏向周期高位",
            value="偏高位",
        )

    return CheckResult(
        name="⚡周期位置", passed=True, severity="⚪",
        detail=f"当前利润率 {current_margin:.1%}，历史均值 {avg_margin:.1%}，处于正常范围",
        value="正常",
    )


def check_reserve_depletion(data: StockData, config: IndustryConfig) -> CheckResult:
    """能源：资本支出纪律（是否在维持产能）"""
    capex_row = safe_get_row(data.cashflow, ["Capital Expenditure", "Capital Expenditures"])
    dep_row = safe_get_row(data.cashflow, ["Depreciation And Amortization", "Depreciation & Amortization"])

    if capex_row is None or dep_row is None:
        dep_row = safe_get_row(data.income_stmt, ["Reconciled Depreciation", "Depreciation And Amortization"])

    if capex_row is None or dep_row is None:
        return CheckResult(name="⚡资本支出纪律", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    capex = abs(capex_row.dropna().iloc[0]) if len(capex_row.dropna()) > 0 else 0
    dep = abs(dep_row.dropna().iloc[0]) if len(dep_row.dropna()) > 0 else 0

    if dep == 0:
        return CheckResult(name="⚡资本支出纪律", passed=True, severity="⚪",
                           detail="折旧数据为零", value="N/A")

    ratio = capex / dep

    if ratio < 0.7:
        return CheckResult(
            name="⚡资本支出纪律", passed=False, severity="🟡",
            detail=f"资本支出/折旧 = {ratio:.2f} → 投资不足以维持现有产能，"
                   f"短期利润好看但长期产能在萎缩",
            value=f"{ratio:.2f}x",
        )

    return CheckResult(
        name="⚡资本支出纪律", passed=True, severity="⚪",
        detail=f"资本支出/折旧 = {ratio:.2f}，在合理维持产能",
        value=f"{ratio:.2f}x",
    )


def check_capex_discipline(data: StockData, config: IndustryConfig) -> CheckResult:
    """能源：是否在周期高点盲目扩产"""
    capex_row = safe_get_row(data.cashflow, ["Capital Expenditure", "Capital Expenditures"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if capex_row is None or rev_row is None:
        return CheckResult(name="⚡扩产风险", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    capex_growth = yoy_growth(capex_row.abs())
    rev_growth = yoy_growth(rev_row)

    if capex_growth is None:
        return CheckResult(name="⚡扩产风险", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    if capex_growth > 0.4:
        return CheckResult(
            name="⚡扩产风险", passed=False, severity="🟡",
            detail=f"资本支出同比增长 {fmt_pct(capex_growth)} → 周期高位大幅扩产有风险",
            value=fmt_pct(capex_growth),
        )

    return CheckResult(
        name="⚡扩产风险", passed=True, severity="⚪",
        detail=f"资本支出增速 {fmt_pct(capex_growth)}，未见激进扩产",
        value=fmt_pct(capex_growth),
    )


def check_commodity_dependency(data: StockData, config: IndustryConfig) -> CheckResult:
    """基础材料：利润波动性（衡量商品价格依赖度）"""
    ni_row = safe_get_row(data.income_stmt, ["Net Income", "Net Income Common Stockholders"])
    if ni_row is None or len(ni_row.dropna()) < 3:
        return CheckResult(name="⚡商品价格依赖", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    profits = ni_row.dropna().values[:4].astype(float)
    if np.mean(np.abs(profits)) == 0:
        return CheckResult(name="⚡商品价格依赖", passed=True, severity="⚪",
                           detail="利润为零", value="N/A")

    cv = np.std(profits) / np.mean(np.abs(profits))

    if cv > 0.6:
        return CheckResult(
            name="⚡商品价格依赖", passed=False, severity="🟡",
            detail=f"利润波动系数 {cv:.2f} → 利润高度依赖商品价格，"
                   f"当前利润可能处于非正常高位",
            value=f"CV={cv:.2f}",
        )

    return CheckResult(
        name="⚡商品价格依赖", passed=True, severity="⚪",
        detail=f"利润波动系数 {cv:.2f}，波动在可接受范围",
        value=f"CV={cv:.2f}",
    )


# --- 科技行业专属 ---

def check_sbc_dilution(data: StockData, config: IndustryConfig) -> CheckResult:
    """科技：股权激励（SBC）对股东的稀释"""
    sbc_row = safe_get_row(data.cashflow, [
        "Stock Based Compensation", "Share Based Compensation",
    ])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    ni_row = safe_get_row(data.income_stmt, ["Net Income", "Net Income Common Stockholders"])

    if sbc_row is None or rev_row is None:
        return CheckResult(name="💻股权激励稀释", passed=True, severity="⚪",
                           detail="无SBC数据", value="N/A")

    sbc = sbc_row.dropna().iloc[0] if len(sbc_row.dropna()) > 0 else 0
    rev = rev_row.dropna().iloc[0] if len(rev_row.dropna()) > 0 else 0
    ni = ni_row.dropna().iloc[0] if ni_row is not None and len(ni_row.dropna()) > 0 else 0

    if rev == 0:
        return CheckResult(name="💻股权激励稀释", passed=True, severity="⚪",
                           detail="营收为零", value="N/A")

    sbc_rev_pct = sbc / rev
    sbc_ni_pct = sbc / ni if ni > 0 else None

    detail_parts = [f"SBC/营收 = {sbc_rev_pct:.1%}"]
    if sbc_ni_pct is not None:
        detail_parts.append(f"SBC/净利润 = {sbc_ni_pct:.1%}")

    if sbc_rev_pct > 0.15:
        return CheckResult(
            name="💻股权激励稀释", passed=False, severity="🔴",
            detail=f"{', '.join(detail_parts)} → SBC占营收超15%，股东被严重稀释，"
                   f"公司在用你的钱给员工发工资",
            value=f"SBC/Rev {sbc_rev_pct:.1%}",
        )

    if sbc_rev_pct > 0.08:
        return CheckResult(
            name="💻股权激励稀释", passed=False, severity="🟡",
            detail=f"{', '.join(detail_parts)} → SBC偏高",
            value=f"SBC/Rev {sbc_rev_pct:.1%}",
        )

    return CheckResult(
        name="💻股权激励稀释", passed=True, severity="⚪",
        detail=f"{', '.join(detail_parts)}，稀释在合理范围",
        value=f"SBC/Rev {sbc_rev_pct:.1%}",
    )


def check_rd_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """科技：研发投入趋势（砍研发 = 吃老本）"""
    rd_row = safe_get_row(data.income_stmt, ["Research And Development", "Research Development"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if rd_row is None or rev_row is None:
        return CheckResult(name="💻研发投入", passed=True, severity="⚪",
                           detail="无研发数据", value="N/A")

    rd_vals = rd_row.dropna()
    rev_vals = rev_row.dropna()
    if len(rd_vals) < 2 or len(rev_vals) < 2:
        return CheckResult(name="💻研发投入", passed=True, severity="⚪",
                           detail="历史数据不足", value="N/A")

    current_ratio = rd_vals.iloc[0] / rev_vals.iloc[0] if rev_vals.iloc[0] > 0 else 0
    prev_ratio = rd_vals.iloc[1] / rev_vals.iloc[1] if rev_vals.iloc[1] > 0 else 0

    change = current_ratio - prev_ratio

    if change < -0.03:
        return CheckResult(
            name="💻研发投入", passed=False, severity="🟡",
            detail=f"研发/营收从 {prev_ratio:.1%} 降至 {current_ratio:.1%} → 在砍研发，可能在吃老本",
            value=f"{current_ratio:.1%}",
        )

    return CheckResult(
        name="💻研发投入", passed=True, severity="⚪",
        detail=f"研发/营收 = {current_ratio:.1%}（去年 {prev_ratio:.1%}），投入稳定",
        value=f"{current_ratio:.1%}",
    )


def check_deferred_revenue_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """科技（SaaS）：递延收入趋势（预付款在涨 = 客户在续约）"""
    dr_row = safe_get_row(data.balance_sheet, [
        "Deferred Revenue", "Current Deferred Revenue",
        "Deferred Revenue Non Current",
    ])

    if dr_row is None or len(dr_row.dropna()) < 2:
        return CheckResult(name="💻递延收入", passed=True, severity="⚪",
                           detail="无递延收入数据（可能非SaaS模式）", value="N/A")

    growth = yoy_growth(dr_row)
    if growth is None:
        return CheckResult(name="💻递延收入", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    if growth < -0.10:
        return CheckResult(
            name="💻递延收入", passed=False, severity="🟡",
            detail=f"递延收入同比 {fmt_pct(growth)} → 预付款减少，客户可能不续约",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="💻递延收入", passed=True, severity="⚪",
        detail=f"递延收入同比 {fmt_pct(growth)}，客户预付款{'增长' if growth > 0 else '稳定'}",
        value=fmt_pct(growth),
    )


# --- 医疗健康专属 ---

def check_revenue_concentration(data: StockData, config: IndustryConfig) -> CheckResult:
    """医疗：产品收入集中度（单一产品依赖风险）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    rd_row = safe_get_row(data.income_stmt, ["Research And Development", "Research Development"])

    if rev_row is None:
        return CheckResult(name="🏥收入集中度", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    rev_growth = yoy_growth(rev_row)

    if rd_row is not None and len(rd_row.dropna()) > 0 and len(rev_row.dropna()) > 0:
        rd_ratio = rd_row.dropna().iloc[0] / rev_row.dropna().iloc[0]
    else:
        rd_ratio = None

    detail_parts = []
    if rev_growth is not None:
        detail_parts.append(f"营收增速 {fmt_pct(rev_growth)}")
    if rd_ratio is not None:
        detail_parts.append(f"研发/营收 {rd_ratio:.1%}")

    return CheckResult(
        name="🏥收入集中度", passed=True, severity="⚪",
        detail=f"{'，'.join(detail_parts)}。注意：需人工查10-K确认单一产品占收入比例",
        value="需人工确认",
    )


def check_pipeline_dependency(data: StockData, config: IndustryConfig) -> CheckResult:
    """医疗：研发管线依赖（研发支出趋势）"""
    rd_row = safe_get_row(data.income_stmt, ["Research And Development", "Research Development"])

    if rd_row is None or len(rd_row.dropna()) < 2:
        return CheckResult(name="🏥研发管线", passed=True, severity="⚪",
                           detail="无研发数据", value="N/A")

    rd_growth = yoy_growth(rd_row)
    trend = multi_year_trend(rd_row)

    if trend == "连续下降":
        return CheckResult(
            name="🏥研发管线", passed=False, severity="🟡",
            detail=f"研发支出连续下降 → 管线可能在枯竭，未来增长堪忧",
            value=f"{trend}",
        )

    return CheckResult(
        name="🏥研发管线", passed=True, severity="⚪",
        detail=f"研发支出趋势：{trend or 'N/A'}，同比 {fmt_pct(rd_growth)}",
        value=fmt_pct(rd_growth),
    )


def check_patent_cliff(data: StockData, config: IndustryConfig) -> CheckResult:
    """医疗：专利悬崖风险（需人工确认，这里给出提醒）"""
    return CheckResult(
        name="🏥专利悬崖",
        passed=True,
        severity="🟡",
        detail="⚠️ 药企必须人工查核心产品专利到期时间。"
               "例如 MRK 的 Keytruda 2028年到期，ABBV 的 Humira 已到期。"
               "在10-K的Risk Factors和SEC专利数据库中查证",
        value="需人工查证",
    )


# --- 金融行业专属 ---

def check_book_value_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """金融：账面价值趋势"""
    bv_row = safe_get_row(data.balance_sheet, [
        "Total Stockholder Equity", "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ])

    if bv_row is None or len(bv_row.dropna()) < 2:
        return CheckResult(name="🏦账面价值", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(bv_row)
    trend = multi_year_trend(bv_row)

    if trend == "连续下降":
        return CheckResult(
            name="🏦账面价值", passed=False, severity="🔴",
            detail=f"股东权益连续下降 → 金融公司家底在缩水",
            value=f"{trend} {fmt_pct(growth)}",
        )

    if growth is not None and growth < -0.1:
        return CheckResult(
            name="🏦账面价值", passed=False, severity="🟡",
            detail=f"股东权益同比 {fmt_pct(growth)} → 需关注减值或损失",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="🏦账面价值", passed=True, severity="⚪",
        detail=f"股东权益趋势：{trend or 'N/A'}，同比 {fmt_pct(growth)}",
        value=fmt_pct(growth),
    )


def check_credit_quality_proxy(data: StockData, config: IndustryConfig) -> CheckResult:
    """金融：信用质量代理指标（拨备/损失趋势）"""
    provision_row = safe_get_row(data.income_stmt, [
        "Provision For Loan Losses", "Credit Loss Expense",
        "Provision For Credit Losses",
    ])

    if provision_row is None or len(provision_row.dropna()) < 2:
        return CheckResult(name="🏦信用质量", passed=True, severity="⚪",
                           detail="无拨备数据（可能非银行类金融）", value="N/A")

    growth = yoy_growth(provision_row)

    if growth is not None and growth > 0.5:
        return CheckResult(
            name="🏦信用质量", passed=False, severity="🔴",
            detail=f"坏账拨备同比增长 {fmt_pct(growth)} → 贷款质量在恶化",
            value=fmt_pct(growth),
        )

    if growth is not None and growth > 0.2:
        return CheckResult(
            name="🏦信用质量", passed=False, severity="🟡",
            detail=f"坏账拨备同比增长 {fmt_pct(growth)} → 需关注",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="🏦信用质量", passed=True, severity="⚪",
        detail=f"坏账拨备同比 {fmt_pct(growth)}，信用质量稳定",
        value=fmt_pct(growth),
    )


def check_capital_ratio(data: StockData, config: IndustryConfig) -> CheckResult:
    """金融：资本充足率代理（权益/总资产）"""
    equity = data.info.get("totalStockholderEquity")
    assets = data.info.get("totalAssets")

    if equity is None or assets is None or assets == 0:
        return CheckResult(name="🏦资本充足率", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    ratio = equity / assets

    if ratio < 0.03:
        return CheckResult(
            name="🏦资本充足率", passed=False, severity="🔴",
            detail=f"权益/总资产 = {ratio:.1%} → 杠杆极高，抗风险能力弱",
            value=f"{ratio:.1%}",
        )

    if ratio < 0.06:
        return CheckResult(
            name="🏦资本充足率", passed=False, severity="🟡",
            detail=f"权益/总资产 = {ratio:.1%} → 杠杆偏高",
            value=f"{ratio:.1%}",
        )

    return CheckResult(
        name="🏦资本充足率", passed=True, severity="⚪",
        detail=f"权益/总资产 = {ratio:.1%}，资本水平正常",
        value=f"{ratio:.1%}",
    )


# --- 消费行业专属 ---

def check_same_store_signal(data: StockData, config: IndustryConfig) -> CheckResult:
    """可选消费：营收增速 vs 门店/资产增速（同店增长代理）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    asset_row = safe_get_row(data.balance_sheet, [
        "Total Non Current Assets", "Property Plant And Equipment Net",
        "Net PPE",
    ])

    if rev_row is None or asset_row is None:
        return CheckResult(name="🛍️同店增长信号", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    rev_growth = yoy_growth(rev_row)
    asset_growth = yoy_growth(asset_row)

    if rev_growth is None or asset_growth is None:
        return CheckResult(name="🛍️同店增长信号", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    if asset_growth > 0.1 and rev_growth < asset_growth * 0.5:
        return CheckResult(
            name="🛍️同店增长信号", passed=False, severity="🟡",
            detail=f"固定资产增 {fmt_pct(asset_growth)} 但营收只增 {fmt_pct(rev_growth)} → "
                   f"扩张但单店效率在降",
            value=f"Rev {fmt_pct(rev_growth)} vs Asset {fmt_pct(asset_growth)}",
        )

    return CheckResult(
        name="🛍️同店增长信号", passed=True, severity="⚪",
        detail=f"营收增速 {fmt_pct(rev_growth)} vs 资产增速 {fmt_pct(asset_growth)}，效率正常",
        value=f"Rev {fmt_pct(rev_growth)}",
    )


def check_consumer_leverage(data: StockData, config: IndustryConfig) -> CheckResult:
    """可选消费：消费企业自身杠杆"""
    debt = data.info.get("totalDebt")
    ebitda = data.info.get("ebitda")

    if debt is None or ebitda is None or ebitda <= 0:
        return CheckResult(name="🛍️消费企业杠杆", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    ratio = debt / ebitda

    if ratio > 4:
        return CheckResult(
            name="🛍️消费企业杠杆", passed=False, severity="🔴",
            detail=f"Debt/EBITDA = {ratio:.1f}x → 消费企业高杠杆很危险，经济下行时首先倒下",
            value=f"{ratio:.1f}x",
        )

    if ratio > 2.5:
        return CheckResult(
            name="🛍️消费企业杠杆", passed=False, severity="🟡",
            detail=f"Debt/EBITDA = {ratio:.1f}x → 偏高",
            value=f"{ratio:.1f}x",
        )

    return CheckResult(
        name="🛍️消费企业杠杆", passed=True, severity="⚪",
        detail=f"Debt/EBITDA = {ratio:.1f}x，杠杆可控",
        value=f"{ratio:.1f}x",
    )


# --- 必选消费专属 ---

def check_margin_stability(data: StockData, config: IndustryConfig) -> CheckResult:
    """必选消费：毛利率稳定性（消费品公司毛利率不应该大幅波动）"""
    gp_row = safe_get_row(data.income_stmt, ["Gross Profit"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if gp_row is None or rev_row is None or len(gp_row.dropna()) < 3:
        return CheckResult(name="🧴毛利率稳定性", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    margins = []
    for i in range(min(len(gp_row.dropna()), len(rev_row.dropna()))):
        r = rev_row.dropna().iloc[i]
        g = gp_row.dropna().iloc[i]
        if r > 0:
            margins.append(g / r)

    if len(margins) < 3:
        return CheckResult(name="🧴毛利率稳定性", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    spread = max(margins) - min(margins)

    if spread > 0.05:
        return CheckResult(
            name="🧴毛利率稳定性", passed=False, severity="🟡",
            detail=f"毛利率在 {min(margins):.1%} ~ {max(margins):.1%} 之间波动（幅度 {spread:.1%}）→ "
                   f"消费品公司毛利率应该稳定，波动说明成本控制或定价权有问题",
            value=f"波动 {spread:.1%}",
        )

    return CheckResult(
        name="🧴毛利率稳定性", passed=True, severity="⚪",
        detail=f"毛利率稳定在 {min(margins):.1%} ~ {max(margins):.1%}，定价权健康",
        value=f"波动 {spread:.1%}",
    )


def check_brand_power(data: StockData, config: IndustryConfig) -> CheckResult:
    """必选消费：品牌力代理（毛利率绝对水平 + SGA效率）"""
    gm = data.info.get("grossMargins")
    om = data.info.get("operatingMargins")

    if gm is None:
        return CheckResult(name="🧴品牌力", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    if gm < 0.25:
        return CheckResult(
            name="🧴品牌力", passed=False, severity="🟡",
            detail=f"毛利率 {gm:.1%} → 消费品毛利率低于25%说明缺乏定价权，可能在打价格战",
            value=f"GM {gm:.1%}",
        )

    detail = f"毛利率 {gm:.1%}"
    if om is not None:
        detail += f"，营业利润率 {om:.1%}"

    return CheckResult(
        name="🧴品牌力", passed=True, severity="⚪",
        detail=f"{detail}，品牌定价权正常",
        value=f"GM {gm:.1%}",
    )


# --- 通信服务专属 ---

def check_subscriber_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """通信：用户/订阅趋势（用营收增速代理）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    if rev_row is None:
        return CheckResult(name="📡用户趋势", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(rev_row)
    trend = multi_year_trend(rev_row)

    if trend == "连续下降":
        return CheckResult(
            name="📡用户趋势", passed=False, severity="🔴",
            detail=f"营收连续下降 → 用户可能在流失（cord-cutting等结构性问题）",
            value=f"{trend}",
        )

    if growth is not None and growth < -0.03:
        return CheckResult(
            name="📡用户趋势", passed=False, severity="🟡",
            detail=f"营收同比 {fmt_pct(growth)} → 可能有用户流失",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="📡用户趋势", passed=True, severity="⚪",
        detail=f"营收趋势：{trend or 'N/A'}，同比 {fmt_pct(growth)}",
        value=fmt_pct(growth),
    )


def check_content_spend_ratio(data: StockData, config: IndustryConfig) -> CheckResult:
    """通信/媒体：内容支出效率"""
    cogs_row = safe_get_row(data.income_stmt, ["Cost Of Revenue", "Cost of Goods Sold"])
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])

    if cogs_row is None or rev_row is None:
        return CheckResult(name="📡内容支出效率", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    cogs_growth = yoy_growth(cogs_row)
    rev_growth = yoy_growth(rev_row)

    if cogs_growth is None or rev_growth is None:
        return CheckResult(name="📡内容支出效率", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    if cogs_growth > rev_growth + 0.10:
        return CheckResult(
            name="📡内容支出效率", passed=False, severity="🟡",
            detail=f"成本增速({fmt_pct(cogs_growth)})远超营收增速({fmt_pct(rev_growth)}) → "
                   f"内容军备竞赛，利润率在被侵蚀",
            value=f"成本 {fmt_pct(cogs_growth)} vs 营收 {fmt_pct(rev_growth)}",
        )

    return CheckResult(
        name="📡内容支出效率", passed=True, severity="⚪",
        detail=f"成本增速 {fmt_pct(cogs_growth)} vs 营收增速 {fmt_pct(rev_growth)}，效率正常",
    )


# --- 工业专属 ---

def check_backlog_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """工业：订单积压趋势代理（用营收 + 应收变化推测）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    ar_row = safe_get_row(data.balance_sheet, ["Net Receivables", "Accounts Receivable"])

    if rev_row is None:
        return CheckResult(name="🏭订单信号", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    rev_growth = yoy_growth(rev_row)
    rev_trend = multi_year_trend(rev_row)

    detail = f"营收趋势：{rev_trend or 'N/A'}，同比 {fmt_pct(rev_growth)}"

    if rev_trend == "连续下降":
        return CheckResult(
            name="🏭订单信号", passed=False, severity="🟡",
            detail=f"{detail} → 工业股营收连续下降可能意味着订单在减少",
            value=fmt_pct(rev_growth),
        )

    return CheckResult(
        name="🏭订单信号", passed=True, severity="⚪",
        detail=f"{detail}。注意：需人工查10-K中的backlog/order数据",
        value=fmt_pct(rev_growth),
    )


def check_capex_cycle(data: StockData, config: IndustryConfig) -> CheckResult:
    """工业：资本支出周期"""
    capex_row = safe_get_row(data.cashflow, ["Capital Expenditure", "Capital Expenditures"])
    if capex_row is None or len(capex_row.dropna()) < 2:
        return CheckResult(name="🏭资本支出周期", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(capex_row.abs())

    if growth is not None and growth > 0.3:
        return CheckResult(
            name="🏭资本支出周期", passed=False, severity="🟡",
            detail=f"资本支出同比 {fmt_pct(growth)} → 大幅扩产，需确认是否有对应订单支撑",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="🏭资本支出周期", passed=True, severity="⚪",
        detail=f"资本支出同比 {fmt_pct(growth)}，节奏正常",
        value=fmt_pct(growth),
    )


# --- 房地产/REITs 专属 ---

def check_ffo_trend(data: StockData, config: IndustryConfig) -> CheckResult:
    """REITs：FFO趋势代理（用经营现金流代替）"""
    cfo_row = safe_get_row(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    if cfo_row is None or len(cfo_row.dropna()) < 2:
        return CheckResult(name="🏠FFO趋势", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(cfo_row)
    trend = multi_year_trend(cfo_row)

    if trend == "连续下降":
        return CheckResult(
            name="🏠FFO趋势", passed=False, severity="🔴",
            detail=f"经营现金流连续下降 → REITs核心盈利在恶化",
            value=f"{trend}",
        )

    return CheckResult(
        name="🏠FFO趋势", passed=True, severity="⚪",
        detail=f"经营现金流趋势：{trend or 'N/A'}，同比 {fmt_pct(growth)}",
        value=fmt_pct(growth),
    )


def check_debt_maturity(data: StockData, config: IndustryConfig) -> CheckResult:
    """REITs：债务压力"""
    debt = data.info.get("totalDebt")
    cfo_row = safe_get_row(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])

    if debt is None or cfo_row is None:
        return CheckResult(name="🏠债务压力", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    cfo = cfo_row.dropna().iloc[0] if len(cfo_row.dropna()) > 0 else 0

    if cfo <= 0:
        return CheckResult(name="🏠债务压力", passed=False, severity="🔴",
                           detail="经营现金流为负但有大量债务", value="N/A")

    ratio = debt / cfo

    if ratio > 8:
        return CheckResult(
            name="🏠债务压力", passed=False, severity="🔴",
            detail=f"债务/经营现金流 = {ratio:.1f}x → 需要8年以上现金流还债，在高利率环境下很危险",
            value=f"{ratio:.1f}x",
        )

    if ratio > 5:
        return CheckResult(
            name="🏠债务压力", passed=False, severity="🟡",
            detail=f"债务/经营现金流 = {ratio:.1f}x → 偏高",
            value=f"{ratio:.1f}x",
        )

    return CheckResult(
        name="🏠债务压力", passed=True, severity="⚪",
        detail=f"债务/经营现金流 = {ratio:.1f}x，债务可控",
        value=f"{ratio:.1f}x",
    )


def check_occupancy_proxy(data: StockData, config: IndustryConfig) -> CheckResult:
    """REITs：出租率代理（营收趋势）"""
    rev_row = safe_get_row(data.income_stmt, ["Total Revenue", "Revenue"])
    if rev_row is None:
        return CheckResult(name="🏠出租率信号", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(rev_row)

    if growth is not None and growth < -0.05:
        return CheckResult(
            name="🏠出租率信号", passed=False, severity="🟡",
            detail=f"营收同比 {fmt_pct(growth)} → 可能反映出租率下降或租金压力",
            value=fmt_pct(growth),
        )

    return CheckResult(
        name="🏠出租率信号", passed=True, severity="⚪",
        detail=f"营收同比 {fmt_pct(growth)}，出租表现正常",
        value=fmt_pct(growth),
    )


# --- 公用事业专属 ---

def check_regulatory_rate_base(data: StockData, config: IndustryConfig) -> CheckResult:
    """公用事业：受管制资产基础增长"""
    asset_row = safe_get_row(data.balance_sheet, [
        "Total Non Current Assets", "Property Plant And Equipment Net", "Net PPE",
    ])
    if asset_row is None or len(asset_row.dropna()) < 2:
        return CheckResult(name="⚡️受管制资产", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    growth = yoy_growth(asset_row)

    return CheckResult(
        name="⚡️受管制资产", passed=True, severity="⚪",
        detail=f"长期资产同比 {fmt_pct(growth)}（公用事业增长主要看rate base扩张）",
        value=fmt_pct(growth),
    )


def check_dividend_coverage(data: StockData, config: IndustryConfig) -> CheckResult:
    """公用事业：分红覆盖率"""
    cfo_row = safe_get_row(data.cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    div_row = safe_get_row(data.cashflow, [
        "Common Stock Dividend Paid", "Cash Dividends Paid",
        "Payment Of Dividends And Other Cash Distributions",
    ])

    if cfo_row is None or div_row is None:
        return CheckResult(name="⚡️分红覆盖", passed=True, severity="⚪",
                           detail="数据不足", value="N/A")

    cfo = cfo_row.dropna().iloc[0] if len(cfo_row.dropna()) > 0 else 0
    div = abs(div_row.dropna().iloc[0]) if len(div_row.dropna()) > 0 else 0

    if div == 0:
        return CheckResult(name="⚡️分红覆盖", passed=True, severity="⚪",
                           detail="未发现分红数据", value="N/A")

    ratio = cfo / div if div > 0 else 999

    if ratio < 1.0:
        return CheckResult(
            name="⚡️分红覆盖", passed=False, severity="🔴",
            detail=f"经营现金流/分红 = {ratio:.2f}x → 现金流不够发分红，在吃老本",
            value=f"{ratio:.2f}x",
        )

    if ratio < 1.3:
        return CheckResult(
            name="⚡️分红覆盖", passed=False, severity="🟡",
            detail=f"经营现金流/分红 = {ratio:.2f}x → 覆盖率偏紧",
            value=f"{ratio:.2f}x",
        )

    return CheckResult(
        name="⚡️分红覆盖", passed=True, severity="⚪",
        detail=f"经营现金流/分红 = {ratio:.2f}x，覆盖充足",
        value=f"{ratio:.2f}x",
    )


# ---------------------------------------------------------------------------
# 7. 检查调度器：注册所有行业专属检查
# ---------------------------------------------------------------------------

EXTRA_CHECK_REGISTRY = {
    "cycle_position": check_cycle_position,
    "reserve_depletion": check_reserve_depletion,
    "capex_discipline": check_capex_discipline,
    "commodity_dependency": check_commodity_dependency,
    "sbc_dilution": check_sbc_dilution,
    "rd_trend": check_rd_trend,
    "deferred_revenue_trend": check_deferred_revenue_trend,
    "revenue_concentration": check_revenue_concentration,
    "pipeline_dependency": check_pipeline_dependency,
    "patent_cliff": check_patent_cliff,
    "book_value_trend": check_book_value_trend,
    "credit_quality_proxy": check_credit_quality_proxy,
    "capital_ratio": check_capital_ratio,
    "same_store_signal": check_same_store_signal,
    "consumer_leverage": check_consumer_leverage,
    "margin_stability": check_margin_stability,
    "brand_power": check_brand_power,
    "subscriber_trend": check_subscriber_trend,
    "content_spend_ratio": check_content_spend_ratio,
    "backlog_trend": check_backlog_trend,
    "capex_cycle": check_capex_cycle,
    "ffo_trend": check_ffo_trend,
    "debt_maturity": check_debt_maturity,
    "occupancy_proxy": check_occupancy_proxy,
    "regulatory_rate_base": check_regulatory_rate_base,
    "dividend_coverage": check_dividend_coverage,
}


def run_all_checks(data: StockData) -> StockReport:
    sector = data.info.get("sector", "")
    config = get_config(sector)

    report = StockReport(
        ticker=data.ticker,
        name=data.info.get("shortName", ""),
        sector=sector,
        industry=data.info.get("industry", ""),
        config_name=config.name,
    )

    # 通用检查
    report.checks.append(check_cash_profit_ratio(data, config))
    report.checks.append(check_receivable_growth(data, config))
    report.checks.append(check_inventory_growth(data, config))
    report.checks.append(check_gaap_vs_actual(data, config))
    report.checks.append(check_insider_activity(data, config))
    report.checks.append(check_buyback_quality(data, config))
    report.checks.append(check_revenue_trend(data, config))
    report.checks.append(check_debt_level(data, config))

    # 行业专属检查
    for check_name in config.extra_checks:
        fn = EXTRA_CHECK_REGISTRY.get(check_name)
        if fn:
            report.checks.append(fn(data, config))

    return report


# ---------------------------------------------------------------------------
# 8. 输出
# ---------------------------------------------------------------------------

def print_report(report: StockReport):
    print(f"\n{'━' * 80}")
    print(f"  {report.ticker} | {report.name}")
    print(f"  行业：{report.sector} → {report.industry}")
    print(f"  检测模式：{report.config_name}行业专属规则")
    print(f"  结论：{report.verdict}")
    print(f"{'━' * 80}")

    # 先打印有问题的
    failed = [c for c in report.checks if not c.passed]
    passed = [c for c in report.checks if c.passed]

    if failed:
        print(f"\n  ⛔ 问题项（{len(failed)}）：")
        for c in failed:
            print(f"    {c.severity} {c.name}")
            print(f"       {c.detail}")

    if passed:
        print(f"\n  ✓ 通过项（{len(passed)}）：")
        for c in passed:
            marker = "✅" if c.severity == "⚪" else "ℹ️"
            print(f"    {marker} {c.name}: {c.detail}")


def print_summary(reports: list[StockReport]):
    print(f"\n\n{'=' * 80}")
    print(f"  排雷汇总  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 80}")

    # 分类
    approved = [r for r in reports if "✅" in r.verdict]
    investigate = [r for r in reports if "🔍" in r.verdict]
    warning = [r for r in reports if "⚠️" in r.verdict]
    rejected = [r for r in reports if "🚫" in r.verdict]
    unknown = [r for r in reports if "❓" in r.verdict]

    print(f"\n  ✅ 进入深度研究（{len(approved)}）：", end="")
    if approved:
        print("  " + "  ".join(f"{r.ticker}" for r in approved))
    else:
        print("  无")

    print(f"  🔍 可研究（{len(investigate)}）：", end="")
    if investigate:
        print("  " + "  ".join(f"{r.ticker}" for r in investigate))
    else:
        print("  无")

    print(f"  ⚠️ 高风险/需关注（{len(warning)}）：", end="")
    if warning:
        print("  " + "  ".join(f"{r.ticker}" for r in warning))
    else:
        print("  无")

    print(f"  🚫 淘汰（{len(rejected)}）：", end="")
    if rejected:
        print("  " + "  ".join(f"{r.ticker}" for r in rejected))
    else:
        print("  无")

    # 详细表格
    print(f"\n  {'代码':<7} {'行业模式':<8} {'🔴':>3} {'🟡':>3} {'✅':>3} {'结论':<16}")
    print(f"  {'-' * 60}")
    for r in reports:
        print(f"  {r.ticker:<7} {r.config_name:<8} {r.red_count:>3} {r.yellow_count:>3} "
              f"{r.pass_count:>3}  {r.verdict}")


# ---------------------------------------------------------------------------
# 9. 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="排雷器：按行业分别检测候选股风险")
    parser.add_argument("tickers", nargs="*", help="要检查的股票代码")
    parser.add_argument("--from-csv", type=str, help="从筛选器CSV结果读取")
    parser.add_argument("--top", type=int, default=30, help="从CSV中取前N只（默认30）")
    args = parser.parse_args()

    tickers = list(args.tickers)

    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            print(f"错误：文件 {csv_path} 不存在")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        csv_tickers = df["ticker"].head(args.top).tolist()
        tickers.extend(csv_tickers)

    if not tickers:
        print("用法：python3 minesweeper.py AAPL MSFT MRK")
        print("  或：python3 minesweeper.py --from-csv results_xxx.csv --top 10")
        sys.exit(1)

    # 去重
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    tickers = unique_tickers

    print(f"\n{'=' * 50}")
    print(f"  排雷器 | 检查 {len(tickers)} 只股票")
    print(f"{'=' * 50}")

    reports = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\r  [{i}/{len(tickers)}] 正在排雷 {ticker:<7}", end="", flush=True)
        try:
            data = fetch_stock_data(ticker)
            report = run_all_checks(data)
            reports.append(report)
        except Exception as e:
            report = StockReport(ticker=ticker, name="", sector="", industry="",
                                 config_name="", error=str(e))
            reports.append(report)

        if i % 5 == 0:
            time.sleep(0.5)

    print()

    # 输出每只股票的详细报告
    for report in reports:
        print_report(report)

    # 汇总
    print_summary(reports)

    print(f"\n  提示：排雷结果基于公开财报数据的自动分析。")
    print(f"        标注「需人工查证」的项目必须阅读10-K原文确认。")
    print(f"        此工具不构成投资建议。\n")


if __name__ == "__main__":
    main()
