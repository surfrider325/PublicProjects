"""
S&P 500 Financial Health Screener — Backend
Run: python app.py
Requires: pip install flask yfinance flask-cors
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import traceback

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_liquidity(current_ratio):
    if current_ratio is None: return 10
    if current_ratio > 2:    return 20
    if current_ratio > 1.5:  return 17
    if current_ratio > 1.0:  return 13
    if current_ratio > 0.7:  return 8
    return 4

def score_profitability(net_margin):
    if net_margin is None: return 10
    if net_margin > 0.20:  return 20
    if net_margin > 0.15:  return 17
    if net_margin > 0.10:  return 14
    if net_margin > 0.05:  return 10
    if net_margin > 0:     return 6
    return 2

def score_leverage(debt_to_equity):
    if debt_to_equity is None: return 10
    if debt_to_equity < 0.5:  return 20
    if debt_to_equity < 1.0:  return 17
    if debt_to_equity < 2.0:  return 13
    if debt_to_equity < 3.0:  return 9
    return 4

def score_growth(revenue_growth):
    if revenue_growth is None: return 10
    if revenue_growth > 0.20:  return 20
    if revenue_growth > 0.10:  return 17
    if revenue_growth > 0.05:  return 13
    if revenue_growth > 0:     return 9
    return 4

def score_efficiency(operating_margin, asset_turnover):
    score = 0
    count = 0
    if operating_margin is not None:
        count += 1
        if operating_margin > 0.25:   score += 20
        elif operating_margin > 0.15: score += 16
        elif operating_margin > 0.08: score += 12
        elif operating_margin > 0:    score += 7
        else:                          score += 3
    if asset_turnover is not None:
        count += 1
        if asset_turnover > 1.5:   score += 20
        elif asset_turnover > 0.8: score += 16
        elif asset_turnover > 0.5: score += 12
        elif asset_turnover > 0.3: score += 7
        else:                       score += 3
    if count == 0: return 10
    return round(score / count)


# Approximate sector benchmark scores (S&P 500 medians)
SECTOR_BENCHMARKS = {
    "Technology":               {"liquidity": 14, "profitability": 15, "leverage": 14, "growth": 14, "efficiency": 15},
    "Information Technology":   {"liquidity": 14, "profitability": 15, "leverage": 14, "growth": 14, "efficiency": 15},
    "Health Care":              {"liquidity": 14, "profitability": 12, "leverage": 13, "growth": 12, "efficiency": 13},
    "Financials":               {"liquidity":  8, "profitability": 12, "leverage":  7, "growth": 10, "efficiency": 12},
    "Consumer Discretionary":   {"liquidity": 11, "profitability": 10, "leverage": 11, "growth": 11, "efficiency": 13},
    "Consumer Staples":         {"liquidity": 11, "profitability": 11, "leverage": 11, "growth":  9, "efficiency": 14},
    "Industrials":              {"liquidity": 13, "profitability": 10, "leverage": 12, "growth": 10, "efficiency": 13},
    "Communication Services":   {"liquidity": 12, "profitability": 12, "leverage": 12, "growth": 11, "efficiency": 13},
    "Energy":                   {"liquidity": 12, "profitability": 11, "leverage": 12, "growth": 10, "efficiency": 12},
    "Materials":                {"liquidity": 13, "profitability": 10, "leverage": 12, "growth":  9, "efficiency": 12},
    "Real Estate":              {"liquidity":  9, "profitability": 10, "leverage":  8, "growth":  9, "efficiency": 11},
    "Utilities":                {"liquidity":  9, "profitability": 10, "leverage":  7, "growth":  8, "efficiency": 11},
    "default":                  {"liquidity": 12, "profitability": 11, "leverage": 11, "growth": 10, "efficiency": 12},
}


def safe_get(info, *keys):
    """Try multiple key aliases, return first non-None value."""
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def compute_scores_from_info(info):
    """Extract metrics and compute scores from yfinance info dict."""
    current_ratio   = safe_get(info, "currentRatio")
    quick_ratio     = safe_get(info, "quickRatio")
    net_margin      = safe_get(info, "profitMargins", "netMargins")
    roe             = safe_get(info, "returnOnEquity")
    debt_to_equity  = safe_get(info, "debtToEquity")
    revenue_growth  = safe_get(info, "revenueGrowth")
    asset_turnover  = None  # not directly in yfinance info; computed below
    operating_margin = safe_get(info, "operatingMargins")

    # yfinance returns D/E as a ratio (e.g. 274 = 2.74x) — normalise if needed
    if debt_to_equity is not None and debt_to_equity > 20:
        debt_to_equity = debt_to_equity / 100

    metrics = {
        "currentRatio":    current_ratio,
        "quickRatio":      quick_ratio,
        "netMargin":       net_margin,
        "roe":             roe,
        "debtToEquity":    debt_to_equity,
        "revenueGrowth":   revenue_growth,
        "assetTurnover":   asset_turnover,
        "operatingMargin": operating_margin,
    }

    scores = {
        "liquidity":     score_liquidity(current_ratio),
        "profitability": score_profitability(net_margin),
        "leverage":      score_leverage(debt_to_equity),
        "growth":        score_growth(revenue_growth),
        "efficiency":    score_efficiency(operating_margin, asset_turnover),
    }

    return metrics, scores


def compute_historical_scores(ticker_obj):
    """Compute per-year scores from annual income statement + balance sheet."""
    history = []
    try:
        income   = ticker_obj.income_stmt      # columns = fiscal year dates
        balance  = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        if income is None or income.empty:
            return []

        years = income.columns[:4]  # up to 4 most recent fiscal years

        for col in reversed(list(years)):
            year_label = str(col.year)

            def g(df, *keys):
                for k in keys:
                    if df is not None and k in df.index:
                        v = df.loc[k, col]
                        if v is not None and str(v) != "nan":
                            return float(v)
                return None

            total_revenue  = g(income, "Total Revenue")
            net_income     = g(income, "Net Income")
            operating_inc  = g(income, "Operating Income", "EBIT")
            total_assets   = g(balance, "Total Assets")
            total_liab     = g(balance, "Total Liabilities Net Minority Interest", "Total Liabilities")
            total_equity   = g(balance, "Stockholders Equity", "Total Stockholders Equity")
            current_assets = g(balance, "Current Assets")
            current_liab   = g(balance, "Current Liabilities")

            # Derive ratios
            net_margin_h      = (net_income / total_revenue)   if net_income and total_revenue else None
            operating_margin_h = (operating_inc / total_revenue) if operating_inc and total_revenue else None
            current_ratio_h   = (current_assets / current_liab) if current_assets and current_liab else None
            debt_to_equity_h  = (total_liab / total_equity)    if total_liab and total_equity and total_equity != 0 else None
            asset_turnover_h  = (total_revenue / total_assets) if total_revenue and total_assets else None

            # Revenue growth requires prior year — approximate as None for oldest year
            revenue_growth_h = None

            scores_h = {
                "year":          f"FY{year_label}",
                "liquidity":     score_liquidity(current_ratio_h),
                "profitability": score_profitability(net_margin_h),
                "leverage":      score_leverage(debt_to_equity_h),
                "growth":        score_growth(revenue_growth_h),
                "efficiency":    score_efficiency(operating_margin_h, asset_turnover_h),
            }
            history.append(scores_h)

        # Fill in revenue growth now that we have the full list
        for i in range(1, len(history)):
            try:
                rev_cols = list(reversed(list(years)))
                curr_rev = income.loc["Total Revenue", rev_cols[i]]
                prev_rev = income.loc["Total Revenue", rev_cols[i-1]]
                if prev_rev and prev_rev != 0:
                    growth = (curr_rev - prev_rev) / abs(prev_rev)
                    history[i]["growth"] = score_growth(growth)
            except Exception:
                pass

    except Exception:
        pass

    return history


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/score/<ticker>")
def score_ticker(ticker):
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None and info.get("symbol") is None:
            return jsonify({"error": f"No data found for {ticker}. Check the symbol."}), 404

        company  = info.get("longName") or info.get("shortName") or ticker
        sector   = info.get("sector") or "Unknown"
        industry = info.get("industry") or ""
        currency = info.get("currency") or "USD"
        price    = info.get("currentPrice") or info.get("regularMarketPrice")
        mkt_cap  = info.get("marketCap")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        fifty_two_week_low  = info.get("fiftyTwoWeekLow")

        metrics, scores = compute_scores_from_info(info)
        historical = compute_historical_scores(t)
        benchmarks = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["default"])

        # Build summary
        total = sum(scores.values())
        strengths, risks = [], []
        if scores["profitability"] >= 17: strengths.append("strong profitability")
        if scores["liquidity"] >= 17:     strengths.append("solid liquidity")
        if scores["leverage"] >= 17:      strengths.append("low leverage")
        if scores["growth"] >= 17:        strengths.append("high revenue growth")
        if scores["efficiency"] >= 17:    strengths.append("excellent efficiency")
        if scores["profitability"] <= 6:  risks.append("weak margins")
        if scores["liquidity"] <= 8:      risks.append("tight liquidity")
        if scores["leverage"] <= 9:       risks.append("high debt load")
        if scores["growth"] <= 4:         risks.append("declining revenue")

        summary_parts = []
        if strengths: summary_parts.append(f"{company} shows {', '.join(strengths[:2])}.")
        if risks:     summary_parts.append(f"Key risks include {', '.join(risks[:2])}.")
        if not summary_parts: summary_parts = [f"{company} shows a balanced financial profile."]
        summary = " ".join(summary_parts)

        return jsonify({
            "ticker":    ticker,
            "company":   company,
            "sector":    sector,
            "industry":  industry,
            "period":    "TTM / Latest",
            "currency":  currency,
            "price":     price,
            "marketCap": mkt_cap,
            "peRatio":   pe_ratio,
            "fiftyTwoWeekHigh": fifty_two_week_high,
            "fiftyTwoWeekLow":  fifty_two_week_low,
            "scores":           scores,
            "metrics":          metrics,
            "sectorBenchmarks": benchmarks,
            "historicalScores": historical,
            "summary":          summary,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    print("\n  S&P 500 Financial Health Screener")
    print("  ─────────────────────────────────")
    print("  Running at: http://localhost:5000\n")
    app.run(debug=True, port=5000)
