"""The two AI agents: a Research Analyst and a Portfolio Manager.

Each agent runs in a loop: Gemini decides which tools to call, we run them and
hand back the results, and it repeats until the agent gives its final answer.
"""
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError, OpenAI

load_dotenv()

WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "COST", "AMD"]

# Main model first, backup second (Google's servers are sometimes overloaded).
MODELS = [os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"), "gemini-flash-lite-latest"]

llm = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60,
    max_retries=0,
)
trading = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
market_data = StockHistoricalDataClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], raw_data=True)
news_data = NewsClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], raw_data=True)


# ---------- Tools the agents can call ----------

def get_price_history(symbols):
    """30-day price summary per symbol (computed here so the AI gets clean numbers)."""
    symbols = [s for s in symbols if s in WATCHLIST + ["SPY"]]
    if not symbols:
        return {"error": "No valid watchlist symbols given."}
    bars = market_data.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=45), feed=DataFeed.IEX,
    ))
    bars = bars.get("bars", bars)
    out = {}
    for sym in symbols:
        closes = [b["c"] for b in bars.get(sym, [])][-30:]
        if len(closes) < 6:
            out[sym] = {"error": "not enough data"}
            continue
        last = closes[-1]
        out[sym] = {
            "last_close": round(last, 2),
            "change_5d_pct": round((last / closes[-6] - 1) * 100, 2),
            "change_30d_pct": round((last / closes[0] - 1) * 100, 2),
            "avg_20d": round(sum(closes[-20:]) / len(closes[-20:]), 2),
            "high_30d": round(max(closes), 2),
            "low_30d": round(min(closes), 2),
            "last_10_closes": [round(c, 2) for c in closes[-10:]],
        }
    return out


def get_news(symbols):
    """Up to 5 headlines per symbol from the last 3 days."""
    out = {}
    for sym in [s for s in symbols if s in WATCHLIST]:
        raw = news_data.get_news(NewsRequest(
            symbols=sym, start=datetime.now(timezone.utc) - timedelta(days=3), limit=5,
        ))
        items = raw.get("news", []) if isinstance(raw, dict) else []
        out[sym] = [{"headline": n.get("headline"), "time": n.get("created_at")} for n in items[:5]]
    return out


def get_latest_prices(symbols):
    raw = market_data.get_stock_latest_trade(StockLatestTradeRequest(
        symbol_or_symbols=list(symbols), feed=DataFeed.IEX,
    ))
    raw = raw.get("trades", raw)
    return {s: raw[s]["p"] for s in symbols if s in raw}


def get_account():
    a = trading.get_account()
    return {"cash": float(a.cash), "equity": float(a.equity),
            "equity_at_last_close": float(a.last_equity)}


def get_positions():
    return [{"symbol": p.symbol, "qty": float(p.qty), "market_value": float(p.market_value),
             "unrealized_pl_pct": round(float(p.unrealized_plpc) * 100, 2)}
            for p in trading.get_all_positions()]


def _tool(name, description, properties=None):
    params = {"type": "object", "properties": properties or {}}
    if properties:
        params["required"] = list(properties)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


SYMBOLS_ARG = {"symbols": {"type": "array", "items": {"type": "string"},
                           "description": "Ticker symbols, e.g. [\"AAPL\", \"NVDA\"]"}}

RESEARCH_TOOLS = {
    "get_price_history": (get_price_history, _tool(
        "get_price_history", "30-day daily price summary for one or more stocks.", SYMBOLS_ARG)),
    "get_news": (get_news, _tool(
        "get_news", "Recent news headlines (last 3 days) for one or more stocks.", SYMBOLS_ARG)),
}

PM_TOOLS = {
    "get_account": (get_account, _tool("get_account", "Current cash and total equity of the paper account.")),
    "get_positions": (get_positions, _tool("get_positions", "Stocks the fund currently holds.")),
    "get_latest_prices": (get_latest_prices, _tool(
        "get_latest_prices", "Latest trade price for one or more stocks.", SYMBOLS_ARG)),
}


# ---------- The agent loop ----------

def _chat(messages, tools):
    """Call Gemini, retrying once and then falling back to the backup model if it's busy."""
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                return llm.chat.completions.create(model=model, messages=messages, tools=tools)
            except (APIStatusError, APITimeoutError) as e:
                last_error = e
                if getattr(e, "status_code", 503) not in (429, 500, 503) and not isinstance(e, APITimeoutError):
                    raise
                time.sleep(3)
    raise last_error


def _parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError(f"Agent did not return JSON: {text!r}")
    return json.loads(match.group(0))


def run_agent(name, system_prompt, task, tools, max_turns=8):
    """Let the agent call tools until it returns its final JSON answer.
    Returns (answer, steps) where steps records every tool call for the dashboard."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    schemas = [schema for _, schema in tools.values()]
    steps = []
    for _ in range(max_turns):
        msg = _chat(messages, schemas).choices[0].message
        if not msg.tool_calls:
            return _parse_json(msg.content), steps
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            fn = tools.get(call.function.name, (None,))[0]
            args = json.loads(call.function.arguments or "{}")
            try:
                result = fn(**args) if fn else {"error": f"Unknown tool {call.function.name}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            steps.append({"tool": call.function.name, "args": args})
            print(f"  [{name}] called {call.function.name}({args})")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})
    messages.append({"role": "user", "content": "Stop calling tools and give your final JSON answer now."})
    return _parse_json(_chat(messages, schemas).choices[0].message.content), steps


RESEARCH_PROMPT = f"""You are the Research Analyst at a small AI-run fund.
Watchlist: {", ".join(WATCHLIST)}.
Use your tools to study every stock on the watchlist: look at price trends AND recent news.
Call tools with several symbols at once to save time. Be skeptical and evidence-based.
When done, reply with ONLY this JSON:
{{"notes": [{{"symbol": "AAPL", "rating": "buy" | "hold" | "avoid", "confidence": 1-10,
  "reason": "one or two sentences citing specific numbers or headlines"}}],
  "market_view": "one sentence on the overall market mood"}}"""

PM_PROMPT = """You are the Portfolio Manager at a small AI-run fund trading a paper account.
You receive the Research Analyst's notes. Use your tools to check cash, current positions and prices,
then decide what to trade. Rules you MUST follow (they are also enforced in code):
- Only trade watchlist stocks. No short selling: only sell stocks you already hold.
- Each buy is at most $10,000. Hold at most 6 different stocks.
- Only act on strong evidence; doing nothing is fine. Sell a holding if research turns clearly negative.
When done, reply with ONLY this JSON:
{"trades": [{"symbol": "NVDA", "side": "buy" | "sell", "dollars": 5000, "reason": "why"}],
 "summary": "two or three plain-English sentences explaining today's decisions"}"""


def run_research():
    return run_agent("Research Analyst", RESEARCH_PROMPT,
                     "Research today's watchlist and write your notes.", RESEARCH_TOOLS)


def run_portfolio_manager(research):
    task = "Research Analyst notes:\n" + json.dumps(research, indent=1) + "\n\nDecide today's trades."
    return run_agent("Portfolio Manager", PM_PROMPT, task, PM_TOOLS)
