"""One run of the AI Hedge Fund: research -> decide -> safety checks -> trade -> log.

Set DRY_RUN=true in .env to log trades without placing them.
Create a file named PAUSED in this folder to stop all trading (kill switch).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from agents import (WATCHLIST, get_account, get_latest_prices, get_positions,
                    run_portfolio_manager, run_research, trading)

HERE = Path(__file__).parent
LOG_FILE = HERE / "dashboard" / "log.json"

# Hard limits - enforced here in code, whatever the AI says.
MAX_DOLLARS_PER_TRADE = 10_000
MAX_POSITIONS = 6
DAILY_LOSS_STOP_PCT = 3  # no new buys if the fund is down this much since yesterday's close

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"


def load_log():
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return {"baseline": None, "runs": []}


def save_log(log):
    LOG_FILE.parent.mkdir(exist_ok=True)
    LOG_FILE.write_text(json.dumps(log, indent=1, default=str))


def check_trades(trades, account, positions):
    """Split the AI's proposed trades into approved and rejected, with reasons."""
    held = {p["symbol"] for p in positions}
    n_positions = len(held)
    cash = account["cash"]
    down_pct = (1 - account["equity"] / account["equity_at_last_close"]) * 100
    approved, rejected = [], []
    for t in trades:
        sym, side = str(t.get("symbol", "")).upper(), str(t.get("side", "")).lower()
        dollars = float(t.get("dollars") or 0)
        reason = None
        if sym not in WATCHLIST:
            reason = "not on the watchlist"
        elif side == "sell":
            if sym not in held:
                reason = "can't sell a stock we don't own (no short selling)"
        elif side == "buy":
            if down_pct >= DAILY_LOSS_STOP_PCT:
                reason = f"daily loss stop: fund is down {down_pct:.1f}% today"
            elif sym not in held and n_positions >= MAX_POSITIONS:
                reason = f"already holding the maximum {MAX_POSITIONS} stocks"
            elif dollars < 1:
                reason = "trade size too small"
            else:
                if dollars > MAX_DOLLARS_PER_TRADE:
                    t["note"] = f"reduced from ${dollars:,.0f} to the ${MAX_DOLLARS_PER_TRADE:,} cap"
                    dollars = MAX_DOLLARS_PER_TRADE
                if dollars > cash:
                    reason = "not enough cash"
        else:
            reason = f"unknown side '{side}'"
        trade = {**t, "symbol": sym, "side": side, "dollars": round(dollars, 2)}
        if reason:
            rejected.append({**trade, "rejected_because": reason})
        else:
            approved.append(trade)
            if side == "buy":
                cash -= dollars
                if sym not in held:
                    held.add(sym)
                    n_positions += 1
    return approved, rejected


def place(trade):
    if trade["side"] == "buy":
        order = trading.submit_order(MarketOrderRequest(
            symbol=trade["symbol"], notional=trade["dollars"],
            side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
    else:
        order = trading.close_position(trade["symbol"])
    return str(order.id)


def main():
    now = datetime.now(timezone.utc).isoformat()
    log = load_log()
    account = get_account()
    spy = get_latest_prices(["SPY"]).get("SPY")
    if log["baseline"] is None:
        log["baseline"] = {"time": now, "equity": account["equity"], "spy": spy}

    run = {"time": now, "dry_run": DRY_RUN, "market_open": trading.get_clock().is_open,
           "equity": account["equity"], "cash": account["cash"], "spy": spy}

    if (HERE / "PAUSED").exists():
        print("Kill switch is ON (PAUSED file found) - not trading.")
        run["status"] = "paused"
    else:
        print("Research Analyst is working...")
        research, research_steps = run_research()
        print("Portfolio Manager is working...")
        decision, pm_steps = run_portfolio_manager(research)

        positions = get_positions()
        approved, rejected = check_trades(decision.get("trades", []), account, positions)
        for t in approved:
            if DRY_RUN:
                t["order_id"] = "dry-run (not placed)"
            else:
                try:
                    t["order_id"] = place(t)
                except Exception as e:
                    t["order_id"] = f"FAILED: {e}"
        run.update(status="ok", research=research, research_steps=research_steps,
                   pm_steps=pm_steps, summary=decision.get("summary", ""),
                   approved=approved, rejected=rejected, positions=positions)
        print(f"\nSummary: {run['summary']}")
        for t in approved:
            print(f"  APPROVED {t['side'].upper()} {t['symbol']} ${t['dollars']:,.0f} -> {t['order_id']}")
        for t in rejected:
            print(f"  REJECTED {t.get('side')} {t.get('symbol')}: {t['rejected_because']}")

    log["runs"].append(run)
    save_log(log)
    print(f"\nSaved to {LOG_FILE.relative_to(HERE)} (dry run: {DRY_RUN})")


if __name__ == "__main__":
    main()
