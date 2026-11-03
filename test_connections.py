"""Step 2: check that Alpaca (paper) and Gemini both work. Read-only; places no orders."""
import os

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. Alpaca paper account (read-only)
try:
    tc = TradingClient(os.environ["ALPACA_KEY"], os.environ["ALPACA_SECRET"], paper=True)
    acct = tc.get_account()
    clock = tc.get_clock()
    print(f"[Alpaca] OK - account {acct.account_number}, cash ${float(acct.cash):,.2f}, "
          f"market open: {clock.is_open}, next open: {clock.next_open}")
except Exception as e:
    print(f"[Alpaca] FAILED - {type(e).__name__}: {e}")

# 2. Gemini, via Google's OpenAI-compatible endpoint
gemini = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30,
    max_retries=0,
)
try:
    models = sorted(m.id for m in gemini.models.list() if "flash" in m.id.lower())
    print(f"[Gemini] OK - {len(models)} Flash models available:")
    for m in models:
        print("   ", m)
except Exception as e:
    models = []
    print(f"[Gemini] model list FAILED - {type(e).__name__}: {e}")

# 3. One tiny test message
if models:
    model = os.environ.get("GEMINI_MODEL") or models[0]
    try:
        r = gemini.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi in five words or fewer."}],
        )
        print(f"[Gemini] {model} replied: {r.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"[Gemini] chat with {model} FAILED - {type(e).__name__}: {e}")
