"""
NIFTY ORB SIGNAL BOT — GitHub Actions Version
───────────────────────────────────────────────
Runs every 1 minute during market hours via GitHub Actions cron.
Each run is stateless — state is persisted in state.json in the repo.

Flow:
  9:31 AM  → Morning briefing + build watchlist
  Every 1m → Check breakouts on watched stocks (1-min bars)
             Check stop/target on open trades
  2:55 PM  → Force-close reminder
  3:05 PM  → End-of-day P&L summary

Credentials via GitHub Secrets:
  TELEGRAM_TOKEN   → Bot token from @BotFather
  TELEGRAM_CHAT_ID → Your Telegram user ID
"""

import os, json, sys
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import requests
import warnings
warnings.filterwarnings("ignore")

# ── CREDENTIALS (from GitHub Secrets / env vars) ──────────────────────
TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── STRATEGY CONFIG ────────────────────────────────────────────────────
RISK          = 1000.0
TOP_N         = 5
MAX_OR_PCT    = 0.75
TARGET_R      = 2.0
MA_PERIOD     = 20
SIDEWAYS_BAND = 0.3

# Market hours (IST)
BRIEFING_START = dt.time(9, 30)
BRIEFING_END   = dt.time(14, 50)   # send briefing any time before 2:50 PM if missed
TRADE_END      = dt.time(14, 55)   # 2:55 PM
EOD_START      = dt.time(15,  5)   # 3:05 PM
BOT_END        = dt.time(15, 15)   # 3:15 PM — last run

NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
    "SUNPHARMA","ULTRACEMCO","BAJFINANCE","NESTLEIND","WIPRO","ONGC","NTPC",
    "POWERGRID","M&M","TECHM","BAJAJFINSV","ADANIENT","ADANIPORTS",
    "COALINDIA","JSWSTEEL","HINDALCO","BPCL","CIPLA","DRREDDY","EICHERMOT",
    "BRITANNIA","DIVISLAB","GRASIM","INDUSINDBK","SBILIFE",
    "APOLLOHOSP","TATACONSUM","HEROMOTOCO","SHRIRAMFIN","BEL"
]

STATE_FILE = "state.json"

# ── HELPERS ────────────────────────────────────────────────────────────
def ist_now():
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)

def ist_time():
    return ist_now().time()

def log(msg):
    print(f"[{ist_now().strftime('%H:%M:%S')} IST] {msg}", flush=True)

def send(msg):
    if not TOKEN or not CHAT_ID:
        log("WARNING: No Telegram credentials found"); return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=15
        )
        return r.status_code == 200
    except Exception as e:
        log(f"Telegram error: {e}"); return False

def charges(buy_tv, sell_tv):
    brokerage = sum(min(0.0003*tv, 20) for tv in [buy_tv, sell_tv])
    stt   = 0.00025  * sell_tv
    txn   = 0.0000297 * (buy_tv + sell_tv)
    sebi  = 0.000001  * (buy_tv + sell_tv)
    stamp = 0.00003   * buy_tv
    gst   = 0.18 * (brokerage + txn + sebi)
    return brokerage + stt + txn + sebi + stamp + gst

# ── STATE MANAGEMENT ───────────────────────────────────────────────────
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return fresh_state()

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def fresh_state():
    return {
        "date": str(ist_now().date()),
        "regime": "", "direction": "", "nifty": 0, "ma20": 0,
        "briefing_sent": False, "close_sent": False, "eod_sent": False,
        "watching": {}
    }

# ── DATA FETCHING ──────────────────────────────────────────────────────
def compute_regime_from_stocks(daily_close):
    """
    Compute market regime from constituent stocks' daily data.
    No external index needed — uses the same data we already download.

    Method: equal-weighted synthetic index
      - For each stock, compute: current price / 20-day-ago price - 1
      - Average across all stocks = synthetic Nifty 20-day return
      - > +0.3%  → UPTREND
      - < -0.3%  → DOWNTREND
      - else     → SIDEWAYS
    """
    returns = []
    for sym in NIFTY50:
        if sym not in daily_close.columns: continue
        s = daily_close[sym].dropna()
        if len(s) < MA_PERIOD + 2: continue
        try:
            cur  = float(s.iloc[-1])
            past = float(s.iloc[-(MA_PERIOD + 1)])
            if past > 0:
                returns.append((cur - past) / past * 100)
        except: continue

    if len(returns) < 10:
        log(f"  Only {len(returns)} stocks had data — cannot compute regime")
        return "UNKNOWN", 0, 0

    avg_ret = round(np.mean(returns), 2)
    log(f"  Synthetic Nifty 20-day avg return: {avg_ret:+.2f}% across {len(returns)} stocks")

    if avg_ret >  SIDEWAYS_BAND: regime = "UPTREND"
    elif avg_ret < -SIDEWAYS_BAND: regime = "DOWNTREND"
    else: regime = "SIDEWAYS"

    return regime, avg_ret, 0   # nifty=avg_ret%, ma20=0 (not used for display)

def fetch_bars():
    """Fetch 1-min bars (live signals) + 5-min bars (OR levels) + daily (gap calc)."""
    import time as _time
    tks = [f"{s}.NS" for s in NIFTY50]

    for attempt in range(3):
        try:
            r1 = yf.download(tks, period="1d",  interval="1m",  progress=False, auto_adjust=True)
            r5 = yf.download(tks, period="2d",  interval="5m",  progress=False, auto_adjust=True)
            rd = yf.download(tks, period="60d", interval="1d",  progress=False, auto_adjust=True)
            if not r1.empty and not r5.empty:
                break
            log(f"  fetch_bars attempt {attempt+1}: empty data, retrying...")
            _time.sleep(3)
        except Exception as e:
            log(f"  fetch_bars attempt {attempt+1} error: {e}")
            _time.sleep(3)
    else:
        return None

    if r1.empty or r5.empty: return None

    def clean(df):
        df = df.copy()
        df.columns = [c.replace(".NS","") for c in df.columns]
        try: df.index = df.index.tz_convert("Asia/Kolkata")
        except: pass
        return df

    return {
        "o1": clean(r1["Open"]),  "h1": clean(r1["High"]),
        "l1": clean(r1["Low"]),   "c1": clean(r1["Close"]),
        "v1": clean(r1["Volume"]),
        "o5": clean(r5["Open"]),  "h5": clean(r5["High"]),
        "l5": clean(r5["Low"]),   "c5": clean(r5["Close"]),
        "v5": clean(r5["Volume"]),
        "dc": rd["Close"].copy().rename(columns=lambda c: c.replace(".NS",""))
    }

def get_stock_data(sym, bars):
    """Get OR levels (5-min) and live session (1-min) for one stock."""
    try:
        today = ist_now().date()

        # OR from 5-min bars
        b5 = pd.DataFrame({
            "H": bars["h5"][sym], "L": bars["l5"][sym],
            "C": bars["c5"][sym], "V": bars["v5"][sym]
        }).dropna()
        b5 = b5[b5.index.date == today]
        orb = b5.between_time("09:15","09:30")
        if len(orb) < 2: return None
        oh, ol  = orb.H.max(), orb.L.min()
        or_pct  = (oh - ol) / ol * 100

        # Live session from 1-min bars (for VWAP + breakout detection)
        b1 = pd.DataFrame({
            "H": bars["h1"][sym], "L": bars["l1"][sym],
            "C": bars["c1"][sym], "V": bars["v1"][sym]
        }).dropna()
        b1   = b1[b1.index.date == today]
        sess = b1.between_time("09:15","15:30").copy()
        if len(sess) < 3: return None
        tp = (sess.H + sess.L + sess.C) / 3
        sess["vwap"] = (tp * sess.V).cumsum() / sess.V.cumsum()

        return {"oh": oh, "ol": ol, "or_pct": or_pct, "sess": sess}
    except:
        return None

# ── SIGNAL LOGIC ───────────────────────────────────────────────────────
def check_breakout_now(data, direction):
    """
    Returns True only when the MOST RECENT 1-min bar just crossed the OR level.
    (Previous bar was inside OR, current bar broke out AND closed above/below VWAP)
    This ensures the alert fires exactly ONCE at the moment of breakout.
    """
    sess = data["sess"]
    if len(sess) < 2: return False
    entry_bars = sess.between_time("09:30","15:30")
    if len(entry_bars) < 2: return False

    prev = entry_bars.iloc[-2]
    curr = entry_bars.iloc[-1]
    oh, ol = data["oh"], data["ol"]

    if direction == "LONG":
        was_inside = prev.C <= oh
        broke_out  = curr.H > oh and curr.C > curr.vwap
    else:
        was_inside = prev.C >= ol
        broke_out  = curr.L < ol and curr.C < curr.vwap

    return was_inside and broke_out

def check_exit(data, trade, direction):
    """
    Checks if stop or target was hit AFTER the entry time.
    Returns ('stop', price) | ('target', price) | ('open', cur_price)
    """
    sess       = data["sess"]
    entry_time = pd.Timestamp(trade["entry_time"])
    post       = sess[sess.index > entry_time].between_time("09:30","15:30")
    if len(post) == 0:
        return "open", trade["entry"]

    for _, row in post.iterrows():
        if direction == "LONG":
            if row.L <= trade["stop"]:   return "stop",   trade["stop"]
            if row.H >= trade["target"]: return "target", trade["target"]
        else:
            if row.H >= trade["stop"]:   return "stop",   trade["stop"]
            if row.L <= trade["target"]: return "target", trade["target"]

    return "open", float(post.iloc[-1].C)

# ── MORNING BRIEFING ───────────────────────────────────────────────────
def do_briefing(state, bars):
    """Rank stocks, compute OR levels, build watchlist, send briefing."""
    today  = ist_now().date()
    regime = state["regime"]

    if regime == "SIDEWAYS":
        send(
            f"<b>🔔 NIFTY ORB — {today.strftime('%d %b %Y')}</b>\n\n"
            f"<b>⚠️ SIDEWAYS MARKET</b>\n"
            f"Nifty {state['nifty']:,.0f} | 20-MA {state['ma20']:,.0f}\n\n"
            f"🚫 <b>NO TRADES TODAY.</b>\n"
            f"Market has no clear trend. Sit on cash."
        )
        state["briefing_sent"] = True
        return state

    direction = state["direction"]
    t_emoji   = "🟢" if direction == "LONG" else "🔴"
    r_emoji   = "📈" if direction == "LONG" else "📉"

    # Gap ranking
    dc = bars["dc"]; prev = dc.shift(1)
    scores = {}
    for sym in NIFTY50:
        if sym not in bars["c5"].columns: continue
        try:
            b = bars["c5"][sym][bars["c5"].index.date == today].between_time("09:15","09:35")
            if len(b) < 3: continue
            prev_ts = pd.Timestamp(today) - pd.Timedelta(days=1)
            pc = prev.loc[prev_ts, sym] if prev_ts in prev.index else None
            if pc is None or np.isnan(float(pc)): continue
            scores[sym] = (float(b.iloc[-1]) - float(pc)) / float(pc) * 100
        except: continue

    if len(scores) < 5:
        send("⚠️ ORB Bot: Not enough data to rank stocks today.")
        state["briefing_sent"] = True
        return state

    sc       = pd.Series(scores).sort_values(ascending=False)
    watchlist = list(sc.head(TOP_N).index) if direction=="LONG" else list(sc.tail(TOP_N).index)

    # Build watchlist with OR levels
    watching = {}
    skipped  = []
    for sym in watchlist:
        d = get_stock_data(sym, bars)
        if d is None: skipped.append(f"{sym}: no data"); continue
        if d["or_pct"] > MAX_OR_PCT:
            skipped.append(f"{sym}: OR {d['or_pct']:.2f}%"); continue
        oh, ol = d["oh"], d["ol"]
        R = oh - ol
        if R <= 0: continue
        if direction == "LONG":
            entry, stop, target = round(oh,2), round(ol,2), round(oh+TARGET_R*R,2)
        else:
            entry, stop, target = round(ol,2), round(oh,2), round(ol-TARGET_R*R,2)
        watching[sym] = {
            "entry": entry, "stop": stop, "target": target,
            "shares": int(RISK/R), "or_pct": round(d["or_pct"],2),
            "gap": round(scores[sym],2), "R": round(R,2),
            "triggered": False, "exited": False,
            "entry_time": None, "exit_reason": None, "exit_price": None
        }

    state["watching"] = watching

    # Send morning message
    diff = (state["nifty"] - state["ma20"]) / state["ma20"] * 100 if state["ma20"] else 0
    date_str = today.strftime("%d %b %Y")
    msg  = f"<b>🔔 NIFTY ORB — {date_str}</b>  <i>{ist_now().strftime('%I:%M %p')} IST</i>\n\n"
    msg += f"<b>{r_emoji} Regime: {state['regime']}</b>\n"
    msg += f"   Nifty 50 avg 20-day return: <b>{state['nifty']:+.2f}%</b>\n"
    msg += f"   Bias: <b>{direction} only</b> today\n\n"
    msg += f"<b>⚙️</b> OR≤0.75% | Fixed 2R exit | ₹{int(RISK):,} risk | Close 2:55 PM\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    if watching:
        msg += f"<b>👁 WATCHING {len(watching)} STOCKS:</b>\n\n"
        for sym, w in watching.items():
            msg += f"{t_emoji} <b>{sym}</b>  (gap {w['gap']:+.2f}%  |  OR {w['or_pct']:.2f}%)\n"
            msg += f"   Entry  ₹{w['entry']:,.2f}  |  Stop  ₹{w['stop']:,.2f}  |  Target  ₹{w['target']:,.2f}\n"
            msg += f"   Qty: {w['shares']} shares  |  Risk: ₹{int(RISK):,}\n\n"
        msg += "⚡ <b>Instant alert when any breakout fires!</b>"
    else:
        msg += "❌ No valid stocks — all had OR > 0.75%\n"
        msg += "No trade alerts today."

    if skipped:
        msg += f"\n\n⛔ Skipped: {', '.join(skipped)}"

    send(msg)
    state["briefing_sent"] = True
    log(f"Briefing sent. Watching: {list(watching.keys())}")
    return state

# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    now   = ist_time()
    today = str(ist_now().date())
    log(f"Bot running | IST: {now.strftime('%H:%M:%S')}")

    # ── Startup check: verify Telegram credentials work ───────────────
    if not TOKEN or not CHAT_ID:
        log("ERROR: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in environment!")
        log(f"TOKEN set: {bool(TOKEN)} | CHAT_ID set: {bool(CHAT_ID)}")
        return

    # Skip if outside bot operating hours
    if now < BRIEFING_START or now > BOT_END:
        log(f"Outside market hours ({now}) — nothing to do."); return

    # Weekend check
    if ist_now().weekday() >= 5:
        log("Weekend — skipping."); return

    # Load state
    state = load_state()

    # New day → reset state
    if state.get("date") != today:
        log("New day — resetting state.")
        state = fresh_state()
        state["date"] = today

    # ── STEP 1: Fetch all market data (1-min + 5-min + 60-day daily) ────
    log("Fetching market data...")
    bars = fetch_bars()
    if bars is None:
        log("Data fetch failed — skipping this run.")
        send("⚠️ ORB Bot: Could not download stock data. Will retry.")
        save_state(state); return

    # ── STEP 2: Compute regime from constituent stocks (no index needed) ─
    if not state["regime"] or state["regime"] == "UNKNOWN":
        log("Computing regime from Nifty 50 constituent stocks...")
        regime, avg_ret, _ = compute_regime_from_stocks(bars["dc"])
        state["regime"]    = regime
        state["direction"] = "LONG" if regime == "UPTREND" else "SHORT"
        state["nifty"]     = avg_ret   # synthetic 20-day avg return %
        state["ma20"]      = 0
        log(f"Regime: {regime} | Avg 20d return: {avg_ret:+.2f}%")
        if regime == "UNKNOWN":
            send("⚠️ ORB Bot: Not enough stock data to determine trend. Will retry.")
            save_state(state); return

    # ── STEP 3: Sideways — no trades ─────────────────────────────────
    if state["regime"] == "SIDEWAYS":
        if not state["briefing_sent"]:
            log("SIDEWAYS — sending no-trade message")
            state["briefing_sent"] = True
            send(
                f"<b>🔔 NIFTY ORB — {ist_now().strftime('%d %b %Y')}</b>\n\n"
                f"<b>⚠️ SIDEWAYS MARKET</b>\n"
                f"Nifty 50 stocks avg 20-day return: {state['nifty']:+.2f}%\n\n"
                f"🚫 <b>NO TRADES TODAY.</b> Sit on cash."
            )
        save_state(state); return

    direction = state["direction"]
    t_emoji   = "🟢" if direction=="LONG" else "🔴"

    # ── STEP 4: Morning briefing (send once, any time before 2:50 PM) ──
    if not state["briefing_sent"] and now <= BRIEFING_END:
        log("Sending morning briefing...")
        state = do_briefing(state, bars)
        # Don't return — continue to monitoring in the same run

    # ── STEP 5: Intraday monitoring ───────────────────────────────────
    if state["briefing_sent"] and now < TRADE_END:
        for sym, w in state["watching"].items():
            if w["exited"]: continue
            data = get_stock_data(sym, bars)
            if data is None: continue

            # Check breakout (only if not yet triggered)
            if not w["triggered"]:
                if check_breakout_now(data, direction):
                    entry_time = str(data["sess"].index[-1])
                    w["triggered"]  = True
                    w["entry_time"] = entry_time
                    log(f"🚨 BREAKOUT: {sym} {direction}")

                    msg  = f"🚨 <b>TRADE ALERT — {sym}</b>\n\n"
                    msg += f"<b>{t_emoji} {direction} triggered!</b>  {ist_now().strftime('%I:%M %p')}\n\n"
                    msg += f"   Entry  : <b>₹{w['entry']:,.2f}</b>\n"
                    msg += f"   Stop   : ₹{w['stop']:,.2f}  (risk ₹{int(RISK):,})\n"
                    msg += f"   Target : <b>₹{w['target']:,.2f}</b>  (+2R = ₹{w['R']*2:.1f} move)\n"
                    msg += f"   Qty    : <b>{w['shares']} shares</b>\n"
                    msg += f"   Gap    : {w['gap']:+.2f}%  |  OR: {w['or_pct']:.2f}%\n\n"
                    msg += f"   ⚡ Place {direction} order NOW\n"
                    msg += f"   🔴 Set stop at ₹{w['stop']:,.2f} immediately\n"
                    msg += f"   🎯 Target ₹{w['target']:,.2f} — let it run"
                    send(msg)

            # Check exit (only if already in trade)
            elif not w["exited"]:
                exit_reason, exit_price = check_exit(data, w, direction)
                if exit_reason in ("stop", "target"):
                    w["exited"]      = True
                    w["exit_reason"] = exit_reason
                    w["exit_price"]  = exit_price

                    sgn   = 1 if direction=="LONG" else -1
                    gross = (exit_price - w["entry"]) * w["shares"] * sgn
                    chg   = charges(w["entry"]*w["shares"], abs(exit_price)*w["shares"])
                    net   = gross - chg

                    if exit_reason == "target":
                        head = f"✅ <b>TARGET HIT — {sym}</b> 🎉"
                    else:
                        head = f"🔴 <b>STOP HIT — {sym}</b>"

                    msg  = f"{head}\n\n"
                    msg += f"   {direction}  {w['shares']} shares\n"
                    msg += f"   Entry  : ₹{w['entry']:,.2f}\n"
                    msg += f"   Exit   : ₹{exit_price:,.2f}  [{exit_reason}]\n"
                    msg += f"   Gross  : ₹{gross:+,.0f}\n"
                    msg += f"   Charges: ₹{chg:.0f}\n"
                    msg += f"   <b>Net: ₹{net:+,.0f}</b>"
                    send(msg)
                    log(f"{exit_reason.upper()}: {sym} net ₹{net:+,.0f}")

    # ── STEP 6: 2:55 PM close reminder ───────────────────────────────
    if now >= TRADE_END and not state["close_sent"]:
        active = [s for s,v in state["watching"].items()
                  if v["triggered"] and not v["exited"]]
        if active:
            msg = f"⏰ <b>2:55 PM — CLOSE ALL POSITIONS NOW!</b>\n\n"
            msg += "Open trades to close at market price:\n\n"
            for sym in active:
                w = state["watching"][sym]
                msg += f"  {t_emoji} <b>{sym}</b>  {direction}  {w['shares']} shares\n"
                msg += f"     Entry ₹{w['entry']:,.2f}  |  Close at market\n"
            msg += "\n<b>⚠️ Do NOT hold past 3:00 PM</b>"
            send(msg)
            log("2:55 PM close-all reminder sent.")
        state["close_sent"] = True

    # ── STEP 7: 3:05 PM End-of-day summary ───────────────────────────
    if now >= EOD_START and not state["eod_sent"]:
        triggered = [(s,v) for s,v in state["watching"].items() if v["triggered"]]
        date_str  = ist_now().strftime("%d %b %Y")
        msg  = f"<b>📊 EOD SUMMARY — {date_str}</b>\n\n"
        msg += f"Regime : {'📈' if direction=='LONG' else '📉'} {state['regime']}\n"
        msg += f"Watched: {len(state['watching'])}  |  Traded: {len(triggered)}\n\n"

        total_net = 0
        for sym, v in triggered:
            ep    = v["exit_price"] or v["entry"]
            er    = v["exit_reason"] or "manual/open"
            sgn   = 1 if direction=="LONG" else -1
            gross = (ep - v["entry"]) * v["shares"] * sgn
            chg   = charges(v["entry"]*v["shares"], abs(ep)*v["shares"])
            net   = gross - chg
            total_net += net
            e_icon = "✅" if net > 0 else "🔴"
            msg += f"{e_icon} <b>{sym}</b>  {direction}\n"
            msg += f"   ₹{v['entry']:,.2f} → ₹{ep:,.2f}  [{er}]  <b>₹{net:+,.0f}</b>\n\n"

        if not triggered:
            msg += "No trades triggered today.\n"
        else:
            msg += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            icon  = "💰" if total_net > 0 else "📉"
            msg += f"{icon} <b>Total Net P&L: ₹{total_net:+,.0f}</b>"

        send(msg)
        state["eod_sent"] = True
        log(f"EOD summary sent. Total net: ₹{total_net:+,.0f}")

    save_state(state)
    log("Run complete.")

if __name__ == "__main__":
    main()
