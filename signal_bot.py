import yfinance as yf
import pandas as pd
import pandas_ta as ta
import asyncio
import schedule
import time
import logging
from datetime import datetime
from telegram import Bot
from telegram.constants import ParseMode

BOT_TOKEN  = "YOUR_BOT_TOKEN_HERE"
CHAT_ID    = "YOUR_CHAT_ID_HERE"

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AVGO", "ORCL", "NFLX",
    "ADBE", "AMD", "QCOM", "INTC", "CSCO",
    "JPM", "V", "MA", "UNH", "XOM",
    "LLY", "JNJ", "COST", "HD", "WMT",
]

STOP_LOSS_PCT   = -15.0
TAKE_PROFIT_PCT =  30.0
SCAN_INTERVAL   = 60
MIN_SIGNALS     = 6

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")]
)
log = logging.getLogger(__name__)

entry_prices = {}
last_alerts = {}

def fetch_data(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 50:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        log.error(f"{symbol} fetch error: {e}")
        return None

def compute_signals(df):
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    sigs   = {}
    vals   = {}

    rsi_s = ta.rsi(close, length=14)
    rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50
    vals["RSI"] = f"{rsi:.1f}"
    sigs["RSI"] = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "NEUTRAL"

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and len(macd_df.columns) >= 2:
        macd_col = [c for c in macd_df.columns if "MACD_" in c and "s" not in c.lower() and "h" not in c.lower()]
        sig_col  = [c for c in macd_df.columns if "MACDs" in c]
        if macd_col and sig_col:
            macd_v = float(macd_df[macd_col[0]].iloc[-1])
            macd_s = float(macd_df[sig_col[0]].iloc[-1])
            vals["MACD"] = f"M:{macd_v:.3f} S:{macd_s:.3f}"
            sigs["MACD"] = "BUY" if macd_v > macd_s else "SELL"
        else:
            sigs["MACD"] = "NEUTRAL"; vals["MACD"] = "N/A"
    else:
        sigs["MACD"] = "NEUTRAL"; vals["MACD"] = "N/A"

    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        upper_col = [c for c in bb.columns if "BBU" in c]
        lower_col = [c for c in bb.columns if "BBL" in c]
        if upper_col and lower_col:
            ub = float(bb[upper_col[0]].iloc[-1])
            lb = float(bb[lower_col[0]].iloc[-1])
            cp = float(close.iloc[-1])
            vals["Bollinger"] = f"U:{ub:.2f} L:{lb:.2f}"
            sigs["Bollinger"] = "BUY" if cp < lb else "SELL" if cp > ub else "NEUTRAL"
        else:
            sigs["Bollinger"] = "NEUTRAL"; vals["Bollinger"] = "N/A"
    else:
        sigs["Bollinger"] = "NEUTRAL"; vals["Bollinger"] = "N/A"

    ema50  = ta.ema(close, length=50)
    ema200 = ta.ema(close, length=200)
    if ema50 is not None and ema200 is not None:
        e50  = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])
        vals["EMA Cross"] = f"50:{e50:.2f} 200:{e200:.2f}"
        sigs["EMA Cross"] = "BUY" if e50 > e200 else "SELL"
    else:
        sigs["EMA Cross"] = "NEUTRAL"; vals["EMA Cross"] = "N/A"

    vol_avg   = float(volume.rolling(20).mean().iloc[-1])
    vol_now   = float(volume.iloc[-1])
    vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1
    day_chg   = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    vals["Volume"] = f"x{vol_ratio:.2f}"
    if vol_ratio > 2:
        sigs["Volume"] = "BUY" if day_chg > 0 else "SELL"
    else:
        sigs["Volume"] = "NEUTRAL"

    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None and len(stoch.columns) >= 2:
        k = float(stoch.iloc[-1, 0])
        d = float(stoch.iloc[-1, 1])
        vals["Stochastic"] = f"%K:{k:.1f} %D:{d:.1f}"
        sigs["Stochastic"] = ("BUY" if k < 20 and d < 20 else
                              "SELL" if k > 80 and d > 80 else "NEUTRAL")
    else:
        sigs["Stochastic"] = "NEUTRAL"; vals["Stochastic"] = "N/A"

    atr_s = ta.atr(high, low, close, length=14)
    if atr_s is not None:
        atr_v   = float(atr_s.iloc[-1])
        cp      = float(close.iloc[-1])
        atr_pct = atr_v / cp * 100
        vals["ATR"] = f"${atr_v:.2f} ({atr_pct:.1f}%)"
        sigs["ATR"] = "NEUTRAL" if atr_pct < 1.5 else ("BUY" if day_chg > 0 else "SELL")
    else:
        sigs["ATR"] = "NEUTRAL"; vals["ATR"] = "N/A"

    cci_s = ta.cci(high, low, close, length=20)
    if cci_s is not None:
        cci_v = float(cci_s.iloc[-1])
        vals["CCI"] = f"{cci_v:.1f}"
        sigs["CCI"] = "BUY" if cci_v < -100 else "SELL" if cci_v > 100 else "NEUTRAL"
    else:
        sigs["CCI"] = "NEUTRAL"; vals["CCI"] = "N/A"

    obv_s = ta.obv(close, volume)
    if obv_s is not None and len(obv_s) >= 5:
        obv_now  = float(obv_s.iloc[-1])
        obv_prev = float(obv_s.iloc[-5])
        vals["OBV"] = f"{obv_now/1e6:.2f}M"
        sigs["OBV"] = "BUY" if obv_now > obv_prev else "SELL"
    else:
        sigs["OBV"] = "NEUTRAL"; vals["OBV"] = "N/A"

    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        wr = float(willr.iloc[-1])
        vals["Williams %R"] = f"{wr:.1f}"
        sigs["Williams %R"] = "BUY" if wr < -80 else "SELL" if wr > -20 else "NEUTRAL"
    else:
        sigs["Williams %R"] = "NEUTRAL"; vals["Williams %R"] = "N/A"

    buy_count  = sum(1 for v in sigs.values() if v == "BUY")
    sell_count = sum(1 for v in sigs.values() if v == "SELL")

    if buy_count >= MIN_SIGNALS:
        summary = "STRONG BUY"
    elif buy_count >= 4:
        summary = "BUY"
    elif sell_count >= MIN_SIGNALS:
        summary = "STRONG SELL"
    elif sell_count >= 4:
        summary = "SELL"
    else:
        summary = "NEUTRAL"

    return {
        "signals": sigs, "values": vals, "summary": summary,
        "buy_count": buy_count, "sell_count": sell_count,
        "price": float(close.iloc[-1]), "day_change": day_chg,
    }

def format_signal_message(symbol, result):
    price   = result["price"]
    day_chg = result["day_change"]
    summary = result["summary"]
    sigs    = result["signals"]
    vals    = result["values"]
    arrow   = "+" if day_chg >= 0 else ""
    lines   = [
        f"SIGNAL: {symbol} — {summary}",
        f"מחיר: ${price:.2f} ({arrow}{day_chg:.2f}%)",
        f"{result['buy_count']}/10 קנייה | {result['sell_count']}/10 מכירה",
        "——————————————",
    ]
    for name, sig in sigs.items():
        icon = "BUY" if sig == "BUY" else "SELL" if sig == "SELL" else "—"
        lines.append(f"{icon} {name}: {vals.get(name,'')}")
    lines.append(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M')}")
    return "\n".join(lines)

def format_alert_message(symbol, alert_type, price, entry, pct):
    if alert_type == "STOP_LOSS":
        return (f"STOP LOSS - {symbol}\n"
                f"ירידה של {abs(pct):.1f}% מכניסה!\n"
                f"מחיר נוכחי: ${price:.2f}\n"
                f"מחיר כניסה: ${entry:.2f}\n"
                f"מומלץ לשקול יציאה מהעסקה")
    else:
        return (f"TAKE PROFIT - {symbol}\n"
                f"עלייה של {pct:.1f}% מכניסה!\n"
                f"מחיר נוכחי: ${price:.2f}\n"
                f"מחיר כניסה: ${entry:.2f}\n"
                f"מומלץ לשקול מימוש רווחים")

async def send_message(text):
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)

async def scan_all():
    log.info(f"Scanning {len(WATCHLIST)} stocks...")
    strong_signals = []
    alert_messages = []

    for symbol in WATCHLIST:
        try:
            df = fetch_data(symbol)
            if df is None:
                continue
            result = compute_signals(df)
            price  = result["price"]

            if symbol not in entry_prices:
                entry_prices[symbol] = price

            entry          = entry_prices[symbol]
            pct_from_entry = (price - entry) / entry * 100

            alert_key_sl = f"{symbol}_SL"
            if pct_from_entry <= STOP_LOSS_PCT and not last_alerts.get(alert_key_sl):
                alert_messages.append(format_alert_message(symbol, "STOP_LOSS", price, entry, pct_from_entry))
                last_alerts[alert_key_sl] = True
            elif pct_from_entry > STOP_LOSS_PCT + 5:
                last_alerts[alert_key_sl] = False

            alert_key_tp = f"{symbol}_TP"
            if pct_from_entry >= TAKE_PROFIT_PCT and not last_alerts.get(alert_key_tp):
                alert_messages.append(format_alert_message(symbol, "TAKE_PROFIT", price, entry, pct_from_entry))
                last_alerts[alert_key_tp] = True

            if "STRONG" in result["summary"] or result["buy_count"] >= 6 or result["sell_count"] >= 6:
                strong_signals.append(format_signal_message(symbol, result))

        except Exception as e:
            log.error(f"Error scanning {symbol}: {e}")

    for msg in alert_messages + strong_signals:
        await send_message(msg)
        await asyncio.sleep(0.5)

    if not strong_signals and not alert_messages:
        log.info("No strong signals this scan.")

def run_scan():
    asyncio.run(scan_all())

def main():
    log.info("Bot starting...")
    run_scan()
    schedule.every(SCAN_INTERVAL).minutes.do(run_scan)
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
