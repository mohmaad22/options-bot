"""
Options Signal Alert Bot (Telegram) — ALERTS ONLY, NO AUTO-EXECUTION
=====================================================================
This bot monitors a watchlist of tickers (default: TSLA, AAPL, NVDA, SPY,
plus any SPACs you add), computes technical signals, and sends alerts to
a Telegram chat suggesting a possible CALL or PUT setup. It NEVER places
trades automatically — you review every alert and execute manually with
your own broker.
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PUT_YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

WATCHLIST = ["TSLA", "AAPL", "NVDA", "SPY"]
CHECK_INTERVAL_SECONDS = 5 * 60

MIN_AVG_VOLUME = 500_000
MIN_PRICE = 3.0

EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 50
RSI_PERIOD = 14
RSI_UPPER = 70
RSI_LOWER = 30
VOLUME_SPIKE_MULT = 1.5

last_alert_state = {}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("options_alert_bot")


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_data(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["EMA_FAST"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["EMA_TREND"] = df["Close"].ewm(span=EMA_TREND, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["VOL_AVG20"] = df["Volume"].rolling(window=20).mean()
    return df


def passes_liquidity_filter(df: pd.DataFrame) -> bool:
    if df.empty or len(df) < 20:
        return False
    avg_vol = df["VOL_AVG20"].iloc[-1]
    last_price = df["Close"].iloc[-1]
    if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOLUME:
        return False
    if last_price < MIN_PRICE:
        return False
    return True


def evaluate_signal(df: pd.DataFrame) -> dict:
    if len(df) < EMA_TREND + 2:
        return {"signal": None}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    bullish_cross = prev["EMA_FAST"] <= prev["EMA_SLOW"] and last["EMA_FAST"] > last["EMA_SLOW"]
    bearish_cross = prev["EMA_FAST"] >= prev["EMA_SLOW"] and last["EMA_FAST"] < last["EMA_SLOW"]

    uptrend = last["Close"] > last["EMA_TREND"]
    downtrend = last["Close"] < last["EMA_TREND"]

    volume_spike = last["Volume"] > (last["VOL_AVG20"] * VOLUME_SPIKE_MULT)

    reasons = []
    signal = None

    if bullish_cross and uptrend and volume_spike and last["RSI"] < RSI_UPPER:
        signal = "bullish"
        reasons = [
            f"EMA{EMA_FAST} قطع فوق EMA{EMA_SLOW}",
            f"السعر فوق EMA{EMA_TREND} (اتجاه صاعد)",
            f"الفوليوم {last['Volume']/last['VOL_AVG20']:.1f}x المعدل",
            f"RSI={last['RSI']:.1f} (مو overbought)",
        ]
    elif bearish_cross and downtrend and volume_spike and last["RSI"] > RSI_LOWER:
        signal = "bearish"
        reasons = [
            f"EMA{EMA_FAST} قطع تحت EMA{EMA_SLOW}",
            f"السعر تحت EMA{EMA_TREND} (اتجاه هابط)",
            f"الفوليوم {last['Volume']/last['VOL_AVG20']:.1f}x المعدل",
            f"RSI={last['RSI']:.1f} (مو oversold)",
        ]

    return {
        "signal": signal,
        "reasons": reasons,
        "price": last["Close"],
        "rsi": last["RSI"],
    }


def suggest_option_params(price: float) -> tuple:
    strike = round(price / 5) * 5 if price > 50 else round(price)
    expiry = (datetime.now() + timedelta(weeks=3)).strftime("%Y-%m-%d (approx)")
    return strike, expiry


async def send_alert(app: Application, ticker: str, result: dict):
    signal = result["signal"]
    price = result["price"]
    reasons = "\n".join(f"  • {r}" for r in result["reasons"])
    strike, expiry = suggest_option_params(price)
    direction = "CALL (شراء - صاعد)" if signal == "bullish" else "PUT (بيع - هابط)"

    text = (
        f"🔔 *{ticker}* — احتمال فرصة *{direction}*\n\n"
        f"السعر: ${price:.2f}\n"
        f"الأسباب:\n{reasons}\n\n"
        f"مرجع تقريبي فقط — سعر تنفيذ قريب من السعر الحالي ≈ {strike}, "
        f"تاريخ انتهاء تقريبي ≈ {expiry}.\n"
        f"⚠️ تأكد بنفسك من سلسلة الأوبشنز الفعلية (الفرق بين سعري البيع والشراء، "
        f"عدد العقود المفتوحة) قبل أي خطوة. هذا تنبيه فقط مو صفقة. راجع ونفذ يدويًا.\n"
        f"⚠️ هذا مو نصيحة مالية. ما فيه ضمان لأي نسبة نجاح."
    )
    await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")


async def scan_once(app: Application):
    for ticker in WATCHLIST:
        try:
            df = fetch_data(ticker)
            if not passes_liquidity_filter(df):
                logger.info(f"{ticker}: skipped (liquidity filter)")
                continue

            result = evaluate_signal(df)
            signal = result["signal"]

            if signal and last_alert_state.get(ticker) != signal:
                await send_alert(app, ticker, result)
                last_alert_state[ticker] = signal
                logger.info(f"{ticker}: sent {signal} alert")
            elif not signal:
                last_alert_state[ticker] = None
        except Exception as e:
            logger.error(f"{ticker}: error during scan — {e}")


async def scan_loop(app: Application):
    while True:
        await scan_once(app)
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "بوت التنبيهات شغال ✅\n"
        f"يراقب: {', '.join(WATCHLIST)}\n\n"
        "الأوامر:\n"
        "/status – قراءة حالية للمؤشرات على القائمة\n"
        "/add TICKER – إضافة رمز للقائمة\n"
        "/remove TICKER – حذف رمز من القائمة\n\n"
        "هذا البوت يرسل تنبيهات فقط. ما ينفذ أي صفقة أبدًا."
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    for ticker in WATCHLIST:
        try:
            df = fetch_data(ticker)
            if df.empty:
                lines.append(f"{ticker}: لا توجد بيانات")
                continue
            last = df.iloc[-1]
            liquid = "✅" if passes_liquidity_filter(df) else "⚠️ سيولة ضعيفة"
            lines.append(
                f"*{ticker}* {liquid} — ${last['Close']:.2f} | "
                f"RSI {last['RSI']:.1f} | "
                f"EMA{EMA_FAST}/{EMA_SLOW}: {last['EMA_FAST']:.2f}/{last['EMA_SLOW']:.2f}"
            )
        except Exception as e:
            lines.append(f"{ticker}: خطأ ({e})")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("الاستخدام: /add TICKER")
        return
    ticker = context.args[0].upper()
    if ticker not in WATCHLIST:
        WATCHLIST.append(ticker)
        await update.message.reply_text(f"تمت إضافة {ticker} للقائمة.")
    else:
        await update.message.reply_text(f"{ticker} موجود بالقائمة أصلاً.")


async def cmd_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("الاستخدام: /remove TICKER")
        return
    ticker = context.args[0].upper()
    if ticker in WATCHLIST:
        WATCHLIST.remove(ticker)
        await update.message.reply_text(f"تم حذف {ticker} من القائمة.")
    else:
        await update.message.reply_text(f"{ticker} مو موجود بالقائمة.")


async def post_init(app: Application):
    asyncio.create_task(scan_loop(app))


def main():
    if TELEGRAM_BOT_TOKEN == "PUT_YOUR_TOKEN_HERE":
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var first.")
    if TELEGRAM_CHAT_ID == "PUT_YOUR_CHAT_ID_HERE":
        raise SystemExit("Set TELEGRAM_CHAT_ID env var first.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("add", cmd_add))
    app.add_handler(CommandHandler("remove", cmd_remove))

    logger.info("Bot starting (alerts only, no auto-execution)...")
    app.run_polling()


if __name__ == "__main__":
    main()
