"""
bot_v2.py -- LSTM-PPO XAUUSD trading bot: three entry strategies sharing
one agent, one GBDT win-rate filter, one 1m execution loop.

A second, independent strategy set alongside bot.py's ICT/SMC bot, built
the same way bot.py itself is: it *executes* bar-by-bar on 1-minute
candles (order fills, SL/TP hit detection all happen at 1m granularity)
but *analyzes* higher timeframes -- every 1m row sees the full indicator
+ entry-signal stack recomputed on 5m, 15m, 1h and 4h candles resampled
from that same 1m data (merged back on with a 5m_/15m_/1h_/4h_ prefix,
forward-filled so a bar only ever sees the most recently *closed*
higher-tf candle -- no lookahead), plus a second stack computed directly
on the 1m data itself (prefixed 1m_). This is exactly bot.py's own
multi-timeframe merge (see its add_indicators()), just run across five
timeframes instead of two, and with a much smaller indicator stack per
timeframe. On any bar, at most one of the three strategies below can
produce a candidate trade -- see _select_candidate() for the priority
order used when more than one would fire at once.

STRATEGY 1 -- stoch/%R zone-breakout (see _add_base_indicators(),
_signal_columns()), 1:4 RR (50-pip SL / 200-pip TP = TP_PIPS):

  Per analyzed timeframe (5m/15m/1h/4h):
    - EMA 7 / 21, each vs. price (distance) and its own slope
    - HalfTrend (bullish/bearish), plus its own distance from price
    - ADX (+DI/-DI) as a trend-strength floor
    - Stochastic oscillator (%K / %K-smooth) and Williams %R
    - The stoch/%R zone-breakout entry signal (below)

  A bar counts as oversold once %K < 20 OR %R < -80, overbought once
  %K > 80 OR %R > -20. Once either state has held for ZONE_BARS (=5)
  consecutive bars *of that timeframe*, that run's high/low forms a
  "zone" -- a breakout above the zone high (confirmed by %K crossing
  back above %K-smooth) is bullish; a breakdown below the zone low (%K
  crossing back below %K-smooth) is bearish. A candidate fires whenever
  ANY of the 4 timeframes signals a breakout with that timeframe's own
  ADX clearing ADX_MIN. Direction filter: only becomes a trade if the
  1h and/or 4h HalfTrend agrees (--htf-mode "any"/"both"). This is one
  reasonable reading of "stoch k </> k smooth, and/or %r, wait 5
  candles in ob/os zone, then breakout, executed on 5m/15m/1h/4h, only
  in direction with 1h and/or 4h halftrend" -- adjust the thresholds
  or _signal_columns()'s combination logic if a different reading was
  intended.

STRATEGIES 2 & 3 -- both computed directly on the 1m data itself (see
_add_m1_levels(), _level_signal_columns()), sharing a 1:2 RR (the same
50-pip SL / a 100-pip TP = LEVEL_TP_PIPS), and both gated by 1m
HalfTrend:

  2. PDH/PDL + Asia high/low break, retest and reversal (poi_reversal):
     a candidate fires in whichever direction the 1m HalfTrend
     currently points, on any bar where price is within
     POI_REACH_PIPS (30) of the previous day's high, the previous
     day's low, the current Asia session's high, or its low (PDH/PDL/
     Asia-high/Asia-low distance, ported from bot.py's own
     PDHDistance/PDLDistance/AsiaHighDistance/AsiaLowDistance).
     HalfTrend's live state stands in for whichever of
     breakout/retest/reversal is actually happening at the level --
     continuing through it trades as a breakout, flipping at it trades
     as a reversal -- rather than classifying the three separately.

  3. OB mitigation (ob_mitigation): a candidate fires when the 1m
     HalfTrend *flips* direction (not just agrees) on a bar that's
     also mitigating -- within the same POI_REACH_PIPS reach, via
     OBMitigation(), ported from bot.py's own BullishOB/BearishOB/
     OBMitigation -- a same-direction order block: a bullish/demand OB
     mitigated with HalfTrend flipping up is a reversal long, a
     bearish/supply OB mitigated with HalfTrend flipping down is a
     reversal short.

Across all three strategies, the LSTM-PPO agent decides whether to
actually take whichever candidate wins (or hold) -- the deterministic
technical stack always picks direction (and which strategy, and that
strategy's TP), the agent only decides timing. Actions are 0=buy,
1=sell, 2=hold (BUY/SELL/HOLD below), per spec.

GBDT (XGBoost) win-rate filter -- GBDTWinRateFilter, shared across all
three strategies (one filter, one feature vector covering every
strategy's signals, one --min-winrate bar applied the same way to the
1:4 RR stoch/%R breakout and the two 1:2 RR level strategies alike, not
tiered per RR) -- has to predict a win rate clearing that bar
(BASE_MIN_WINRATE, 35% by default; override with --min-winrate for
both --train and --test) before a buy/sell from any strategy is
allowed through. The filter only starts *gating* trades once 5 simulated
training weeks have accumulated (WEEKS_BEFORE_FILTER) -- before that it
keeps fitting/accumulating samples in the background but never blocks a
trade, so the first weeks of training aren't starved waiting on data
that doesn't exist yet.

Position risk scales with the filter's confidence: once it's active,
every full 10 percentage points its predicted win rate clears above the
minimum bar adds one more unit of the base --risk to the position
(risk_multiplier() below) -- so a barely-qualifying setup risks the
plain --risk amount, a strongly-favoured one risks several multiples of
it (capped at MAX_RISK_MULTIPLIER).

Weekly stats (trade count, PnL, R-multiple, win rate, mean win/loss,
streaks, Z-score, profit factor, recovery factor, Sharpe, Sortino, GBDT
filter state, and a trades-by-strategy breakdown) print every simulated
training week, same cadence and metrics as bot.py's own weekly report
(plus the strategy breakdown, which bot.py's single-strategy report has
no need for).

Checkpoints and the GBDT filter's sample pickle are saved under a
directory + tag distinct from bot.py's (SAVE_DIR / model_tag()) so the
two bots -- different feature sets, different state_size -- never
collide or load each other's incompatible checkpoints.

Usage mirrors bot.py:
    python bot_v2.py --train
    python bot_v2.py --test --symbol XAUUSD-STDc --risk 0.01
    python bot_v2.py --train --test --risk 0.02 --htf-mode both
"""

import pandas as pd
import numpy as np
import os
import pickle
import math
import time
import argparse
import multiprocessing
from io import StringIO
from collections import deque
from datetime import datetime, timedelta

from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# MetaTrader5 only ships Windows wheels and needs a running terminal --
# imported lazily/defensively so training and indicator work stay usable
# on any platform; test_bot() fails loudly (not at import time) if it's
# missing.
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ==========================================================================
# ACTIONS -- 0=buy, 1=sell, 2=hold, per spec (note this is NOT the same
# ordering bot.py uses for its own agent).
# ==========================================================================
ACTIONS = ["buy", "sell", "hold"]
BUY, SELL, HOLD = 0, 1, 2

# ==========================================================================
# STRATEGY CONSTANTS
# ==========================================================================
SEQ_LEN = 15                    # LSTM lookback window, in 1m bars -- matches bot.py's own SEQ_LEN
SL_PIPS = 50.0                  # fixed stop-loss
RR_RATIO = 4.0                  # 1:4 risk:reward
TP_PIPS = SL_PIPS * RR_RATIO    # 200-pip target
PIP_VALUE = 0.1                 # $ per pip for XAUUSD, matches bot.py's convention
COMMISSION = 0.6                # pips, subtracted from every closed trade in the backtest

ADX_MIN = 20                    # trend-strength floor a timeframe's breakout must clear to count
ZONE_BARS = 5                   # consecutive OB/OS bars (of whichever tf) required before a breakout can fire
STOCH_OS, STOCH_OB = 20, 80     # %K oversold / overbought thresholds
WR_OS, WR_OB = -80, -20         # Williams %R oversold / overbought thresholds

WEEKS_BEFORE_FILTER = 5         # GBDT win-rate filter starts gating after this many training weeks
TRADING_WEEK_BARS = 1440 * 5    # 1m bars in a 5-day trading week -- same definition as bot.py's save_count

# Breakeven at this 1:4 RR is 1/(1+RR_RATIO) = 20%, so the spec's
# "breakeven * 1.1" would put this at 22% -- overridden to a flat 35%
# per explicit request.
BASE_MIN_WINRATE = 0.35

RISK_STEP = 0.10                # +10 predicted-win-rate points
MAX_RISK_MULTIPLIER = 5.0       # cap on how many multiples of --risk one trade can size to

# --- PDH/PDL + Asia high/low reversal strategy, and OB-mitigation
# strategy -- see _add_m1_levels()/_level_signal_columns() below. Both
# share the same 50-pip SL (SL_PIPS above) but trade a 1:2 RR, half
# the stoch-breakout strategy's 1:4.
LEVEL_RR_RATIO = 2.0
LEVEL_TP_PIPS = SL_PIPS * LEVEL_RR_RATIO  # 100-pip target
POI_REACH_PIPS = 30             # "within 30 pips of poi" -- shared by both level-based strategies
OB_MULTIPLIER = 1.5             # bot.py's BullishOB/BearishOB impulse-candle size multiplier
OB_LOOKBACK = 72                # bot.py's OBMitigation() lookback, in bars

MAGIC = 234567                  # MT5 order/position tag for this bot -- distinct from bot.py's 123456
SAVE_DIR = "LSTM-PPO-saves-stoch-halftrend"  # separate from bot.py's LSTM-PPO-saves, see module docstring


def model_tag(symbol):
    # Namespaces checkpoints/GBDT pickles under this strategy's own tag
    # so bot.py's loadcheckpoint() (which matches on "symbol in filename")
    # never picks up one of this bot's incompatible (different
    # state_size) checkpoints, and vice versa.
    return f"{symbol}-stoch-halftrend"


# ==========================================================================
# INDICATORS
# ==========================================================================

def EMA(df, period):
    return df["Close"].ewm(span=period, adjust=False).mean().round(2)


def ADX(df, period=14):
    """Returns +DI, -DI and ADX using Wilder's smoothing. Columns
    required: High, Low, Close. (Same implementation as bot.py's ADX.)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    dn = -low.diff()

    plus_dm_array = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm_array = np.where((dn > up) & (dn > 0), dn, 0.0)

    plus_dm = pd.Series(plus_dm_array, index=df.index)
    minus_dm = pd.Series(minus_dm_array, index=df.index)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return round(adx, 2), round(plus_di, 2), round(minus_di, 2)


def STOCH(df, period=14, smooth_d=3):
    """Returns %K and %K-smooth. Columns required: High, Low, Close.
    (Same implementation as bot.py's STOCH.)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=smooth_d).mean()

    return round(k, 2), round(d, 2)


def WilliamsR(df, period=14):
    """Williams %R: -100 * (highest_high - close) / (highest_high -
    lowest_low). Ranges -100 (oversold extreme) to 0 (overbought
    extreme). Columns required: High, Low, Close."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    r = -100 * (highest_high - close) / (highest_high - lowest_low)

    return round(r, 2)


def HalfTrend(df, amplitude=2):
    """Port of the "HalfTrend" indicator (Alex Orekhov / everget) -- a
    trailing trend band that flips once price and both extreme-EMAs
    confirm a break of the opposite side's recent range. Returns
    (bullish_halftrend, bearish_halftrend, half_trend_line). (Same
    implementation as bot.py's HalfTrend.)"""
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    n = len(df)

    high_ma = df["High"].ewm(span=amplitude, adjust=False).mean().to_numpy()
    low_ma = df["Low"].ewm(span=amplitude, adjust=False).mean().to_numpy()
    highest_high = df["High"].rolling(amplitude).max().to_numpy()
    lowest_low = df["Low"].rolling(amplitude).min().to_numpy()

    trend = np.zeros(n, dtype=np.int8)
    next_trend = np.zeros(n, dtype=np.int8)
    max_low = np.full(n, np.nan)
    min_high = np.full(n, np.nan)
    half_trend_line = np.full(n, np.nan)

    start = amplitude
    if start >= n:
        empty = np.zeros(n, dtype=np.int8)
        return empty, empty, half_trend_line

    max_low[start] = low[start]
    min_high[start] = high[start]
    half_trend_line[start] = close[start]

    for i in range(start + 1, n):

        if next_trend[i - 1] == 0:
            max_low[i] = max(max_low[i - 1], lowest_low[i])

            if high_ma[i] < max_low[i] and close[i] < low[i - 1]:
                trend[i] = 1
                next_trend[i] = 1
                min_high[i] = highest_high[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = 0
                min_high[i] = min_high[i - 1]

        else:
            min_high[i] = min(min_high[i - 1], highest_high[i])

            if low_ma[i] > min_high[i] and close[i] > high[i - 1]:
                trend[i] = 0
                next_trend[i] = 0
                max_low[i] = lowest_low[i]
            else:
                trend[i] = trend[i - 1]
                next_trend[i] = 1
                max_low[i] = max_low[i - 1]

        if trend[i] == 0:
            half_trend_line[i] = (
                max(max_low[i], half_trend_line[i - 1])
                if trend[i - 1] == 0
                else max_low[i]
            )
        else:
            half_trend_line[i] = (
                min(min_high[i], half_trend_line[i - 1])
                if trend[i - 1] == 1
                else min_high[i]
            )

    bullish_halftrend = (trend == 0).astype(np.int8)
    bearish_halftrend = (trend == 1).astype(np.int8)

    return bullish_halftrend, bearish_halftrend, half_trend_line


def _run_length(mask):
    """Consecutive-True run length of a boolean Series, restarting at 0
    wherever it's False -- e.g. [F,T,T,F,T,T,T] -> [0,1,2,0,1,2,3]."""
    streak_id = (~mask).cumsum()
    run = mask.groupby(streak_id).cumcount() + 1
    return run.where(mask, 0)


def stoch_r_zone_breakout(df, zone_bars=ZONE_BARS):
    """Entry-gate signal: a bar is oversold once %K < STOCH_OS or %R <
    WR_OS, overbought once %K > STOCH_OB or %R > WR_OB. Once either
    state has held zone_bars consecutive bars, a close breaking above
    that run's High (for oversold) or below its Low (for overbought),
    confirmed by %K crossing %K-smooth in the same direction, is a
    breakout signal. Returns (bull_breakout, bear_breakout, os_streak,
    ob_streak), all aligned to df.index. Requires columns k, k_smooth,
    williams_r, High, Low, Close (i.e. run after STOCH/WilliamsR)."""
    k = df["k"]
    k_smooth = df["k_smooth"]
    wr = df["williams_r"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    oversold = (k < STOCH_OS) | (wr < WR_OS)
    overbought = (k > STOCH_OB) | (wr > WR_OB)

    os_streak = _run_length(oversold)
    ob_streak = _run_length(overbought)

    # Rolling zone_bars high/low, shifted so "the zone" only ever refers
    # to bars strictly before the one being evaluated -- no lookahead.
    zone_high = high.rolling(zone_bars).max().shift(1)
    zone_low = low.rolling(zone_bars).min().shift(1)

    zone_formed_os = os_streak.shift(1) >= zone_bars
    zone_formed_ob = ob_streak.shift(1) >= zone_bars

    bull_breakout = (
        zone_formed_os & (close > zone_high) & (k > k_smooth)
    ).fillna(False)

    bear_breakout = (
        zone_formed_ob & (close < zone_low) & (k < k_smooth)
    ).fillna(False)

    return bull_breakout, bear_breakout, os_streak, ob_streak


# ==========================================================================
# LEVEL / ORDER-BLOCK INDICATORS -- ported from bot.py, for the PDH/PDL
# + Asia high/low reversal strategy and the OB-mitigation strategy
# below (both gated by 1m HalfTrend, computed via _add_m1_levels()).
# ==========================================================================

def GetRange(df):
    """Candle range in pips -- bot.py's own convention,
    (High-Low)*10, i.e. already pip units given PIP_VALUE=0.1."""
    return (df["High"] - df["Low"]) * 10


def RangeMA(df, period=14):
    """Rolling mean of df["range"] (GetRange()). Ported from bot.py's
    RangeMA()."""
    return round(df["range"].rolling(period).mean(), 2)


def BullishOB(df, multiplier=OB_MULTIPLIER):
    """Bullish (demand) order block: the last down-candle before an
    up-candle at least `multiplier`x its size, with the down-candle's
    own body bigger than the recent average range. Ported verbatim
    from bot.py's BullishOB(). Requires df["range_ma"] (GetRange() +
    RangeMA()) to already be set."""
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bearish = df["Close"] < df["Open"]
    next_bullish = df["Close"].shift(-1) > df["Open"].shift(-1)

    return (
        bearish
        & next_bullish
        & (body > df["range_ma"])
        & (next_body >= body * multiplier)
    ).astype(int)


def BearishOB(df, multiplier=OB_MULTIPLIER):
    """Bearish (supply) order block -- mirror of BullishOB(). Ported
    verbatim from bot.py's BearishOB()."""
    body = (df["Close"] - df["Open"]).abs()
    next_body = body.shift(-1)

    bullish = df["Close"] > df["Open"]
    next_bearish = df["Close"].shift(-1) < df["Open"].shift(-1)

    return (
        bullish
        & next_bearish
        & (body > df["range_ma"])
        & (next_body >= body * multiplier)
    ).astype(int)


def OBMitigation(df, threshold_pips=POI_REACH_PIPS, lookback=OB_LOOKBACK):
    """True on any bar within `threshold_pips` of a prior order
    block's own candle low/high (not the binary bullish_ob/bearish_ob
    flag itself, which isn't a price level). Same algorithm as bot.py's
    OBMitigation(), adapted to take its threshold in pips -- bot.py's
    own version takes a raw-price threshold (its default of 30 is
    literally $30, i.e. 300 pips at PIP_VALUE=0.1); this version's
    default of 30 is genuinely 30 pips, per spec ("same 30 pip
    reach")."""
    threshold = threshold_pips * PIP_VALUE

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

    bullish_ob = df["bullish_ob"].to_numpy().astype(bool)
    bearish_ob = df["bearish_ob"].to_numpy().astype(bool)

    bull = np.zeros(len(df), dtype=np.bool_)
    bear = np.zeros(len(df), dtype=np.bool_)

    n = len(df)

    for i in range(n):
        start = max(0, i - lookback)

        for j in range(i - 1, start - 1, -1):
            if bullish_ob[j] and abs(low[i] - low[j]) <= threshold:
                bull[i] = True
                break

        for j in range(i - 1, start - 1, -1):
            if bearish_ob[j] and abs(high[i] - high[j]) <= threshold:
                bear[i] = True
                break

    return bull, bear


def PDHDistance(df):
    """Previous day's high minus current close, in raw price. Ported
    verbatim from bot.py's PDHDistance(). Requires a DatetimeIndex."""
    day = df.index.date

    daily_high = df["High"].groupby(day).transform("max")

    pdh = (
        daily_high
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return pdh - df["Close"]


def PDLDistance(df):
    """Current close minus previous day's low, in raw price. Ported
    verbatim from bot.py's PDLDistance()."""
    day = df.index.date

    daily_low = df["Low"].groupby(day).transform("min")

    pdl = (
        daily_low
        .groupby(day)
        .first()
        .shift(1)
        .reindex(day)
        .to_numpy()
    )

    return df["Close"] - pdl


def AsiaHighDistance(df):
    """Current Asia session's high minus current close, in raw price.
    Adapted from bot.py's AsiaHighDistance(): its session mask,
    `(hour >= 1) | (hour <= 9)`, is a bug (that OR covers nearly the
    entire day, not a session) -- this uses AND, matching that
    function's own "01:00-08:59" docstring and bot.py's GetKillzone()
    asia-session convention."""
    asia = (df.index.hour >= 1) & (df.index.hour < 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_high = (
        df["High"]
        .where(asia)
        .groupby(trade_day)
        .transform("max")
        .ffill()
    )

    return asia_high - df["Close"]


def AsiaLowDistance(df):
    """Current close minus current Asia session's low, in raw price.
    Same OR->AND session-mask fix as AsiaHighDistance() above."""
    asia = (df.index.hour >= 1) & (df.index.hour < 9)

    trade_day = (df.index - pd.Timedelta(hours=24)).date

    asia_low = (
        df["Low"]
        .where(asia)
        .groupby(trade_day)
        .transform("min")
        .ffill()
    )

    return df["Close"] - asia_low


# ==========================================================================
# FEATURE PIPELINE
# ==========================================================================

# The indicator + entry-signal stack computed independently on each
# analyzed timeframe (see _add_base_indicators()) -- never on the raw
# 1m data itself, which only supplies OHLC for execution.
BASE_INDICATOR_FEATURES = [
    "k", "k_smooth", "williams_r", "adx", "+di", "-di",
    "EMA7_dist", "EMA7_slope", "EMA21_dist", "EMA21_slope",
    "bullish_halftrend", "bearish_halftrend", "halftrend_dist",
    "os_streak", "ob_streak", "bull_breakout", "bear_breakout",
]

# (column prefix, pandas resample frequency) -- the four timeframes
# analyzed off of the raw 1m execution data. 1h/4h double as the
# direction filter (see _htf_ok_columns()); all four feed the
# stoch/%R zone-breakout entry signal (see bull_signal/bear_signal in
# train_bot()/test_bot()).
ANALYZED_TIMEFRAMES = (
    ("5m", "5min"),
    ("15m", "15min"),
    ("1h", "1h"),
    ("4h", "4h"),
)

# The indicator stack computed directly on the raw 1m execution data
# itself (never resampled) -- see _add_m1_levels() -- feeding the
# PDH/PDL + Asia high/low reversal strategy and the OB-mitigation
# strategy, both gated by this same 1m HalfTrend.
M1_LEVEL_FEATURES = [
    "bullish_halftrend", "bearish_halftrend",
    "bullish_ob", "bearish_ob",
    "bullish_ob_mitigation", "bearish_ob_mitigation",
    "pdh_dist", "pdl_dist", "asia_high_dist", "asia_low_dist",
]

FEATURES = [
    f"{prefix}_{col}"
    for prefix, _ in ANALYZED_TIMEFRAMES
    for col in BASE_INDICATOR_FEATURES
] + [f"1m_{col}" for col in M1_LEVEL_FEATURES]


def _add_base_indicators(df):
    """Adds the shared indicator + zone-breakout stack
    (BASE_INDICATOR_FEATURES) to an OHLC dataframe at whatever
    timeframe it's given. Called once per analyzed timeframe by
    add_indicators() below -- never on the raw 1m data directly."""
    df = df.copy()

    df["adx"], df["+di"], df["-di"] = ADX(df)
    df["k"], df["k_smooth"] = STOCH(df)
    df["williams_r"] = WilliamsR(df)

    df["EMA7"] = EMA(df, 7)
    df["EMA7_slope"] = df["EMA7"].diff()
    df["EMA7_dist"] = df["Close"] - df["EMA7"]

    df["EMA21"] = EMA(df, 21)
    df["EMA21_slope"] = df["EMA21"].diff()
    df["EMA21_dist"] = df["Close"] - df["EMA21"]

    df["bullish_halftrend"], df["bearish_halftrend"], halftrend_line = HalfTrend(df)
    df["halftrend_dist"] = df["Close"] - halftrend_line

    df["bull_breakout"], df["bear_breakout"], df["os_streak"], df["ob_streak"] = (
        stoch_r_zone_breakout(df)
    )

    return df[BASE_INDICATOR_FEATURES]


def _add_m1_levels(df):
    """Adds M1_LEVEL_FEATURES -- HalfTrend plus the order-block and
    PDH/PDL/Asia-session levels the two level-based strategies trade
    off of -- directly on the raw 1m execution data. Unlike
    _add_base_indicators() (run on a resampled higher-tf copy and
    merged back with a lag), this runs on the same 1m frame the bot
    executes on, so every value is already "live" at the bar it's
    computed on -- no forward-fill needed."""
    df = df.copy()

    df["bullish_halftrend"], df["bearish_halftrend"], _ = HalfTrend(df)

    df["range"] = GetRange(df)
    df["range_ma"] = RangeMA(df)
    df["bullish_ob"] = BullishOB(df)
    df["bearish_ob"] = BearishOB(df)
    df["bullish_ob_mitigation"], df["bearish_ob_mitigation"] = OBMitigation(df)

    df["pdh_dist"] = PDHDistance(df) / PIP_VALUE
    df["pdl_dist"] = PDLDistance(df) / PIP_VALUE
    df["asia_high_dist"] = AsiaHighDistance(df) / PIP_VALUE
    df["asia_low_dist"] = AsiaLowDistance(df) / PIP_VALUE

    return df[M1_LEVEL_FEATURES]


def add_indicators(df):
    """df must be raw 1-minute OHLC candles (Open/High/Low/Close). The
    bot executes at this same 1m granularity (bar-by-bar SL/TP
    tracking, order fills) -- same execution model as bot.py.

    Two kinds of indicators get merged onto every 1m row:
      - M1_LEVEL_FEATURES (_add_m1_levels): computed directly on the
        1m data itself -- HalfTrend, order blocks/mitigation, PDH/PDL
        and Asia high/low distance -- prefixed 1m_.
      - BASE_INDICATOR_FEATURES (_add_base_indicators): computed on
        each of ANALYZED_TIMEFRAMES (5m/15m/1h/4h) resampled from this
        same 1m data (right-labeled/left-closed, so a bin is only
        visible once it has actually closed), then merged back
        forward-filled so a bar only ever sees the most recently
        *closed* higher-tf candle -- prefixed 5m_/15m_/1h_/4h_. Same
        no-lookahead multi-timeframe merge bot.py's own
        add_indicators() uses, just run across four timeframes instead
        of two.
    """

    result = df[["Open", "High", "Low", "Close"]].copy()

    m1_levels = _add_m1_levels(df).add_prefix("1m_")
    result = pd.concat([result, m1_levels], axis=1)

    for prefix, freq in ANALYZED_TIMEFRAMES:

        df_tf = df.resample(
            freq, label="right", closed="left"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }).dropna()

        df_tf = _add_base_indicators(df_tf).add_prefix(f"{prefix}_")

        result = pd.concat(
            [result, df_tf.reindex(result.index, method="ffill")],
            axis=1
        )

    result.dropna(inplace=True)
    return result


def load_last_mb_xauusd(file_path=None, mb=20, delimiter=",", col_names=None):
    """Same loader as bot.py's -- reads only the last `mb` megabytes of
    the CSV rather than the whole (potentially huge) file."""
    if file_path is None:
        file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "download", "XAUUSD.csv"
        )

    print(f"Loading: {file_path}")

    file_size = os.path.getsize(file_path)
    offset = max(file_size - mb * 1024 * 1024, 0)

    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read().decode(errors="ignore")

        if offset > 0:
            data = data.split("\n", 1)[-1]

    df = pd.read_csv(StringIO(data), delimiter=delimiter, header=0)

    df.columns = ["Date", "Timestamp", "Open", "High", "Low", "Close", "Volume"]

    df["Date"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Timestamp"],
        format="%Y%m%d %H:%M:%S",
        utc=True
    )
    df.set_index("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    print(f"Loaded: {file_path}")

    return df.dropna()


# ==========================================================================
# LSTM-PPO AGENT -- identical to bot.py's (state-vector-agnostic, so
# reused verbatim aside from the BUY/SELL/HOLD relabeling below).
# ==========================================================================

class PPOLSTMNetwork(nn.Module):
    def __init__(self, state_size=12, hidden_size=64, action_size=3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=state_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2
        )

        self.policy = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]

        logits = self.policy(h)
        value = self.value(h).squeeze(-1)

        return logits, value


class LSTMPPOAgent:
    def __init__(
        self,
        state_size,
        hidden_size,
        action_size,
        lr=3e-4,
        gamma=0.95,
        clip_ratio=0.2,
        gae_lambda=0.95
    ):
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda

        self.train_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = PPOLSTMNetwork(
            state_size, hidden_size, action_size
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

        self.trajectory = []

    def _state_tensor(self, state_seq):
        return torch.tensor(
            state_seq, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def select_action(self, state_seq, in_position=False, training=False):

        state = self._state_tensor(state_seq)

        with torch.no_grad():
            logits, value = self.model(state)

        logits = logits.squeeze(0)

        # Only HOLD is valid while a position is open -- can't stack a
        # second trade. (bot.py's equivalent masks to [0]=hold; here
        # hold is action index 2, per the BUY/SELL/HOLD spec ordering.)
        valid_actions = [HOLD] if in_position else [BUY, SELL, HOLD]

        masked_logits = logits.clone()

        for i in range(self.action_size):
            if i not in valid_actions:
                masked_logits[i] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample() if training else torch.argmax(probs)

        logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item())
        )

    def store_transition(self, state_seq, action, logprob, value, reward, done):
        self.trajectory.append((
            np.array(state_seq, dtype=np.float32),
            action, logprob, value, reward, done
        ))

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0

        values = np.append(values, 0.0)

        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )
            gae = (
                delta
                + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            )
            advantages.insert(0, gae)

        return np.array(advantages, dtype=np.float32)

    def train(self):

        if len(self.trajectory) < 32:
            return

        states, actions, old_logprobs, values, rewards, dones = zip(*self.trajectory)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        old_logprobs = np.array(old_logprobs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        n = len(states)

        for _ in range(self.train_epochs):
            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                logits, values_pred = self.model(b_states)
                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - b_old_logprobs)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * b_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, b_returns)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

        self.trajectory.clear()

    def savecheckpoint(self, tag):
        os.makedirs(SAVE_DIR, exist_ok=True)

        filename = (
            f"{SAVE_DIR}/"
            f"{datetime.now().strftime('%Y-%m-%d')}-"
            f"{tag}.checkpoint.pt"
        )

        torch.save(
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            filename
        )

    def loadcheckpoint(self, tag):
        if not os.path.exists(SAVE_DIR):
            return

        files = [
            os.path.join(SAVE_DIR, f)
            for f in os.listdir(SAVE_DIR)
            if f.endswith(".checkpoint.pt") and tag in f
        ]

        if not files:
            return

        latest = max(files, key=os.path.getmtime)

        checkpoint = torch.load(latest, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Loaded checkpoint: {latest}")


# ==========================================================================
# GBDT WIN-RATE FILTER
# ==========================================================================

class GBDTWinRateFilter:
    """XGBoost (GBDT) win-rate filter -- every closed trade's entry-time
    feature vector + win/loss outcome is accumulated, and periodically
    refit. predict_win_rate() estimates a new setup's win probability;
    a trade only clears the gate once that estimate is at least the
    caller's min-winrate bar. One filter, one bar shared across all
    three entry strategies -- not tiered per strategy/RR.

    Two differences from a plain always-on filter, per spec:
      - ready()/allows() don't gate anything until `weeks_trained`
        (bumped once per simulated training week in train_bot) reaches
        WEEKS_BEFORE_FILTER -- fitting/accumulating still happens the
        whole time, it just isn't *applied* until then.
      - min_winrate()'s base bar is a plain number (BASE_MIN_WINRATE by
        default, 35%; overridable per run via train_bot()'s/test_bot()'s
        min_winrate= param, i.e. the --min-winrate CLI flag) computed by
        the caller and passed in, not hardcoded here.

    Persisted as raw (X, y) samples (plus weeks_trained), not the fitted
    model itself, so the sample set survives an algorithm change and
    keeps accumulating across runs -- same design as bot.py's
    WinRateModel, capped at max_mb by trimming to the most recent
    samples once the pickled size exceeds it.
    """

    def __init__(self, symbol, min_samples=30, max_mb=20):
        self.symbol = symbol
        self.min_samples = min_samples
        self.max_mb = max_mb
        self.model = None
        self.X = []
        self.y = []
        self.weeks_trained = 0
        self.trimmed_last_save = False

    def add_sample(self, state, won):
        self.X.append(np.asarray(state, dtype=np.float32))
        self.y.append(1 if won else 0)

    def fit(self):
        if len(self.X) < self.min_samples or len(set(self.y)) < 2:
            return False

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )
        self.model.fit(np.array(self.X), np.array(self.y))
        return True

    def predict_win_rate(self, state):
        if self.model is None:
            return None

        proba = self.model.predict_proba(
            np.asarray(state, dtype=np.float32).reshape(1, -1)
        )[0]

        classes = list(self.model.classes_)
        return proba[classes.index(1)] if 1 in classes else 0.0

    def ready(self, min_weeks=WEEKS_BEFORE_FILTER):
        # The filter only "starts" once 5 simulated training weeks have
        # completed -- see train_bot(). Before that it's a no-op even
        # if a model already happens to be fit.
        return self.model is not None and self.weeks_trained >= min_weeks

    def allows(self, state, min_winrate, min_weeks=WEEKS_BEFORE_FILTER):
        if not self.ready(min_weeks):
            return True
        return self.predict_win_rate(state) >= min_winrate

    def _path(self):
        return os.path.join(SAVE_DIR, f"{self.symbol}.gbdt_winrate.pkl")

    def _trim_to_max_size(self):
        if len(self.X) < 2:
            self.trimmed_last_save = False
            return

        max_bytes = self.max_mb * 1024 * 1024
        current_bytes = len(pickle.dumps({"X": self.X, "y": self.y}))

        if current_bytes <= max_bytes:
            self.trimmed_last_save = False
            return

        bytes_per_sample = current_bytes / len(self.X)
        keep = max(int(max_bytes / bytes_per_sample), self.min_samples)

        self.X = self.X[-keep:]
        self.y = self.y[-keep:]
        self.trimmed_last_save = True

    def file_size_mb(self):
        path = self._path()
        if not os.path.exists(path):
            return 0.0
        return os.path.getsize(path) / (1024 * 1024)

    def min_winrate(self, base_min_winrate, elevated_min_winrate=0.9, size_threshold_mb=5):
        if self.trimmed_last_save or self.file_size_mb() > size_threshold_mb:
            return elevated_min_winrate
        return base_min_winrate

    def save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self._trim_to_max_size()

        with open(self._path(), "wb") as f:
            pickle.dump(
                {"X": self.X, "y": self.y, "weeks_trained": self.weeks_trained}, f
            )

    def load(self):
        path = self._path()
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.X = data["X"]
        self.y = data["y"]
        self.weeks_trained = data.get("weeks_trained", 0)

        self.fit()
        return True


def risk_multiplier(predicted_win_rate, min_winrate, step=RISK_STEP, max_multiplier=MAX_RISK_MULTIPLIER):
    """"Add 1R to risk per +10% predicted win rate": every full `step`
    (default 10 points) the filter's predicted win rate clears above
    the min-winrate bar adds one more unit of the base --risk to the
    position -- e.g. a setup predicted at 65% against a 35% bar (a
    30-point margin) risks 1 + 3 = 4x base risk. Returns 1.0 (base risk,
    no bonus) whenever the filter isn't active yet (predicted_win_rate
    is None) or the setup only just clears the bar."""
    if predicted_win_rate is None:
        return 1.0

    margin = predicted_win_rate - min_winrate
    if margin <= 0:
        return 1.0

    bonus_r = math.floor(margin / step)
    return min(1.0 + bonus_r, max_multiplier)


# ==========================================================================
# STATS HELPERS -- identical to bot.py's.
# ==========================================================================

def sharpe_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return (mean_ret - risk_free_rate) / std_ret


def sortino_ratio(returns, risk_free_rate=0.0):
    mean_ret = np.mean(returns)
    downside_diff = [(r - risk_free_rate) ** 2 for r in returns if r < risk_free_rate]

    if len(downside_diff) == 0:
        return 0

    downside_std = np.sqrt(np.mean(downside_diff))

    if downside_std == 0:
        return 0

    return (mean_ret - risk_free_rate) / downside_std


def max_drawdown(returns):
    if len(returns) == 0:
        return 0

    equity = np.cumsum(returns)
    peak = equity[0]
    max_dd = 0

    for value in equity:
        peak = max(peak, value)
        dd = peak - value
        max_dd = max(max_dd, dd)

    return max_dd


def streak_stats(returns):
    win_streaks = []
    loss_streaks = []

    current_len = 0
    current_sign = 0

    for r in returns:
        sign = 1 if r > 0 else (-1 if r < 0 else 0)

        if sign != 0 and sign == current_sign:
            current_len += 1
        else:
            if current_sign == 1 and current_len > 0:
                win_streaks.append(current_len)
            elif current_sign == -1 and current_len > 0:
                loss_streaks.append(current_len)
            current_len = 1 if sign != 0 else 0
            current_sign = sign

    if current_sign == 1 and current_len > 0:
        win_streaks.append(current_len)
    elif current_sign == -1 and current_len > 0:
        loss_streaks.append(current_len)

    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

    return avg_win_streak, avg_loss_streak


# ==========================================================================
# MT5 I/O
# ==========================================================================

def open_long(symbol, lot_size, sl_pips, tp_pips, magic=MAGIC):
    tick = mt5.symbol_info_tick(symbol)
    entry = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": entry,
        "sl": entry - sl_pips / 10,
        "tp": entry + tp_pips / 10,
        "deviation": 20,
        "magic": magic,
        "comment": "stoch-halftrend",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}")
        return None


def open_short(symbol, lot_size, sl_pips, tp_pips, magic=MAGIC):
    tick = mt5.symbol_info_tick(symbol)
    entry = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry,
        "sl": entry + sl_pips / 10,
        "tp": entry - tp_pips / 10,
        "deviation": 20,
        "magic": magic,
        "comment": "stoch-halftrend",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode}")
        return None


def open_positions(symbol, magic=MAGIC):
    positions = mt5.positions_get(symbol=symbol)
    positions = [p for p in positions if p.magic == magic]
    return len(positions)


def get_today_profit(symbol, day, now=None, magic=MAGIC):
    day_start = datetime.combine(day, datetime.min.time())

    if now is None:
        now = datetime.now()

    deals = mt5.history_deals_get(day_start, now + timedelta(days=1))

    if deals is None:
        return 0.0

    return sum(
        d.profit + d.swap + d.commission
        for d in deals
        if d.magic == magic
        and d.symbol == symbol
        and d.entry == mt5.DEAL_ENTRY_OUT
    )


def close_trades(magic=MAGIC):
    positions = mt5.positions_get()

    if positions is None:
        print("Failed to get positions:", mt5.last_error())
        return

    for pos in positions:
        if pos.magic != magic:
            continue

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": magic,
            "comment": "End of Day",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Failed to close {pos.ticket}: {result.retcode}")


# ==========================================================================
# TRAINING
# ==========================================================================

def _htf_ok_columns(df, htf_mode):
    if htf_mode == "any":
        long_ok = (df["1h_bullish_halftrend"] == 1) | (df["4h_bullish_halftrend"] == 1)
        short_ok = (df["1h_bearish_halftrend"] == 1) | (df["4h_bearish_halftrend"] == 1)
    else:  # "both"
        long_ok = (df["1h_bullish_halftrend"] == 1) & (df["4h_bullish_halftrend"] == 1)
        short_ok = (df["1h_bearish_halftrend"] == 1) & (df["4h_bearish_halftrend"] == 1)
    return long_ok, short_ok


def _signal_columns(df):
    """bull_signal/bear_signal: True on any bar where ANY analyzed
    timeframe (5m/15m/1h/4h) has a live zone-breakout AND that same
    timeframe's own ADX clears ADX_MIN -- i.e. the stoch/%R zone-
    breakout strategy is "executed on" all four timeframes at once,
    any one of them firing is enough to produce a candidate."""
    bull_signal = pd.Series(False, index=df.index)
    bear_signal = pd.Series(False, index=df.index)

    for prefix, _ in ANALYZED_TIMEFRAMES:
        tf_adx_ok = df[f"{prefix}_adx"] >= ADX_MIN
        bull_signal |= df[f"{prefix}_bull_breakout"].astype(bool) & tf_adx_ok
        bear_signal |= df[f"{prefix}_bear_breakout"].astype(bool) & tf_adx_ok

    return bull_signal, bear_signal


def _level_signal_columns(df):
    """Candidate signals for the two 1m-HalfTrend-gated level
    strategies (both need M1_LEVEL_FEATURES from _add_m1_levels()):

    - poi_bull/poi_bear ("PDH/PDL + Asia high/low break, retest and
      reversal"): the 1m HalfTrend's current direction, on any bar
      where price is within POI_REACH_PIPS of any of the four levels
      (PDH, PDL, Asia high, Asia low). HalfTrend's live state stands in
      for whichever of breakout/retest/reversal is actually happening
      at the level -- continuing through it trades as a breakout,
      flipping at it trades as a reversal -- rather than classifying
      the three separately.
    - ob_bull/ob_bear ("OB mitigation"): the 1m HalfTrend *flipping*
      direction this bar (not just agreeing, per spec's "halftrend
      reversal") on a bar that's also mitigating (within
      POI_REACH_PIPS of, via OBMitigation()'s own threshold) a
      same-direction order block -- a bullish/demand OB mitigated with
      HalfTrend flipping up is a reversal long, a bearish/supply OB
      mitigated with HalfTrend flipping down is a reversal short.
    """
    bull_ht = df["1m_bullish_halftrend"].astype(bool)
    bear_ht = df["1m_bearish_halftrend"].astype(bool)

    near_poi = (
        (df["1m_pdh_dist"].abs() <= POI_REACH_PIPS)
        | (df["1m_pdl_dist"].abs() <= POI_REACH_PIPS)
        | (df["1m_asia_high_dist"].abs() <= POI_REACH_PIPS)
        | (df["1m_asia_low_dist"].abs() <= POI_REACH_PIPS)
    )

    poi_bull = bull_ht & near_poi
    poi_bear = bear_ht & near_poi

    flip_bull = bull_ht & ~bull_ht.shift(1).fillna(False)
    flip_bear = bear_ht & ~bear_ht.shift(1).fillna(False)

    ob_bull = flip_bull & df["1m_bullish_ob_mitigation"].astype(bool)
    ob_bear = flip_bear & df["1m_bearish_ob_mitigation"].astype(bool)

    return poi_bull, poi_bear, ob_bull, ob_bear


def _select_candidate(
    bull_stoch, bear_stoch, long_htf_ok, short_htf_ok,
    poi_bull, poi_bear, ob_bull, ob_bear,
):
    """Picks a candidate (action, tp_pips, strategy_name) for the
    current bar across all three entry strategies, in a fixed priority
    order: stoch/%R zone-breakout first, then the PDH/PDL/Asia-level
    reversal, then OB mitigation. More than one firing on the same bar
    (in the same or opposite directions) is rare given how differently
    each triggers, and resolved by this order rather than reconciled --
    the agent still decides whether to actually take whatever candidate
    wins. Returns (HOLD, TP_PIPS, None) if nothing fires."""
    if bull_stoch and long_htf_ok:
        return BUY, TP_PIPS, "stoch_breakout"
    if bear_stoch and short_htf_ok:
        return SELL, TP_PIPS, "stoch_breakout"
    if poi_bull:
        return BUY, LEVEL_TP_PIPS, "poi_reversal"
    if poi_bear:
        return SELL, LEVEL_TP_PIPS, "poi_reversal"
    if ob_bull:
        return BUY, LEVEL_TP_PIPS, "ob_mitigation"
    if ob_bear:
        return SELL, LEVEL_TP_PIPS, "ob_mitigation"
    return HOLD, TP_PIPS, None


def train_bot(symbol="XAUUSD", risk=0.01, htf_mode="any", min_winrate=BASE_MIN_WINRATE):

    print("Training bot (stoch/%R breakout + PDH/PDL/Asia reversal + OB mitigation)")
    start = time.perf_counter()

    TRAIN_HISTORY_MB = 20
    df_m1 = load_last_mb_xauusd(mb=TRAIN_HISTORY_MB)
    print(f"Computing 1m/5m/15m/1h/4h indicators over 1m execution data... ({time.strftime('%H:%M')})")
    df = add_indicators(df_m1)
    elapsed = int((time.perf_counter() - start) // 60)
    print(f"Loaded indicators on {len(df)} 1m bars (Elapsed: {elapsed}m)")

    long_htf_ok, short_htf_ok = _htf_ok_columns(df, htf_mode)
    df["long_htf_ok"] = long_htf_ok
    df["short_htf_ok"] = short_htf_ok

    bull_signal, bear_signal = _signal_columns(df)
    df["bull_signal"] = bull_signal
    df["bear_signal"] = bear_signal

    poi_bull, poi_bear, ob_bull, ob_bear = _level_signal_columns(df)
    df["poi_bull"] = poi_bull
    df["poi_bear"] = poi_bear
    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear

    tag = model_tag(symbol)

    agent = LSTMPPOAgent(state_size=len(FEATURES), hidden_size=64, action_size=3)
    gbdt = GBDTWinRateFilter(tag)

    try:
        agent.loadcheckpoint(tag)
        gbdt.load()
        print(f"[{symbol}] Loaded checkpoint")
    except Exception as e:
        print(f"[{symbol}] Starting fresh ({e})")

    MIN_WINRATE = gbdt.min_winrate(min_winrate)

    # Precompute everything the loop needs as plain arrays once, up
    # front, same spirit as bot.py's weekly-slice caching but simpler
    # since the full feature matrix here (1m bars x 78 features) is
    # small enough to hold in memory for the whole run at once.
    feature_matrix = df[FEATURES].to_numpy(dtype=np.float32)
    close_arr = df["Close"].to_numpy()
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    bull_signal_arr = df["bull_signal"].to_numpy()
    bear_signal_arr = df["bear_signal"].to_numpy()
    long_htf_ok_arr = df["long_htf_ok"].to_numpy()
    short_htf_ok_arr = df["short_htf_ok"].to_numpy()
    poi_bull_arr = df["poi_bull"].to_numpy()
    poi_bear_arr = df["poi_bear"].to_numpy()
    ob_bull_arr = df["ob_bull"].to_numpy()
    ob_bear_arr = df["ob_bear"].to_numpy()

    save_counter = 0
    in_position = False
    position_type = None
    entry_price = sl_price = tp_price = 0.0
    entry_state = None
    entry_strategy = None
    mult = 1.0

    trade_returns = []
    strategy_counts = {"stoch_breakout": 0, "poi_reversal": 0, "ob_mitigation": 0}
    state_buffer = deque(maxlen=SEQ_LEN)

    loop_count = 0
    training_start_2 = time.time()
    training_start_3 = time.time()

    try:
        while True:
            loop_count += 1
            print(f"[{symbol}] [INFO] Starting pass {loop_count} over dataset")

            for i in range(SEQ_LEN, len(df)):

                current_price = close_arr[i]
                high = high_arr[i]
                low = low_arr[i]

                state = feature_matrix[i]
                state_buffer.append(state)

                if len(state_buffer) < SEQ_LEN:
                    continue

                state_seq = np.array(state_buffer)

                result = agent.select_action(state_seq, in_position, training=True)
                if result is None:
                    continue

                action, logprob, value = result

                # Candidate direction (and which of the three
                # strategies it comes from, and that strategy's own
                # TP) comes from the deterministic technical stack, not
                # the agent's own guess -- the agent's job is only
                # deciding whether to *take* a signal that already
                # exists (mirrors bot.py's HalfTrend-redirect:
                # technicals pick direction, the agent times entries).
                candidate, candidate_tp_pips, candidate_strategy = _select_candidate(
                    bull_signal_arr[i], bear_signal_arr[i],
                    long_htf_ok_arr[i], short_htf_ok_arr[i],
                    poi_bull_arr[i], poi_bear_arr[i],
                    ob_bull_arr[i], ob_bear_arr[i],
                )

                if action in (BUY, SELL):
                    action = candidate if candidate != HOLD else HOLD

                # GBDT win-rate filter -- only gates once
                # WEEKS_BEFORE_FILTER simulated weeks have completed
                # (gbdt.ready()); before that it's a no-op so early
                # training isn't blocked waiting on data that doesn't
                # exist yet, but it keeps fitting/accumulating the
                # whole time.
                predicted_wr = None
                if action in (BUY, SELL) and gbdt.ready():
                    predicted_wr = gbdt.predict_win_rate(state)
                    if predicted_wr < MIN_WINRATE:
                        action = HOLD

                pnl = 0.0
                done = False

                if action == BUY and not in_position:
                    in_position = True
                    position_type = "long"
                    entry_price = current_price
                    entry_state = state.copy()
                    entry_strategy = candidate_strategy
                    mult = risk_multiplier(predicted_wr, MIN_WINRATE)

                    sl_price = entry_price - SL_PIPS * PIP_VALUE
                    tp_price = entry_price + candidate_tp_pips * PIP_VALUE

                    strategy_counts[entry_strategy] += 1

                    agent.store_transition(state_seq, action, logprob, value, pnl, done)

                elif action == SELL and not in_position:
                    in_position = True
                    position_type = "short"
                    entry_price = current_price
                    entry_state = state.copy()
                    entry_strategy = candidate_strategy
                    mult = risk_multiplier(predicted_wr, MIN_WINRATE)

                    sl_price = entry_price + SL_PIPS * PIP_VALUE
                    tp_price = entry_price - candidate_tp_pips * PIP_VALUE

                    strategy_counts[entry_strategy] += 1

                    agent.store_transition(state_seq, action, logprob, value, pnl, done)

                if in_position:
                    trade_closed = False

                    if position_type == "long":
                        if high >= tp_price:
                            pnl = ((tp_price - entry_price) / PIP_VALUE - COMMISSION) * mult
                            trade_closed = True
                        if low <= sl_price:
                            pnl = ((sl_price - entry_price) / PIP_VALUE - COMMISSION) * mult
                            trade_closed = True

                    elif position_type == "short":
                        if low <= tp_price:
                            pnl = ((entry_price - tp_price) / PIP_VALUE - COMMISSION) * mult
                            trade_closed = True
                        if high >= sl_price:
                            pnl = ((entry_price - sl_price) / PIP_VALUE - COMMISSION) * mult
                            trade_closed = True

                    if trade_closed:
                        in_position = False
                        done = True
                        trade_returns.append(pnl)
                        gbdt.add_sample(entry_state, pnl > 0)

                        agent.store_transition(state_seq, action, logprob, value, pnl, done)

                save_counter += 1

                # ==========================================================
                # WEEKLY TRAINING
                # ==========================================================
                if save_counter % TRADING_WEEK_BARS == 0:

                    # Printed every week regardless of how many trades
                    # it had -- including zero -- so a quiet week is
                    # still visible rather than silently skipped. Every
                    # stat below is guarded against an empty
                    # trade_returns (np.mean/np.std/winrate would
                    # otherwise warn or divide by zero); streak_stats()
                    # and max_drawdown() already handle empty input on
                    # their own.
                    wins = [r for r in trade_returns if r > 0]
                    losses = [r for r in trade_returns if r < 0]

                    weekly_pnl = np.sum(trade_returns)
                    winrate = len(wins) / len(trade_returns) if trade_returns else 0.0
                    mean_win = np.mean(wins) if wins else 0.0
                    mean_loss = np.mean(losses) if losses else 0.0

                    sharpe = sharpe_ratio(trade_returns) if trade_returns else 0.0
                    sortino = sortino_ratio(trade_returns) if trade_returns else 0.0

                    std_ret = np.std(trade_returns) if trade_returns else 0.0
                    zscore = np.mean(trade_returns) / std_ret if std_ret > 0 else 0.0

                    avg_win_streak, avg_loss_streak = streak_stats(trade_returns)

                    gross_profit = sum(wins)
                    gross_loss = abs(sum(losses))
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                    max_dd = max_drawdown(trade_returns)
                    R_pnl = weekly_pnl / SL_PIPS
                    rf = R_pnl / (max_dd / SL_PIPS) if max_dd > 0 else 0.0

                    print()
                    print("================================================")
                    print(f"[{symbol}] WEEKLY PPO TRAINING")
                    print("================================================")
                    print(f"Trades:          {len(trade_returns)}")
                    print(f"Weekly PnL:      {weekly_pnl:.0f} pips")
                    print(f"Weekly R PnL:    {R_pnl:.2f}R")
                    print(f"Max DD:          {max_dd/SL_PIPS:.2f}R")
                    print(f"Winrate:         {winrate*100:.2f}%")
                    print(f"Mean Win:        {mean_win:.0f} pips")
                    print(f"Mean Loss:       {mean_loss:.0f} pips")
                    print(f"Avg Win Streak:  {avg_win_streak:.2f}")
                    print(f"Avg Loss Streak: {avg_loss_streak:.2f}")
                    print(f"Z-score:         {zscore:.2f}")
                    print(f"PF:              {profit_factor:.2f}")
                    print(f"RF:              {rf:.2f}")
                    print(f"Sharpe:          {sharpe:.2f}")
                    print(f"Sortino:         {sortino:.2f}")
                    print(f"MIN_WINRATE:     {MIN_WINRATE*100:.1f}%")
                    filter_state = (
                        "ACTIVE" if gbdt.ready()
                        else f"bootstrapping ({gbdt.weeks_trained}/{WEEKS_BEFORE_FILTER} weeks)"
                    )
                    print(f"GBDT filter:     {filter_state}")
                    print(
                        f"By strategy:     "
                        f"stoch={strategy_counts['stoch_breakout']} "
                        f"poi={strategy_counts['poi_reversal']} "
                        f"ob={strategy_counts['ob_mitigation']}"
                    )
                    print("================================================")
                    print()

                    print(
                        f"[{symbol}] [INFO] Trained on data "
                        f"(Elapsed: {timedelta(seconds=int(time.time() - training_start_3))})"
                    )

                    trade_returns = []
                    strategy_counts = {"stoch_breakout": 0, "poi_reversal": 0, "ob_mitigation": 0}

                    training_start = time.time()
                    print(f"[{symbol}] [INFO] Training PPO...")
                    agent.train()
                    agent.trajectory.clear()

                    print(
                        f"[{symbol}] [INFO] Finished training PPO "
                        f"(Elapsed: {timedelta(seconds=int(time.time() - training_start))})"
                    )

                    agent.savecheckpoint(tag)

                    gbdt.weeks_trained += 1
                    if gbdt.fit():
                        gbdt.save()
                        MIN_WINRATE = gbdt.min_winrate(min_winrate)
                        print(
                            f"[{symbol}] [INFO] GBDT filter refit on "
                            f"{len(gbdt.X)} trades and saved"
                        )

                    completed = int(save_counter / TRADING_WEEK_BARS)
                    total = max(int(round(len(df) / TRADING_WEEK_BARS, 0)), 1)

                    elapsed_run = time.time() - training_start_2
                    avg_time = elapsed_run / max(completed, 1)
                    remaining = max(total - completed, 0)
                    eta = remaining * avg_time

                    print(
                        f"[{symbol}] [INFO] "
                        f"{completed}/{total} ({completed/total*100:.1f}%) | "
                        f"Elapsed: {timedelta(seconds=int(elapsed_run))} | "
                        f"ETA: {timedelta(seconds=int(eta))}"
                    )

                    training_start_3 = time.time()

    except KeyboardInterrupt:
        print(f"[{symbol}] [INFO] KeyboardInterrupt received, stopping after {loop_count} pass(es) over the dataset")

    agent.train()
    agent.savecheckpoint(tag)

    if gbdt.fit():
        gbdt.save()

    print(f"[{symbol}] [INFO] Finished training")

    return agent


# ==========================================================================
# LIVE TRADING
# ==========================================================================

def _rename_mt5_rates(d):
    d = d.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "time": "Date"
    })
    d["Date"] = pd.to_datetime(d["Date"], unit="s", utc=True)
    d.set_index("Date", inplace=True)
    return d[["Open", "High", "Low", "Close"]]


def test_bot(symbol="XAUUSD", risk=0.01, htf_mode="any", min_winrate=BASE_MIN_WINRATE):

    if mt5 is None:
        raise RuntimeError(
            "test_bot() needs the MetaTrader5 package and a running MT5 "
            "terminal (Windows-only) -- see README."
        )

    mt5.initialize()

    tag = model_tag(symbol)

    agent = LSTMPPOAgent(state_size=len(FEATURES), hidden_size=64, action_size=3)
    agent.loadcheckpoint(tag)

    gbdt = GBDTWinRateFilter(tag)
    gbdt.load()
    MIN_WINRATE = gbdt.min_winrate(min_winrate)

    # ==========================================================
    # INITIAL LOAD
    # ==========================================================
    # 1-minute bars -- generous margin over the largest lookback (the
    # 4h branch's EMA21/ADX/STOCH need ~21 four-hour candles = ~5040
    # 1m bars) plus HalfTrend/zone-breakout warmup on every analyzed
    # timeframe, so add_indicators()'s dropna() never wipes out the
    # tail we actually need. Same role bot.py's LIVE_HISTORY_BARS
    # plays for its own (smaller, 15m-max) multi-timeframe merge.
    LIVE_HISTORY_BARS = 12000

    rates_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, LIVE_HISTORY_BARS)
    raw_df = _rename_mt5_rates(pd.DataFrame(rates_m1))

    df = add_indicators(raw_df.copy())

    last_m1_epoch = int(raw_df.index[-1].timestamp())
    last_trading_date = df.index[-1].date()
    day_start_balance = mt5.account_info().balance

    # ==========================================================
    # MAIN LOOP
    # ==========================================================

    while True:

        now = datetime.now()
        seconds_until_next_minute = 60 - now.second - now.microsecond / 1_000_000
        if seconds_until_next_minute <= 0:
            seconds_until_next_minute += 0.25
        time.sleep(seconds_until_next_minute)

        # ======================================================
        # CHECK FOR NEW M1 CANDLE
        # ======================================================
        new_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        current_m1_epoch = int(new_m1[0]["time"])

        while current_m1_epoch == last_m1_epoch:
            new_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            current_m1_epoch = int(new_m1[0]["time"])

        last_m1_epoch = current_m1_epoch

        # ======================================================
        # APPEND NEW CANDLE, RECOMPUTE INDICATORS
        # ======================================================
        new_row = _rename_mt5_rates(pd.DataFrame(new_m1))

        if new_row.index[-1] != raw_df.index[-1]:
            raw_df = pd.concat([raw_df, new_row])
            raw_df = raw_df.tail(LIVE_HISTORY_BARS)
            df = add_indicators(raw_df.copy())

        state_seq = df[FEATURES].tail(SEQ_LEN).to_numpy(dtype=np.float32)
        if state_seq.shape[0] != SEQ_LEN:
            print(f"Bad state shape: {state_seq.shape}")
            continue

        # ======================================================
        # PPO DECISION
        # ======================================================
        open_pos = open_positions(symbol)

        result = agent.select_action(state_seq, open_pos > 0, training=False)
        if result is None:
            continue

        action, logprob, value = result

        current = df.iloc[-1]
        prev = df.iloc[-2]

        if htf_mode == "any":
            long_ok = bool(current["1h_bullish_halftrend"]) or bool(current["4h_bullish_halftrend"])
            short_ok = bool(current["1h_bearish_halftrend"]) or bool(current["4h_bearish_halftrend"])
        else:
            long_ok = bool(current["1h_bullish_halftrend"]) and bool(current["4h_bullish_halftrend"])
            short_ok = bool(current["1h_bearish_halftrend"]) and bool(current["4h_bearish_halftrend"])

        # Stoch/%R breakout candidate: ANY analyzed timeframe
        # (5m/15m/1h/4h) signalling a zone-breakout with its own ADX
        # clearing ADX_MIN is enough -- same _signal_columns() logic
        # as train_bot(), evaluated on just the latest bar here.
        bull_signal = any(
            bool(current[f"{prefix}_bull_breakout"]) and current[f"{prefix}_adx"] >= ADX_MIN
            for prefix, _ in ANALYZED_TIMEFRAMES
        )
        bear_signal = any(
            bool(current[f"{prefix}_bear_breakout"]) and current[f"{prefix}_adx"] >= ADX_MIN
            for prefix, _ in ANALYZED_TIMEFRAMES
        )

        # PDH/PDL + Asia high/low reversal candidate: 1m HalfTrend's
        # current direction within POI_REACH_PIPS of any of the four
        # levels -- same logic as _level_signal_columns()'s poi_bull/
        # poi_bear, evaluated on just the latest bar here.
        bull_ht = bool(current["1m_bullish_halftrend"])
        bear_ht = bool(current["1m_bearish_halftrend"])

        near_poi = (
            abs(current["1m_pdh_dist"]) <= POI_REACH_PIPS
            or abs(current["1m_pdl_dist"]) <= POI_REACH_PIPS
            or abs(current["1m_asia_high_dist"]) <= POI_REACH_PIPS
            or abs(current["1m_asia_low_dist"]) <= POI_REACH_PIPS
        )
        poi_bull = bull_ht and near_poi
        poi_bear = bear_ht and near_poi

        # OB-mitigation candidate: 1m HalfTrend flipping direction this
        # bar (not just agreeing) while also mitigating a
        # same-direction order block -- same logic as
        # _level_signal_columns()'s ob_bull/ob_bear.
        flip_bull = bull_ht and not bool(prev["1m_bullish_halftrend"])
        flip_bear = bear_ht and not bool(prev["1m_bearish_halftrend"])
        ob_bull = flip_bull and bool(current["1m_bullish_ob_mitigation"])
        ob_bear = flip_bear and bool(current["1m_bearish_ob_mitigation"])

        candidate, candidate_tp_pips, candidate_strategy = _select_candidate(
            bull_signal, bear_signal, long_ok, short_ok,
            poi_bull, poi_bear, ob_bull, ob_bear,
        )

        if action in (BUY, SELL):
            action = candidate if candidate != HOLD else HOLD

        state = state_seq[-1]
        predicted_wr = None
        if action in (BUY, SELL) and gbdt.ready():
            predicted_wr = gbdt.predict_win_rate(state)
            if predicted_wr < MIN_WINRATE:
                action = HOLD

        current_time = df.index[-1]

        if current_time.date() != last_trading_date:
            last_trading_date = current_time.date()
            day_start_balance = mt5.account_info().balance

        # Flatten and force HOLD heading into the daily close, same
        # convention as bot.py.
        if current_time.hour == 23 and current_time.minute >= 55:
            if open_pos != 0:
                close_trades()
            action = HOLD

        # ======================================================
        # OPEN NEW TRADE
        # ======================================================
        if open_pos == 0 and action in (BUY, SELL):

            account = mt5.account_info()
            balance = account.balance

            mult = risk_multiplier(predicted_wr, MIN_WINRATE)
            effective_risk = risk * mult

            # Lot size: hitting the initial SL costs effective_risk% of
            # balance.
            lot = min(max((balance * effective_risk) / (SL_PIPS * 10), 0.01), 100.0)
            lot = round(lot, 2)

            print(f"[{symbol}] Opening {ACTIONS[action]} via {candidate_strategy} "
                  f"(TP={candidate_tp_pips:.0f} pips, lot={lot})")

            if action == BUY:
                open_long(symbol, lot, SL_PIPS, candidate_tp_pips)
            else:
                open_short(symbol, lot, SL_PIPS, candidate_tp_pips)


# ==========================================================================
# ENTRY POINT
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LSTM-PPO XAUUSD bot: stochastic/%R zone-breakout entries, "
            "gated by 1h/4h HalfTrend direction and a GBDT win-rate filter."
        )
    )

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument(
        "--risk", type=float, default=0.01,
        help=(
            "Base fraction of account balance risked per trade, before "
            "the GBDT filter's win-rate risk multiplier is applied "
            "(default 0.01 = 1%%)."
        )
    )
    parser.add_argument(
        "--htf-mode", choices=["any", "both"], default="any",
        help=(
            "'any' (default): a breakout only needs 1h OR 4h HalfTrend "
            "to agree with its direction. 'both': needs both."
        )
    )
    parser.add_argument(
        "--min-winrate", type=float, default=BASE_MIN_WINRATE, dest="min_winrate",
        help=(
            "GBDT win-rate filter threshold (default "
            f"{BASE_MIN_WINRATE:.2f} = {BASE_MIN_WINRATE*100:.0f}%%). Applies "
            "as one shared function across all three entry strategies -- the "
            "1:4 RR stoch/%R breakout and the two 1:2 RR PDH/PDL/Asia + OB "
            "level strategies alike, not tiered per RR -- and to both "
            "--train and --test (GBDTWinRateFilter.min_winrate() may still "
            "raise it further once the filter's sample buffer saturates; "
            "see BASE_MIN_WINRATE / elevated_min_winrate)."
        )
    )

    args = parser.parse_args()

    # Separate OS processes, not threads -- see bot.py's main() for why
    # (both train_bot/test_bot are long-running CPU loops that would
    # starve each other under one GIL as threads).
    procs = []

    if args.train:
        p = multiprocessing.Process(
            target=train_bot,
            kwargs=dict(
                symbol=args.symbol, risk=args.risk, htf_mode=args.htf_mode,
                min_winrate=args.min_winrate,
            ),
            daemon=True
        )
        p.start()
        procs.append(p)

    if args.test:
        p = multiprocessing.Process(
            target=test_bot,
            kwargs=dict(
                symbol=args.symbol, risk=args.risk, htf_mode=args.htf_mode,
                min_winrate=args.min_winrate,
            ),
            daemon=True
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
