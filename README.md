# bot.py — LSTM-PPO XAUUSD Trading Bot

An ICT/SMC-style feature-engineering pipeline feeding an LSTM-PPO reinforcement-learning
agent that trades XAUUSD (gold) via MetaTrader5. One file, two modes: train against
historical 1-minute data, or trade live.

A second, independent strategy lives in `bot_v2.py` — see
["bot_v2.py — stoch/%R zone-breakout + HalfTrend variant"](#bot_v2py--stochr-zone-breakout--halftrend-variant)
below.

## Requirements

- Windows (the `MetaTrader5` Python package wraps the MT5 terminal API and only works there)
- Python 3.10+
- A running MetaTrader5 terminal, logged into a broker account, with algotrading enabled
- `pip install -r requirements.txt`

## Data

- `download/XAU_1m_data.csv` — historical 1-minute XAUUSD OHLCV, semicolon-delimited
  (`Date;Open;High;Low;Close;Volume`). Required for `--train`.
- `download/dxy_history.csv`, `download/real_yield_history.csv`,
  `download/gld_holdings_history.csv` — cached daily macro series (DXY, real yields, GLD
  ETF holdings). Auto-fetched and cached by `add_macro_features()` on first use (needs
  network access), refreshed after ~20h.
- `LSTM-PPO-saves/` — model checkpoints, named `{date}-{symbol}.checkpoint.pt`.
- `LSTM-PPO-saves/{symbol}.winrate.pkl` — accumulated (feature vector, win/loss)
  samples backing the `WinRateModel` win-rate filter, capped at ~20MB (see
  "Win-rate filter" below).

## Usage

```bash
python bot.py --train                              # train against historical data
python bot.py --test --symbol XAUUSD-STDc           # trade live via MT5
python bot.py --train --test --symbol XAUUSD-STDc   # both at once
```

Both flags spawn daemon threads (`train_bot` / `test_bot`); the process blocks on
`thread.join()` until interrupted (Ctrl+C).

## How it works

### Feature engineering — `add_indicators(df)`

Every 1-minute candle gets the full indicator stack below, then the same stack is
recomputed on 15-minute candles aggregated from that same data and merged back in
(see "Multi-timeframe merge"), so the final feature vector for one 1-minute bar
actually contains both timeframes' worth of state.

**Trend / momentum**
- EMA 7 / 21 / 50 / 200, each with its slope, plus the 7-21 and 50-200 diffs (`EMA`)
- ADX, +DI, -DI (`ADX`)
- Stochastic %K / smoothed %K (`STOCH`)

**Price location / volatility**
- Session VWAP with ±1 stddev bands, above/below flags, and slope (`VWAP`)
- Candle range and its rolling average (`GetRange`, `RangeMA`)
- Indecision (small-body doji-like) candles (`Indecision`)
- Bullish/bearish candle direction (`BullishBearish`)
- Exhaustion candles — body much smaller than the previous one (`ExhaustionCandle`)

**Smart Money Concepts (SMC/ICT) structure**
- Order blocks, bullish & bearish, plus mitigation (`BullishOB`, `BearishOB`, `OBMitigation`)
- Fair value gaps, bullish & bearish, plus retracement into the gap and inverse FVGs
  (`BullishFVG`, `BearishFVG`, `FVGRetracement`, `BullishIFVG`, `BearishIFVG`)
- Market structure shift (MSS) — a close breaking the last swing high/low — plus a
  retest of the broken level and a 0.786 Fibonacci retracement measured from the MSS
  breakout candle back to the swing that anchored it (`MSS`, `MSS_Retest`,
  `MSSFibRetracement` — the retracement variant is specific to this bot, not a
  standard ICT indicator)
- Break of structure, the resulting trend state, and change of character
  (`BoS`, `Trend`, `CHoCH`)
- Breaker blocks — an order block that's been fully mitigated and flipped
  (`BullishBB`, `BearishBB`)
- Rejection blocks (long-wick candles) plus mitigation of the swing they rejected
  (`RejectionBlocks`, `ReversalBlockMitigation`)
- Mitigation blocks — used in the FVG+MB/RB confluence term of the score
  (`BullishMB`, `BearishMB`)
- Equal highs / equal lows (liquidity pools) plus a retest of that level
  (`EQH_EQL`, `EQHEQLRetest`)
- Internal vs. external range liquidity — whether price is inside or has swept
  beyond the current dealing range (`add_irl_erl`)

**Fibonacci**
- Retracement zones (23.6% / 38.2% / 50% / 61.8% / 78.6%) of the last swing-low →
  swing-high (bullish) or swing-high → swing-low (bearish) impulse (`Fibonacci`)

**Trendlines**
- Dynamic ascending support and descending resistance lines drawn through the two
  most recent conforming swing lows / swing highs (searching back past any
  non-conforming swing in between), with "is price above support / below
  resistance" flags (`TrendLines`)

**Session & liquidity distances**
- Killzone: Asia / London / New York session flag (`GetKillzone`)
- Distance from the current Asia session high/low (`AsiaHighDistance`, `AsiaLowDistance`)
- Distance from the previous day's high/low and volume point of control
  (`PDHDistance`, `PDLDistance`, `PDPOCDistance`)

**Composite scores**
- `buy_score` / `sell_score` — a weighted sum of the confluences above (trend
  alignment, OB mitigation, breaker blocks, MSS/CHoCH, FVG+MB/RB confluence,
  liquidity distances, equal-high/low retests, Fibonacci-in-trend, trendline
  position), gated to only fire inside a killzone and above `adx=20` (`BuyScore`,
  `SellScore`)

**Multi-timeframe merge**
- The raw 1-minute OHLCV is resampled into 15-minute candles — right-labeled so a bin
  covering `[10:00, 10:15)` is only "visible" starting at 10:15, i.e. once it has
  actually closed — and the entire stack above is rerun on those 15-minute candles.
  Every resulting column is forward-filled back onto each 1-minute row with a `15m_`
  prefix (e.g. `15m_EMA7`, `15m_bullish_mss`, `15m_buy_score`), so no 1-minute bar
  ever sees a still-forming 15-minute candle's data.

**Macro (daily, added separately by `add_macro_features`, not part of `add_indicators`)**
- DXY, real yield, and GLD-holdings-flow z-scores (`dxy_zscore`, `real_yield_zscore`,
  `gld_flow_zscore`)

### Agent — `LSTMPPOAgent` / `PPOLSTMNetwork`

An LSTM encoder over a `SEQ_LEN`-bar window of features feeds a PPO actor-critic head
with 3 actions (`hold`, `long`, `short`). The reward for a closed trade is its raw
pnl (in pips).

### Win-rate filter — `WinRateModel`

A second, independent gate sits between the PPO agent's action and the market. Before
a `long`/`short` action is allowed to actually open a position, an `XGBClassifier`
predicts the probability that a setup like this one — the same single-bar feature
vector the agent just saw — will win, based on every closed trade's entry-time
features and outcome seen so far. If the predicted win rate doesn't clear
`MIN_WINRATE`, the action is silently downgraded to `hold` instead.

- **Training (`train_bot`)**: every closed trade's entry-time state and win/loss
  outcome is recorded (`add_sample`). At the end of each simulated week, the model is
  refit on every sample accumulated so far and persisted (`fit`, `save`) alongside
  that week's checkpoint.
- **Live trading (`test_bot`)**: loads the persisted model at startup and applies it
  read-only — it never fits or saves during live trading, only `train_bot` does.
- **Bootstrapping**: before a saved file exists, or before `min_samples` trades have
  accumulated, the filter is a no-op — every action passes through unfiltered — so
  early training and first live runs aren't blocked waiting on data that doesn't
  exist yet.
- **Persistence**: raw `(X, y)` samples, not the fitted model itself, are pickled to
  `LSTM-PPO-saves/{symbol}.winrate.pkl`, so the sample set survives an algorithm
  change and keeps accumulating indefinitely — capped at `max_mb` (default 20MB) by
  trimming to the most recently seen samples once the pickled size exceeds it.

### Training — `train_bot()`

- Loads the CSV, adds indicators, adds macro z-score features (`dxy_zscore`,
  `real_yield_zscore`, `gld_flow_zscore`).
- Simulates trading bar-by-bar with a fixed take-profit (`TP_PIPS = 20`) and a
  recovery-factor-throttled stop-loss (`BASE_SL_PIPS = 40`, scaled down to as little as
  `MIN_SL_SCALE = 1/3` of that when the trailing 50-trade recovery factor drops below
  `TARGET_RF = 1.0`).
- Every simulated week (`save_count = 1440 * 5` one-minute bars ≈ 5 trading days):
  preloads that week's feature rows as one vectorized slice, prints a weekly report
  (trade count, PnL, winrate, mean win/loss, max drawdown, profit factor, recovery
  factor, Sharpe, Sortino, Z-score, average win/loss streak), trains the PPO agent,
  unconditionally clears its trajectory buffer, and saves a checkpoint.
- Loops indefinitely, re-running the full dataset each "pass," until interrupted.

### Live trading — `test_bot(symbol)`

Connects to MetaTrader5, polls for new 1-minute candles, recomputes indicators on a
rolling ~3000-bar window each time a candle closes, and feeds the trained agent's
action into `open_long()` / `open_short()` / `close_trades()` / `modify_sl()`.

## Configuration

Key constants (edit in-file — there's no config file):

| Constant | Where | Purpose |
|---|---|---|
| `SEQ_LEN` | `train_bot`, `test_bot` | LSTM lookback window, in bars |
| `TP_PIPS` / `BASE_SL_PIPS` | `train_bot`, `test_bot` | Fixed take-profit / base stop-loss |
| `TARGET_RF` / `MIN_SL_SCALE` | `train_bot`, `test_bot` | Recovery-factor stop-loss throttle |
| `save_count` | `train_bot` | Bars per simulated "week" (report/train/checkpoint cadence) |
| `magic=123456` | throughout | MT5 order/position tag identifying this EA's trades |
| `MIN_WINRATE` | `train_bot`, `test_bot` | Win-rate filter threshold a setup's predicted win probability must clear to open a trade (currently hardcoded to `0.9`; the commented-out alternative is the RR-implied breakeven, `1 / (1 + RR_RATIO)`) |
| `min_samples` / `max_mb` | `WinRateModel.__init__` | Minimum accumulated trades before the win-rate filter starts predicting / max `.winrate.pkl` size before oldest samples are trimmed |

## Notes

- `state_size` (i.e. `len(FEATURES)`) must match whatever a saved checkpoint was
  trained with — after any change to the feature set, retrain from scratch before
  `test_bot` can load the new checkpoint. The same applies to `{symbol}.winrate.pkl`:
  its stored feature vectors are shaped by whatever `FEATURES` was at the time each
  sample was recorded, so a feature-set change also makes the accumulated win-rate
  samples stale and it should be rebuilt from a fresh `train_bot` run.
- `PDPOCDistance()` expects a `Volume` column; MT5's live feed only has `tick_volume`,
  which `test_bot` renames to `Volume` before calling `add_indicators`.

## bot_v2.py — three-strategy LSTM-PPO variant

A second, independent strategy set in its own file, built the same way `bot.py` itself
is: it *executes* bar-by-bar on **1-minute candles** (order fills, SL/TP hit detection
all happen at 1m granularity) but *analyzes* higher timeframes — every 1m row sees the
full indicator + entry-signal stack recomputed on **5m, 15m, 1h and 4h** candles
resampled from that same 1m data (merged back on with a `5m_`/`15m_`/`1h_`/`4h_` prefix,
forward-filled so a bar only ever sees the most recently *closed* higher-tf candle — no
lookahead), plus a second stack computed directly on the **1m** data itself (prefixed
`1m_`). This is exactly `bot.py`'s own multi-timeframe merge, just run across five
timeframes instead of two, and with a much smaller indicator stack per timeframe. Same
agent/checkpoint/GBDT-win-rate-filter skeleton as `bot.py`, and its own save directory
(`LSTM-PPO-saves-stoch-halftrend/`) so the two bots never collide or load each other's
(differently-shaped) checkpoints.

Three independent entry strategies share one LSTM-PPO agent, one GBDT win-rate filter,
and one 1m execution loop — on any bar, at most one can produce a candidate trade; see
["Combining the three strategies"](#combining-the-three-strategies) for how ties are
resolved.

```bash
python bot_v2.py --train                                # train against historical data
python bot_v2.py --test --symbol XAUUSD-STDc --risk 0.01 # trade live via MT5
python bot_v2.py --train --test --risk 0.02 --htf-mode both --min-winrate 0.4
```

### Strategy 1 — stoch/%R zone-breakout (1:4 RR)

**Indicators, per analyzed timeframe** (`_add_base_indicators()`) — computed
independently on each of 5m, 15m, 1h and 4h:

- EMA 7 / 21, each vs. price (distance) and its own slope
- HalfTrend (bullish/bearish), plus its distance from price
- ADX (+DI/-DI) — used as a trend-strength floor (`ADX_MIN = 20`)
- Stochastic oscillator (%K / %K-smooth) and Williams %R
- The stoch/%R zone-breakout entry signal itself (below)

**Entry gate** (`stoch_r_zone_breakout()`, run on every analyzed timeframe): a bar (of
whichever timeframe) counts as **oversold** once %K < 20 OR %R < -80, **overbought**
once %K > 80 OR %R > -20. Once either state has held for `ZONE_BARS` (5) consecutive
bars *of that timeframe*, that run's high/low forms a "zone" — a close breaking above
the zone high (confirmed by %K crossing back above %K-smooth) is a bullish breakout
signal; a close breaking below the zone low (%K crossing back below %K-smooth) is
bearish. A candidate fires whenever **any** of the 4 analyzed timeframes signals a
breakout with that same timeframe's own ADX clearing `ADX_MIN` — see
`_signal_columns()`.

**Direction filter:** only becomes a trade if the 1h and/or 4h HalfTrend agrees with
its direction — `--htf-mode any` (default) needs one of the two, `--htf-mode both`
needs both. The bot never trades against both.

**Risk/reward:** 1:4 — a 50-pip stop, 200-pip target (`SL_PIPS` / `RR_RATIO` /
`TP_PIPS`).

### Strategies 2 & 3 — 1m-HalfTrend-gated level strategies (1:2 RR)

Both computed directly on the raw 1m execution data (`_add_m1_levels()`,
`_level_signal_columns()`), both sharing the same 50-pip stop but a 1:2 RR — a
100-pip target (`LEVEL_RR_RATIO` / `LEVEL_TP_PIPS`), half strategy 1's — and both gated
by **1m HalfTrend**. The order-block and PDH/PDL/Asia-session indicators
(`BullishOB`/`BearishOB`/`OBMitigation`/`PDHDistance`/`PDLDistance`/
`AsiaHighDistance`/`AsiaLowDistance`) are ported from `bot.py`'s own — see "Notes /
interpretation choices" below for the two adaptations made porting them.

**2. PDH/PDL + Asia high/low break, retest and reversal** (`poi_reversal`): a candidate
fires in whichever direction the 1m HalfTrend currently points, on any bar where price
is within `POI_REACH_PIPS` (30) of the previous day's high, the previous day's low, the
current Asia session's high, or its low. HalfTrend's live state stands in for whichever
of breakout/retest/reversal is actually happening at the level — continuing through it
trades as a breakout, flipping at it trades as a reversal — rather than the code
classifying the three separately.

**3. OB mitigation** (`ob_mitigation`): a candidate fires when the 1m HalfTrend
*flips* direction (not just agrees, per spec's "halftrend reversal") on a bar that's
also mitigating — within that same `POI_REACH_PIPS` reach — a same-direction order
block: a bullish/demand OB mitigated with HalfTrend flipping up is a reversal long, a
bearish/supply OB mitigated with HalfTrend flipping down is a reversal short.

### Combining the three strategies

Across all three, the LSTM-PPO agent decides whether to actually take whichever
candidate wins (or hold) — the deterministic technical stack always picks direction
(and which strategy, and that strategy's own TP), the same "agent times entries,
technicals pick direction" split `bot.py`'s own HalfTrend-redirect uses; the agent's 3
actions are **0=buy, 1=sell, 2=hold** (`BUY`/`SELL`/`HOLD` in the file — note this
ordering is not the same as `bot.py`'s own agent).

More than one strategy firing on the same bar (in the same or opposite directions) is
rare given how differently each triggers, and resolved by a fixed priority order rather
than reconciled — see `_select_candidate()`: stoch/%R breakout first, then the
PDH/PDL/Asia-level reversal, then OB mitigation.

The GBDT win-rate filter and its risk multiplier (below) are shared across all three —
one filter, fed the full feature vector regardless of which strategy fired, so it
learns per-setup-type patterns implicitly rather than needing a separate filter per
strategy.

### Weekly stats

Same metrics as `bot.py`'s own weekly report — trade count, PnL (pips and R-multiple),
max drawdown, win rate, mean win/loss, average win/loss streak, Z-score, profit factor,
recovery factor, Sharpe, Sortino — plus the current `MIN_WINRATE`, whether the GBDT
filter is active yet or still bootstrapping, and a trades-by-strategy breakdown
(`stoch=`/`poi=`/`ob=` counts) that `bot.py`'s single-strategy report has no need for.
Unlike `bot.py`, which skips the report on a week with 5 or fewer trades, `bot_v2.py`
prints every week regardless of trade count — including a quiet week with none at all —
so there's never a silent gap between reports. Printed every `TRADING_WEEK_BARS`
(1440 × 5 = one 5-day week of 1-minute bars, same definition as `bot.py`'s `save_count`)
during `train_bot()`.

### Risk sizing

A buy/sell from any of the three strategies only clears the GBDT win-rate filter once
it predicts a win rate at or above the min-winrate bar — `BASE_MIN_WINRATE` by default
(35%, flat — well above either strategy tier's own RR-implied breakeven), overridable
with `--min-winrate` for both `--train` and `--test` (e.g. `--min-winrate 0.4`). One
shared bar, applied the same way to the 1:4 RR stoch/%R breakout and the two 1:2 RR
level strategies alike — not tiered per RR.

Position risk scales with the filter's confidence: once active, every full 10
percentage points its predicted win rate clears above that minimum adds one more unit
of the base `--risk` to the position (`risk_multiplier()`) — a barely-qualifying setup
risks the plain `--risk` amount, a strongly-favoured one risks several multiples of it
(capped at `MAX_RISK_MULTIPLIER = 5`).

### GBDT win-rate filter — `GBDTWinRateFilter`

Same accumulate-then-refit XGBoost design as `bot.py`'s `WinRateModel`, refit at the
same weekly cadence as the PPO agent, with one addition: it doesn't **gate** any trade
until `WEEKS_BEFORE_FILTER` (5) simulated training weeks have completed
(`weeks_trained`, persisted alongside the sample pickle) — before that it keeps
fitting/accumulating samples in the background so it's warm by week 5, but never blocks
a trade, so the first weeks of training aren't starved waiting on data that doesn't
exist yet. Persisted to
`LSTM-PPO-saves-stoch-halftrend/{symbol}-stoch-halftrend.gbdt_winrate.pkl`.

### Notes / interpretation choices

- The stoch/%R breakout spec ("stoch k </> k smooth, and/or %r, wait 5 candles in
  ob/os zone, then breakout, executed on 5m/15m/1h/4h, only in direction with 1h and/or
  4h halftrend") admits more than one reading — the thresholds, per-timeframe "any one
  fires" combination, and confirmation logic above are one reasonable interpretation;
  adjust the constants near the top of `bot_v2.py` (`STOCH_OS`/`STOCH_OB`,
  `WR_OS`/`WR_OB`, `ZONE_BARS`, `ADX_MIN`, `ANALYZED_TIMEFRAMES`) or
  `_signal_columns()`'s combination logic if a different one was intended.
- Porting `bot.py`'s `OBMitigation()`: its own `threshold` parameter is a **raw price**
  distance (its default of 30 is literally $30, i.e. 300 pips at `PIP_VALUE=0.1`) —
  `bot_v2.py`'s version takes `threshold_pips` instead, so its default of 30 is
  genuinely 30 pips, matching spec ("same 30 pip reach").
- Porting `bot.py`'s `AsiaHighDistance()`/`AsiaLowDistance()`: their session mask,
  `(hour >= 1) | (hour <= 9)`, is a bug in `bot.py` (that OR covers nearly the entire
  day, not a session) — `bot_v2.py`'s versions use AND, matching those functions' own
  "01:00–08:59" docstring and `bot.py`'s `GetKillzone()` Asia-session convention.
- "Add 1R to risk per +10% predicted win rate" is implemented as a risk *multiplier*
  (1 + one extra unit of `--risk` per 10-point margin above the win-rate bar, capped at
  5x) rather than a running additive R-ladder — see `risk_multiplier()`.
- `state_size` (`len(FEATURES)`, 78 = 17 indicators × 4 analyzed timeframes + 10 1m
  level features) must match whatever a saved checkpoint was trained with, same caveat
  as `bot.py`'s own — a feature-set change needs a fresh `train_bot()` run before
  `test_bot()` can load the new checkpoint, and makes any accumulated
  `.gbdt_winrate.pkl` samples stale.
- `test_bot()` requires the `MetaTrader5` package and a running MT5 terminal
  (Windows-only); `train_bot()` and the indicator pipeline have no such dependency and
  run anywhere.
