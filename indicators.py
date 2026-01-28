import numpy as np
from numba import njit


# ============================================================
# Helper: cast to float32 safely
# ============================================================
def to_f32(arr) -> np.ndarray:
    """
    Ensure input is a 1D float32 numpy array.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 1:
        raise ValueError("Input must be 1D array")
    return a


# ============================================================
# Helper: simple moving average (SMA)
# ============================================================
@njit
def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Simple moving average using float32.
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period > n:
        return out

    csum = np.float32(0.0)
    for i in range(n):
        csum += values[i]
        if i >= period:
            csum -= values[i - period]
        if i >= period - 1:
            out[i] = csum / np.float32(period)
    return out

# ============================================================
# simple moving average (SMA)
# ============================================================

@njit
def sma(arr: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (SMA).
    Zwraca float32, zgodne ze sztuką.
    """
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float32)

    # init NaN
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period > n:
        return out

    # rolling sum
    s = 0.0
    for i in range(period):
        s += arr[i]

    out[period - 1] = s / period

    for i in range(period, n):
        s += arr[i] - arr[i - period]
        out[i] = s / period

    return out

# ============================================================
# EMA — Exponential Moving Average
# ============================================================
@njit
def ema(arr: np.ndarray, period: int):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float32)

    alpha = 2.0 / (period + 1.0)

    # Start EMA od pierwszej wartości (standard w tradingu)
    out[0] = arr[0]

    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]

    return out

# ============================================================
# RSI — Relative Strength Index
# ============================================================
@njit
def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI).
    Input:
        close: float32 closing prices
        period: lookback period
    Output:
        RSI values between 0 and 100 (float32).
        >70 = overbought, <30 = oversold.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period >= n:
        return out

    gains = np.empty(n, dtype=np.float32)
    losses = np.empty(n, dtype=np.float32)
    gains[0] = np.float32(0.0)
    losses[0] = np.float32(0.0)

    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gains[i] = delta
            losses[i] = np.float32(0.0)
        else:
            gains[i] = np.float32(0.0)
            losses[i] = -delta

    avg_gain = np.float32(0.0)
    avg_loss = np.float32(0.0)
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= np.float32(period)
    avg_loss /= np.float32(period)

    if avg_loss == 0.0:
        out[period] = np.float32(100.0)
    else:
        rs = avg_gain / avg_loss
        out[period] = np.float32(100.0) - (np.float32(100.0) / (np.float32(1.0) + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / np.float32(period)
        avg_loss = (avg_loss * (period - 1) + losses[i]) / np.float32(period)

        if avg_loss == 0.0:
            out[i] = np.float32(100.0)
        else:
            rs = avg_gain / avg_loss
            out[i] = np.float32(100.0) - (np.float32(100.0) / (np.float32(1.0) + rs))

    return out


# ============================================================
# Stochastic RSI
# ============================================================
@njit
def stoch_rsi(close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14) -> np.ndarray:
    """
    Stochastic RSI.
    Input:
        close: float32 closing prices
        rsi_period: RSI lookback period
        stoch_period: Stoch lookback over RSI
    Output:
        Values between 0 and 1 (float32).
        >0.8 = overbought, <0.2 = oversold.
    """
    r = rsi(close, rsi_period)
    n = r.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if stoch_period <= 0 or stoch_period >= n:
        return out

    for i in range(n):
        if i < stoch_period - 1:
            out[i] = np.nan
        else:
            window_min = r[i]
            window_max = r[i]
            for j in range(i - stoch_period + 1, i + 1):
                val = r[j]
                if np.isnan(val):
                    continue
                if val < window_min:
                    window_min = val
                if val > window_max:
                    window_max = val
            if window_max == window_min:
                out[i] = np.float32(0.0)
            else:
                out[i] = (r[i] - window_min) / (window_max - window_min)
    return out

# ============================================================
# Stochastic
# ============================================================

@njit
def stoch(high: np.ndarray,
             low: np.ndarray,
             close: np.ndarray,
             k_period: int = 14,
             d_period: int = 3):
    """
    Classic Stochastic Oscillator (%K and %D)
    Fully correct version (TradingView-compatible).
    Returns %K and %D in range 0–1 (float32).
    """

    n = close.shape[0]
    k = np.empty(n, dtype=np.float32)
    d = np.empty(n, dtype=np.float32)

    # init NaN
    for i in range(n):
        k[i] = np.nan
        d[i] = np.nan

    if k_period <= 0 or k_period >= n:
        return k, d

    # compute %K (full window scan every time)
    for i in range(k_period - 1, n):
        lowest_low = low[i]
        highest_high = high[i]

        # full scan of window -> EXACT TradingView logic
        for j in range(i - k_period + 1, i + 1):
            if low[j] < lowest_low:
                lowest_low = low[j]
            if high[j] > highest_high:
                highest_high = high[j]

        if highest_high == lowest_low:
            k[i] = 0.0
        else:
            k[i] = (close[i] - lowest_low) / (highest_high - lowest_low)

    # compute %D = SMA(%K, d_period)
    for i in range(k_period - 1 + d_period - 1, n):
        s = 0.0
        count = 0
        for j in range(i - d_period + 1, i + 1):
            val = k[j]
            if not np.isnan(val):
                s += val
                count += 1
        if count > 0:
            d[i] = s / count

    return k, d

# ============================================================
# Stochastic - signal
# ============================================================

@njit
def stoch_signal(k: np.ndarray,
                  d: np.ndarray,
                  oversold: float = 0.2,
                  overbought: float = 0.8):
    """
    Generate trading signals from Stochastic %K and %D.
    Returns:
        1  = long signal
        -1 = short signal
        0  = no signal
    token:
           sygnał LONG → %K przecina %D od dołu, a oba są < 0.2
           sygnał SHORT → %K przecina %D od góry, a oba są > 0.8
    """

    n = k.shape[0]
    out = np.zeros(n, dtype=np.int8)

    for i in range(1, n):

        # LONG: K crosses above D in oversold zone
        if (
            k[i-1] < d[i-1] and
            k[i] > d[i] and
            k[i] < oversold and
            d[i] < oversold
        ):
            out[i] = 1

        # SHORT: K crosses below D in overbought zone
        elif (
            k[i-1] > d[i-1] and
            k[i] < d[i] and
            k[i] > overbought and
            d[i] > overbought
        ):
            out[i] = -1

    return out

# ============================================================
# ROC — Rate of Change
# ============================================================
# token: checked 2025/12/29 OK
@njit
def roc(close: np.ndarray, period: int = 12) -> np.ndarray:
    """
    Rate of Change (ROC).
    Input:
        close: float32 closing prices
        period: lookback period
    Output:
        Percentage change over 'period' (float32).
        Positive = upward momentum, Negative = downward momentum.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period >= n:
        return out

    for i in range(period, n):
        prev = close[i - period]
        if prev == 0.0:
            out[i] = np.nan
        else:
            out[i] = (close[i] / prev - np.float32(1.0)) * np.float32(100.0)
    return out


# ============================================================
# True Range & ATR
# ============================================================
@njit
def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Range (TR) per bar.
    """
    n = close.shape[0]
    tr = np.empty(n, dtype=np.float32)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        m = hl
        if hc > m:
            m = hc
        if lc > m:
            m = lc
        tr[i] = m
    return tr


# ============================================================
# ATR (Average True Range)
# ============================================================
@njit
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range (ATR).
    """
    tr = true_range(high, low, close)
    n = tr.shape[0]
    out = np.empty(n, dtype=np.float32)

    # pierwsza wartość ATR = średnia TR z pierwszych 'period' świec
    if n < period:
        for i in range(n):
            out[i] = np.nan
        return out

    s = np.float32(0.0)
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period

    # kolejne wartości ATR = Wilder smoothing
    alpha = np.float32(1.0 / period)
    for i in range(period, n):
        out[i] = (out[i - 1] * (1.0 - alpha)) + (tr[i] * alpha)

    # wcześniejsze wartości = NaN
    for i in range(period - 1):
        out[i] = np.nan

    return out


# ============================================================
# ATRP (ATR Percent)
# ============================================================
@njit
def atrp(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    ATR Percent (ATRP) = (ATR / Close) * 100
    """
    atr_vals = atr(high, low, close, period)
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)

    for i in range(n):
        c = close[i]
        if c == 0.0 or np.isnan(atr_vals[i]):
            out[i] = np.nan
        else:
            out[i] = (atr_vals[i] / c) * np.float32(100.0)

    return out



# ============================================================
# MFI — Money Flow Index
# ============================================================
@njit
def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Money Flow Index (MFI).
    Input:
        high, low, close: OHLC arrays
        volume: traded volume
        period: lookback period
    Output:
        Values between 0 and 100 (float32).
        >80 = overbought, <20 = oversold.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period >= n:
        return out

    typical_price = (high + low + close) / np.float32(3.0)
    money_flow = typical_price * volume

    pos_flow = np.zeros(n, dtype=np.float32)
    neg_flow = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            pos_flow[i] = money_flow[i]
            neg_flow[i] = np.float32(0.0)
        elif typical_price[i] < typical_price[i - 1]:
            pos_flow[i] = np.float32(0.0)
            neg_flow[i] = money_flow[i]
        else:
            pos_flow[i] = np.float32(0.0)
            neg_flow[i] = np.float32(0.0)

    for i in range(period, n):
        pos_sum = np.float32(0.0)
        neg_sum = np.float32(0.0)
        for j in range(i - period + 1, i + 1):
            pos_sum += pos_flow[j]
            neg_sum += neg_flow[j]
        if neg_sum == 0.0:
            out[i] = np.float32(100.0)
        else:
            mf_ratio = pos_sum / neg_sum
            out[i] = np.float32(100.0) - (np.float32(100.0) / (np.float32(1.0) + mf_ratio))

    return out


# ============================================================
# OBV — On-Balance Volume
# ============================================================
@njit
def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    On-Balance Volume (OBV).
    Input:
        close: closing prices
        volume: traded volume
    Output:
        Cumulative volume flow (float32).
        Rising OBV = buying pressure, Falling OBV = selling pressure.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out

    out[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - volume[i]
        else:
            out[i] = out[i - 1]

    return out


# ============================================================
# Bollinger Bands
# ============================================================
@njit
def bollinger_bands(close: np.ndarray, period: int = 20, std_factor: float = 2.0):
    n = close.shape[0]

    sma = _sma(close, period)  # załóżmy, że _sma jest już Numba-safe i bez NaN

    std = np.zeros(n, dtype=np.float32)

    if period > 0 and period <= n:
        for i in range(period - 1, n):
            mean = sma[i]
            acc = 0.0
            for j in range(i - period + 1, i + 1):
                diff = close[j] - mean
                acc += diff * diff
            std[i] = np.sqrt(acc / period)

    upper = sma + std_factor * std
    lower = sma - std_factor * std

    return upper, sma, lower

# ============================================================
# Bollinger Bands CROSS
# ============================================================

@njit
def bbands_cross(close: np.ndarray, upper: np.ndarray, lower: np.ndarray):
    n = close.shape[0]
    out = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if close[i] <= lower[i]:
            out[i] = 1      # strefa LONG
        elif close[i] >= upper[i]:
            out[i] = -1     # strefa SHORT
        else:
            out[i] = 0      # neutral

    return out



# ============================================================
# MACD — Moving Average Convergence Divergence
# ============================================================
@njit
def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD indicator.
    Input:
        close: closing prices
        fast: fast EMA period
        slow: slow EMA period
        signal: signal EMA period
    Output:
        macd_line, signal_line, histogram (float32 arrays).
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    n = close.shape[0]
    macd_line = np.empty(n, dtype=np.float32)
    for i in range(n):
        macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = ema(macd_line, signal)
    hist = np.empty(n, dtype=np.float32)
    for i in range(n):
        hist[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, hist

# ============================================================
# MACD — Moving Average Convergence Divergence - histogram reversal
# ============================================================
@njit
def macd_hist_signal_filtered(hist: np.ndarray):
    """
    MACD histogram reversal with trend context.
    Returns:
        1  -> histogram flips upward below zero (LONG)
        -1 -> histogram flips downward above zero (SHORT)
        0  -> no signal
        token: xzmiana kierunku histogramu.
    """
    n = hist.shape[0]
    out = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        prev = hist[i-1]
        curr = hist[i]

        # LONG: histogram rośnie i jest poniżej zera
        if curr > prev and curr < 0:
            out[i] = 1

        # SHORT: histogram spada i jest powyżej zera
        elif curr < prev and curr > 0:
            out[i] = -1

        else:
            out[i] = 0

    return out


# ============================================================
# CCI — Commodity Channel Index
# ============================================================
@njit
def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Commodity Channel Index (CCI).
    Input:
        high, low, close: OHLC arrays
        period: lookback period
    Output:
        CCI values (float32), typically between -200 and +200.
        >+100 = overbought, <-100 = oversold.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period >= n:
        return out

    tp = (high + low + close) / np.float32(3.0)
    sma_tp = _sma(tp, period)

    for i in range(period - 1, n):
        mean = sma_tp[i]
        acc = np.float32(0.0)
        for j in range(i - period + 1, i + 1):
            acc += abs(tp[j] - mean)
        md = acc / np.float32(period)
        if md == 0.0:
            out[i] = np.float32(0.0)
        else:
            out[i] = (tp[i] - mean) / (np.float32(0.015) * md)

    return out


# ============================================================
# Williams %R
# ============================================================
@njit
def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Williams %R.
    Input:
        high, low, close: OHLC arrays
        period: lookback period
    Output:
        Values between -100 and 0 (float32).
        > -20 = overbought, < -80 = oversold.
    """
    n = close.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.nan

    if period <= 0 or period >= n:
        return out

    for i in range(period - 1, n):
        highest = high[i]
        lowest = low[i]
        for j in range(i - period + 1, i + 1):
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        rng = highest - lowest
        if rng == 0.0:
            out[i] = np.float32(0.0)
        else:
            out[i] = -np.float32(100.0) * (highest - close[i]) / rng

    return out

# ============================================================
# ADX i DI + / -
# ============================================================
@njit
def adx_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """
    ADX + DI+ + DI- w logice zgodnej z TradingView:
    - DI length = period
    - ADX smoothing = period
    Zwraca:
        adx      -> siła trendu (0–100)
        di_plus  -> DI+ (0–100)
        di_minus -> DI- (0–100)
    Wszystko float32.
    """
    n = close.shape[0]

    adx = np.empty(n, dtype=np.float32)
    di_plus = np.empty(n, dtype=np.float32)
    di_minus = np.empty(n, dtype=np.float32)

    adx[:] = np.nan
    di_plus[:] = np.nan
    di_minus[:] = np.nan

    if period <= 0 or n < period * 2:
        return adx, di_plus, di_minus

    # -----------------------------------------
    # 1. DM+ / DM-
    # -----------------------------------------
    plus_dm = np.zeros(n, dtype=np.float32)
    minus_dm = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]

        if up > 0.0 and up > down:
            plus_dm[i] = up
        if down > 0.0 and down > up:
            minus_dm[i] = down

    # -----------------------------------------
    # 2. ATR (Wilder RMA – Twoja funkcja atr)
    # -----------------------------------------
    atr_vals = atr(high, low, close, period)

    # -----------------------------------------
    # 3. RMA(DM+) i RMA(DM-)  — klucz TradingView
    # -----------------------------------------
    rma_plus = np.empty(n, dtype=np.float32)
    rma_minus = np.empty(n, dtype=np.float32)
    rma_plus[:] = np.nan
    rma_minus[:] = np.nan

    s_plus = 0.0
    s_minus = 0.0
    for i in range(period):
        s_plus += plus_dm[i]
        s_minus += minus_dm[i]

    rma_plus[period - 1] = s_plus / period
    rma_minus[period - 1] = s_minus / period

    for i in range(period, n):
        rma_plus[i] = (rma_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        rma_minus[i] = (rma_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # -----------------------------------------
    # 4. DI+ / DI-
    # -----------------------------------------
    for i in range(period - 1, n):
        atr_i = atr_vals[i]
        if atr_i > 0.0 and not np.isnan(atr_i):
            di_plus[i] = (rma_plus[i] / atr_i) * 100.0
            di_minus[i] = (rma_minus[i] / atr_i) * 100.0
        else:
            di_plus[i] = 0.0
            di_minus[i] = 0.0

    # -----------------------------------------
    # 5. DX
    # -----------------------------------------
    dx = np.empty(n, dtype=np.float32)
    dx[:] = np.nan

    for i in range(period - 1, n):
        denom = di_plus[i] + di_minus[i]
        if denom > 0.0:
            dx[i] = abs(di_plus[i] - di_minus[i]) / denom * 100.0
        else:
            dx[i] = 0.0

    # -----------------------------------------
    # 6. ADX = RMA(DX)
    # -----------------------------------------
    start = period * 2 - 1
    if start >= n:
        return adx, di_plus, di_minus

    s_dx = 0.0
    cnt = 0
    for i in range(period - 1, start + 1):
        val = dx[i]
        if not np.isnan(val):
            s_dx += val
            cnt += 1

    adx[start] = s_dx / cnt

    for i in range(start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, di_plus, di_minus


# ============================================================
# ADX (wraper)
# ============================================================
@njit
def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wrapper na adx_di – zwraca tylko ADX,
    żeby zachować kompatybilność ze starym API.
    """
    adx_vals, _, _ = adx_di(high, low, close, period)
    return adx_vals


# ============================================================
# Keltner channel
# ============================================================


@njit
def keltner_channel(high, low, close, period=20, atr_period=14, multiplier=1.5):
    """
    Numba-safe Keltner Channel.

    Parameters:
        high (np.ndarray): High prices
        low (np.ndarray): Low prices
        close (np.ndarray): Close prices
        period (int): EMA period for middle band
        atr_period (int): ATR period
        multiplier (float): ATR multiplier for channel width

    Returns:
        middle (np.ndarray): EMA(period)
        upper (np.ndarray): EMA(period) + ATR * multiplier
        lower (np.ndarray): EMA(period) - ATR * multiplier

    token: 2026/01/05 - tradingview ok, ale TV liczy 20, 10, 2  -w tych ustawieniahc wyniki sa zbiezne
    """

    n = close.shape[0]

    # --- EMA ---
    middle = np.empty(n, dtype=np.float64)
    alpha = 2.0 / (period + 1.0)

    middle[0] = close[0]
    for i in range(1, n):
        middle[i] = alpha * close[i] + (1 - alpha) * middle[i - 1]

    # --- ATR ---
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha_atr = 1.0 / atr_period

    for i in range(1, n):
        atr[i] = atr[i - 1] + alpha_atr * (tr[i] - atr[i - 1])

    # --- Upper / Lower bands ---
    upper = middle + atr * multiplier
    lower = middle - atr * multiplier

    return middle, upper, lower


# ============================================================
# Supertrend
# ============================================================

@njit
def supertrend(high, low, close, period=10, multiplier=3.0):

    # st → linia SuperTrendu
    # trend →
    #     1 = trend wzrostowy
    #     -1 = trend spadkowy
    # uzycie:
    # st, trend = supertrend(high, low, close)
    #
    # df["supertrend"] = st
    # df["supertrend_trend"] = trend


    n = close.shape[0]

    # --- ATR ---
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    alpha = 1.0 / period

    for i in range(1, n):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    # --- Basic bands ---
    upper_basic = (high + low) / 2 + multiplier * atr
    lower_basic = (high + low) / 2 - multiplier * atr

    # --- Final bands ---
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)

    upper[0] = upper_basic[0]
    lower[0] = lower_basic[0]

    for i in range(1, n):
        upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]

    # --- SuperTrend line ---
    st = np.empty(n, dtype=np.float64)
    trend = np.empty(n, dtype=np.int8)

    st[0] = lower[0]
    trend[0] = 1  # start as uptrend

    for i in range(1, n):
        if st[i - 1] == upper[i - 1] and close[i] > upper[i]:
            trend[i] = 1
        elif st[i - 1] == lower[i - 1] and close[i] < lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        st[i] = lower[i] if trend[i] == 1 else upper[i]

    return st, trend


# ============================================================
# VWAP - Volume Weighted Average Price
# ============================================================

@njit
def vwap(high, low, close, volume):
    typical = (high + low + close) / 3
    pv = typical * volume
    return np.cumsum(pv) / np.cumsum(volume)



# ============================================================
# Basic sanity tests
# ============================================================
def _run_tests():
    """
    Basic sanity tests for all indicators using random data.
    These are not full unit tests, but ensure functions run correctly
    and return arrays of expected shapes.
    """
    print("Running numba float32 indicator tests...")

    n = 500
    close = to_f32(np.linspace(100, 200, n) + np.random.randn(n) * 2.0)
    high = to_f32(close + np.random.rand(n) * 2.0)
    low = to_f32(close - np.random.rand(n) * 2.0)
    volume = to_f32(np.random.randint(100, 1000, n))

    rsi_vals = rsi(close, 14);        assert rsi_vals.shape[0] == n
    stoch_vals = stoch_rsi(close);    assert stoch_vals.shape[0] == n
    roc_vals = roc(close, 12);        assert roc_vals.shape[0] == n
    atr_vals = atr(high, low, close); assert atr_vals.shape[0] == n
    mfi_vals = mfi(high, low, close, volume); assert mfi_vals.shape[0] == n
    obv_vals = obv(close, volume);    assert obv_vals.shape[0] == n

    upper, mid, lower = bollinger_bands(close); 
    assert upper.shape[0] == n and mid.shape[0] == n and lower.shape[0] == n

    macd_line, signal_line, hist = macd(close)
    assert macd_line.shape[0] == n and signal_line.shape[0] == n and hist.shape[0] == n

    cci_vals = cci(high, low, close);         assert cci_vals.shape[0] == n
    wr_vals = williams_r(high, low, close);   assert wr_vals.shape[0] == n
    adx_vals = adx(high, low, close);         assert adx_vals.shape[0] == n

    print("All numba float32 indicators tests passed.")


if __name__ == "__main__":
    _run_tests()
