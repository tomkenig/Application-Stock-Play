import ccxt
import numpy as np
import json
import time
import uuid
import order_event



from indicators_mod import (
    get_indicator_roc,
    get_indicator_rsi,
)

from signal_generator import SignalGenerator, SignalEvent
from symbol_resolver import resolve_symbol
from trading_engine import TradingEngine
from datetime import datetime, timezone
from signal_generator import SignalEvent
from order_manager import OrderManager

# ============================
# Load configs
# ============================
def load_json(path):
    # Reads JSON file and returns Python dict
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================
# Safe fetch OHLCV
# ============================
def safe_fetch_ohlcv(exchange_client, symbol, timeframe, limit):
    """
    Fetches OHLCV candles safely.
    Returns numpy array or None on error.
    """
    try:
        ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return np.asarray(ohlcv, dtype=np.float32)
    except Exception as e:
        print("[ERROR] fetch_ohlcv:", e)
        return None


# ============================
# Heartbeat
# ============================
def write_heartbeat():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open("heartbeat.log", "a", encoding="utf-8") as f:
        f.write(f"{ts} — OK\n")
    print("\n==============================")
    print(f"\nHeartbeat...\n")


# ============================
# UUID
# ============================
def generate_id():
    return str(uuid.uuid4())

# ============================
# MAIN
# ============================
if __name__ == "__main__":

    # Load configuration files
    app_config = load_json("app_config.json")
    tactics = load_json("strategy_config.json")
    credentials = load_json("credentials.json")

    # Resolve trading symbol (e.g. BTC/USDT)
    exchange_name = app_config["exchange"]
    base = app_config["symbol_base"]
    quote = app_config["symbol_quote"]
    symbol = resolve_symbol(exchange_name, base, quote)

    # Timeframe and candle limit
    timeframe = app_config["tick_interval"]
    candle_limit = app_config["candle_limit"]

    # Run mode: on_tick (wait for candle close) or interval (fixed sleep)
    run_mode = app_config.get("run_mode", "on_tick")
    run_interval_seconds = app_config.get("run_interval_seconds", 10)

    # ============================
    # Validate strategy intervals
    # ============================
    for tactic in tactics:
        tactic_tf = tactic.get("tick_interval")
        if tactic_tf != timeframe:
            print(f"[ERROR] Strategy '{tactic['tactic_group_name']}' uses tick_interval={tactic_tf}, "
                  f"but app instance runs on {timeframe}.")
            print("Fix strategy_config.json before running.")
            exit(1)

    # ============================
    # Exchange init
    # ============================
    creds = credentials.get(exchange_name, {})
    exchange_client = getattr(ccxt, exchange_name)({
        "apiKey": creds.get("apiKey"),
        "secret": creds.get("secret"),
        "password": creds.get("password"),
    })

    # ============================
    # Indicator mapping
    # ============================
    INDICATOR_FUNCTIONS = {
        "get_indicator_roc": get_indicator_roc,
        "get_indicator_rsi": get_indicator_rsi,
    }

    # Create signal generator
    signal_generator = SignalGenerator(INDICATOR_FUNCTIONS)

    print("=== SIGNAL GENERATOR STARTED ===")
    print("Run mode:", run_mode)

    # ============================
    # Trading Engine
    # ============================

    # ile ticków sygnał ma przeżyć
    expiration_ticks = app_config.get("signal_expiration_ticks", 1)

    # konwersja timeframe → sekundy
    tf = timeframe.lower()
    if tf.endswith("m"):
        tick_seconds = int(tf[:-1]) * 60
    elif tf.endswith("h"):
        tick_seconds = int(tf[:-1]) * 60 * 60
    elif tf.endswith("d"):
        tick_seconds = int(tf[:-1]) * 60 * 60 * 24
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    # tworzymy engine z parametrami
    trading_engine = TradingEngine(expiration_ticks, tick_seconds)

    # wczytujemy sygnały po restarcie
    trading_engine.load_signals_from_json("signals.json")

    # usuwamy stare sygnały
    server_ms = exchange_client.fetch_time()
    trading_engine.clear_expired_signals(server_ms)

    # zapisujemy nowa liste sygnlow po usunieciu
    trading_engine.save_signals_to_json("signals.json")

    # ============================
    # Order Manager
    # ============================

    # Tworzenie instancji OrderManager
    order_manager = OrderManager(exchange_client)

    # ============================
    # Main loop
    # ============================
    while True:

        # ----------------------------
        # RUN MODE: on_tick
        # Wait until next candle closes
        # ----------------------------
        if run_mode == "on_tick":
            try:
                server_ms = exchange_client.fetch_time()
                server_time = server_ms / 1000

                # Calculate next candle close timestamp
                next_candle = ((server_time // tick_seconds) + 1) * tick_seconds
                wait_seconds = next_candle - server_time

                print(f"Czekam {wait_seconds:.1f} sekund do zamknięcia świecy...")
                time.sleep(wait_seconds)

            except Exception as e:
                print("[WARN] fetch_time:", e)
                time.sleep(5)
                continue

        # ----------------------------
        # RUN MODE: interval
        # Fixed sleep between cycles
        # ----------------------------
        elif run_mode == "interval":
            print(f"Czekam {run_interval_seconds} sekund...")
            time.sleep(run_interval_seconds)

        # ----------------------------
        # Fetch OHLCV
        # ----------------------------
        server_ms = exchange_client.fetch_time()
        ohlc_data = safe_fetch_ohlcv(exchange_client, symbol, timeframe, candle_limit)

        if ohlc_data is None:
            time.sleep(5)
            continue

        # Extract close prices
        close = ohlc_data[:, 4]
        current_price = float(close[-1])

        print("\n==============================")
        print("Current price:", current_price)

        # ----------------------------
        # ENTRY SIGNAL HANDLER
        # ----------------------------
        # ----------------------------
        # TACTICS CHECKER - Evaluate tactics (strategies)
        # ----------------------------
        for tactic in tactics:
            print("\n==============================")
            print("\nChecking:", tactic["tactic_group_name"])

            # Compute indicator values
            indicator_values = signal_generator.compute_indicators(close, tactic)

            # ----------------------------
            # IF signal = True (signal exists)
            # ----------------------------

            # Check if tactic (strategy) conditions are met
            if signal_generator.check_tactic(indicator_values, tactic):

                # ----------------------------
                # CREATE ENTRY_SIGNAL (SIGNAL EVENT)
                # ----------------------------

                unique_id = str(generate_id())

                # 1) generation_timestamp = czas wygenerowania sygnału (czas serwera giełdy w ms)
                generation_timestamp = server_ms

                # 2) tick_timestamp = czas zamknięcia świecy, na której powstał sygnał
                #    (korzystamy z ostatnio policzonego next_candle i tick_seconds)
                current_candle_close = next_candle - tick_seconds
                tick_timestamp = int(current_candle_close * 1000)

                # 3) expiration_time = tick_timestamp + N ticków
                expiration_time = tick_timestamp + ((expiration_ticks + 1) * tick_seconds * 1000)

                entry_signal_event = SignalEvent(
                    tactic_group_name=tactic["tactic_group_name"],
                    tactic_name=tactic["tactic_name"],
                    tactic_id=tactic["tactic_id"],
                    side=tactic["tactic_side"],  # buy / sell
                    symbol=symbol,
                    entry_type=tactic["entry_type"],  # market / limit
                    price=current_price,
                    generation_timestamp=generation_timestamp,
                    tick_timestamp=tick_timestamp,
                    stake=tactic["stake"],            # amount in QUOTE
                    take_profit=tactic["take_profit"],
                    stop_loss=tactic.get("stop_loss", tactic.get("stoploss", -1)),
                    wait_periods=tactic["wait_periods"],
                    expiration_time=expiration_time,
                    internal_id=unique_id,
                    signal_id=f"ENTRY_SIGNAL_{unique_id}",
                    status="NEW",
                )

                print(f"[ENTRY_SIGNAL_EVENT] {entry_signal_event.tactic_name} | {entry_signal_event.entry_type.upper()}"
                      f" @ {entry_signal_event.price}, SIGNAL_ID: {entry_signal_event.signal_id}")

                # ----------------------------
                # ADD ENTRY_SIGNAL TO SIGNAL LIST (MEMORY) - Send event to trading engine
                # ----------------------------

                trading_engine.add_signal(entry_signal_event)

                # ----------------------------
                # CREATE ENTRY_ORDER (ORDER EVENT)
                # ----------------------------

                # LIMIT BUY
                if entry_signal_event.status in ('NEW', 'CRASHED'):
                    try:
                        entry_order_event = order_manager.create_order(
                            side=entry_signal_event.side,
                            symbol=symbol,
                            order_type=entry_signal_event.entry_type,
                            amount=entry_signal_event.stake / current_price,
                            price=current_price - 3000 ############################### -3k for tests only
                            )
                        print(
                            f"[ENTRY_ORDER_EVENT] ORDER_ID: {entry_order_event.order_id}, "
                            f"STATUS: {entry_order_event.status}")
                        # SIGNAL: CHANGE SIGNAL STATUS TO ORDERED:
                        entry_signal_event.status = "USED"
                    except Exception as e:
                        print(f"[WARN] Cannot create order from signal {entry_signal_event.signal_id}:", e)
                        entry_signal_event.status = "CRASHED"
                        time.sleep(5)
                        continue

                # ----------------------------
                # ADD ENTRY_ORDER TO ORDER LIST (MEMORY) - Send event to trading engine
                # ----------------------------

                ###############trading_engine.add_signal(entry_signal_event)




        # Remove expired signals during runtime
        trading_engine.clear_expired_signals(server_ms)

        # Save updated list
        trading_engine.save_signals_to_json("signals.json")

        # Heartbeat
        write_heartbeat()

