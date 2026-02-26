# todo: jest problem - interval/on_tick - bierze inne candle i podaje inna date tick_timestamp/. Wstepne rozwiazanie - w interval badac siwece -2


import ccxt
import numpy as np
import json
import time
import uuid

from indicators_mod import (
    get_indicator_roc,
    get_indicator_rsi,
)

from signal_generator import SignalGenerator, SignalEvent
from symbol_resolver import resolve_symbol
from signal_engine import SignalEngine
from datetime import datetime, timezone
from signal_generator import SignalEvent
from order_manager import OrderManager

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__

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
    print(f"Heartbeat...")


# ============================
# UUID
# ============================
def generate_uuid():
    return str(uuid.uuid4())


# ============================
# SIDE CHOOSE
# ============================

def get_close_side(entry_side):
    if entry_side=="buy":
        return "sell"
    if entry_side=="sell":
        return "buy"


# ============================
# MAIN
# ============================
if __name__ == "__main__":

    # Load configuration files
    app_config = load_json("app_config.json")
    credentials = load_json("credentials.json")
    tactics = load_json("strategy_config.json")

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
    # Exchange init (credentials and exchange_client
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
    # Signal Engine
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
    signal_engine = SignalEngine(expiration_ticks, tick_seconds)

    # wczytujemy sygnały po restarcie
    signal_engine.load_signals_from_json("signals.json")

    # usuwamy stare sygnały
    server_ms = exchange_client.fetch_time()
    signal_engine.clear_expired_signals(server_ms)

    # zapisujemy nowa liste sygnlow po usunieciu
    signal_engine.save_signals_to_json("signals.json")

    # ============================
    # Order Manager
    # ============================

    # Tworzenie instancji OrderManager
    order_manager = OrderManager(exchange_client)
    order_manager.load_orders_from_json("orders.json")

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

                print(f"Waiting {wait_seconds:.1f} seconds to candle close...")
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
            try:
                server_ms = exchange_client.fetch_time()
                server_time = server_ms / 1000

                # Calculate next candle close timestamp
                next_candle = ((server_time // tick_seconds) + 0) * tick_seconds

                print(f"Waiting {run_interval_seconds} seconds...")
                time.sleep(run_interval_seconds)

            except Exception as e:
                print("[WARN] fetch_time:", e)
                time.sleep(5)
                continue

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
        if run_mode == "on_tick":
            current_price = float(close[-1])
        elif run_mode == "interval":
            current_price = float(close[-2])

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
            print("Checking:", tactic["tactic_group_name"])

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

                unique_id = str(generate_uuid())

                # 1) generation_timestamp = czas wygenerowania sygnału (czas serwera giełdy w ms)
                generation_timestamp = server_ms

                # 2) tick_timestamp = czas zamknięcia świecy, na której powstał sygnał
                #    (korzystamy z ostatnio policzonego next_candle i tick_seconds)
                current_candle_close = next_candle - tick_seconds
                tick_timestamp = int(current_candle_close * 1000)

                # ----------------------------
                # DUPLICATE CHECK - prevent from duplicate signal event
                # ----------------------------
                duplicate = any(
                    s.tactic_id == tactic["tactic_id"] and s.tick_timestamp == tick_timestamp
                    for s in signal_engine.pending_signals
                )

                if duplicate:
                    print(f"[SKIP] Duplicate signal detected for tactic_id={tactic['tactic_id']} "
                          f"at tick_timestamp={tick_timestamp}")
                    continue

                # 3) expiration_time = tick_timestamp + N ticków
                expiration_timestamp = tick_timestamp + ((expiration_ticks + 1) * tick_seconds * 1000)

                entry_signal_event = SignalEvent(
                    tactic_group_name=tactic["tactic_group_name"],
                    tactic_name=tactic["tactic_name"],
                    tactic_id=tactic["tactic_id"],
                    side=tactic["tactic_side"],  # buy / sell
                    symbol=symbol,
                    entry_type=tactic["entry_type"],  # market / limit
                    price=current_price,
                    generation_timestamp=generation_timestamp,
                    generation_timestamp_dt=datetime.fromtimestamp(generation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    tick_timestamp=tick_timestamp, # open candle timestamp
                    tick_timestamp_dt= datetime.fromtimestamp(tick_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    stake=tactic["stake"],            # amount in QUOTE
                    take_profit=tactic["take_profit"],
                    stop_loss=tactic.get("stop_loss", tactic.get("stoploss", -1)),
                    expiration_timestamp=expiration_timestamp,
                    expiration_timestamp_dt=datetime.fromtimestamp(expiration_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    close_order_expiration_ticks=tactic["close_order_expiration_ticks"],
                    close_order_expiration_timestamp=tick_timestamp + ((tactic["close_order_expiration_ticks"] + 1) * tick_seconds * 1000),
                    close_order_expiration_timestamp_dt=datetime.fromtimestamp((tick_timestamp + ((tactic["close_order_expiration_ticks"] + 1) * tick_seconds * 1000)) / 1000).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    internal_id=unique_id,
                    signal_id=f"ENTRY_SIGNAL_{unique_id}",
                    status="new",
                    entry_order=None,
                    close_order=None,
                    close_escape_order=None,
                    stage="entry_order"
                )

                print(f"[ENTRY_SIGNAL_EVENT] {entry_signal_event.tactic_name} | {entry_signal_event.entry_type.upper()}"
                      f" @ {entry_signal_event.price}, SIGNAL_ID: {entry_signal_event.signal_id} "
                      f" \ngeneration_timestamp_dt:  {entry_signal_event.generation_timestamp_dt} "
                      f" \ntick_timestamp_dt (open):  {entry_signal_event.tick_timestamp_dt} "
                      f" \nexpiration_timestamp_dt:  {entry_signal_event.expiration_timestamp_dt} "
                      f" \nclose_order_expiration_timestamp_dt:  {entry_signal_event.close_order_expiration_timestamp_dt} ")


                # ----------------------------
                # ADD ENTRY_SIGNAL TO SIGNAL LIST (MEMORY) - Send event to signal engine
                # ----------------------------

                signal_engine.add_signal(entry_signal_event)

                # ----------------------------
                # CREATE ENTRY_ORDER (ORDER EVENT)
                # ----------------------------
                # LIMIT BUY
                if entry_signal_event.status in ('new', 'crashed'):
                    try:
                        entry_order_event = order_manager.create_order(
                            side=entry_signal_event.side,
                            symbol=entry_signal_event.symbol,
                            order_type=entry_signal_event.entry_type,
                            amount=entry_signal_event.stake / current_price,
                            price=current_price , # - 10000, ############################### -3k for tests only,
                            expiration_timestamp=entry_signal_event.expiration_timestamp, # entry_order_expiration_timestamp
                            signal_id=entry_signal_event.signal_id,
                            stage="entry_order"
                            )
                        print(
                            f"[ENTRY_ORDER_EVENT] ORDER_ID: {entry_order_event.order_id}, "
                            f"STATUS: {entry_order_event.status}")
                        # SIGNAL: CHANGE SIGNAL STATUS TO ORDERED:
                        entry_signal_event.status = "ordered"

                        # SIGNAL: ADD ENTRY_ORDER PARAMS TO SIGNAL:
                        entry_signal_event.entry_order = entry_order_event

                    except Exception as e:
                        print(f"[WARN] Cannot create order from signal {entry_signal_event.signal_id}:", e)
                        entry_signal_event.status = "crashed"
                        time.sleep(5)
                        continue
                    print("\n==============================")
                    print("pobieranie orderow z gieldy")

            now_ms = int(time.time() * 1000)
            order_manager.check_open_orders(now_ms) # w tym elemencie zmianiane sa statusy orderow oraz np cancel remaining
            order_manager.save_orders_to_json("orders.json")
            signal_engine.sync_orders(order_manager) # SYNCHRONIZACJA ORDEROW DO SYGNALOW # SYNCHRONIZACJA ENTRY

            for o in order_manager.active_orders:
                print(f"ID: {o.order_id}, Status: {o.status}, Side: {o.side}, "
                      f"Symbol: {o.symbol}, Price: {o.price}, "
                      f"Amount: {o.amount}, Filled: {o.filled}, Remaining: {o.remaining}")

            # ----------------------------
            # CREATE REVERSE ORDER (ORDER EVENT)
            # ----------------------------

            # po order_manager.check_open_orders(now_ms)
            signal_engine.sync_orders(order_manager)

            # sprawdzamy statusy entry_order dla sygnałów
            for signal in signal_engine.pending_signals:
                order = signal.entry_order
                if not order:
                    continue

                 # ile tickow order ma przezyc?
                reverse_order_expiration_timestamp = signal.tick_timestamp + ((tactic["close_order_expiration_ticks"] + 1) * tick_seconds * 1000)
                # CASE 1: filled → wystaw reverse order
                if order.status == "filled" and signal.stage == "entry_order":
                    close_order = order_manager.create_order(
                        side=get_close_side(tactic["tactic_side"]),
                        symbol=signal.symbol,
                        order_type="limit",
                        amount=order.amount,
                        price=order.price * (1 + signal.take_profit),  # cena wyższa o TP
                        signal_id=signal.signal_id,
                        expiration_timestamp=signal.close_order_expiration_timestamp,
                        stage="close_order"
                    )
                    signal.close_order = close_order
                    signal.stage="close_order"
                    print(f"[CLOSE_ORDER] wystawiono  {order.side} dla {signal.signal_id}")

                # CASE 2: partial + expiration minął → wystaw reverse order
                elif order.status == "partial" and now_ms >= signal.expiration_timestamp and signal.stage == "entry_order":
                    close_order = order_manager.create_order(
                        side=get_close_side(tactic["tactic_side"]),
                        symbol=signal.symbol,
                        order_type="limit",
                        amount=order.filled,  # tylko kupiona część
                        price=order.price * (1 + signal.take_profit),
                        signal_id=signal.signal_id,
                        expiration_timestamp=signal.close_order_expiration_timestamp,
                        stage="close_order"
                    )
                    signal.close_order = close_order
                    signal.stage = "close_order"
                    print(f"[CLOSE_ORDER_PARTIAL] wystawiono  {order.side} dla {signal.signal_id}")

            # ----------------------------
            # CREATE REVERSE ESCAPE ORDER (ORDER EVENT) - # JESLI REVERSE W STATUSIE CANCELED - PUŚC MARKET
            # ----------------------------

            for signal in signal_engine.pending_signals:
                order = signal.close_order
                if not order:
                    continue

                # CASE : canceled → wystaw reverse order market
                if order.status == "canceled" and signal.stage == "close_order":
                    close_order = order_manager.create_order(
                        side=get_close_side(tactic["tactic_side"]),
                        symbol=signal.symbol,
                        order_type="market",
                        amount=order.amount,
                        price=None,  # market order has no price. price=order.price * (1 + signal.take_profit),
                        signal_id=signal.signal_id,
                        expiration_timestamp=signal.close_order_expiration_timestamp + 63_072_000, # plus 2 lata None
                        stage="close_escape_order"
                    )
                    signal.close_order = close_order
                    signal.stage = "close_escape_order"
                    print(f"[CLOSE_ESCAPE_ORDER] wystawiono {order.side} dla {signal.signal_id}")

            # ----------------------------
            # CHECK ALL
            # ----------------------------



            # ----------------------------
            # ARCHIVE -- ALL WHERE ENTRY_ORDER ARE CANCELED,
            # ----------------------------




        # Remove expired signals during runtime
        signal_engine.clear_expired_signals(server_ms)

        # Save updated list
        signal_engine.save_signals_to_json("signals.json")

        # Heartbeat
        write_heartbeat()
