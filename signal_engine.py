import json
import time
import os
from datetime import datetime, timezone
from signal_generator import SignalEvent
from order_manager import OrderEvent   # zakładam że masz klasę OrderEvent


class SignalEngine:
    def __init__(self, expiration_ticks, tick_seconds):
        self.expiration_ticks = expiration_ticks
        self.tick_seconds = tick_seconds
        self.pending_signals = []

    def add_signal(self, event):
        for s in self.pending_signals:
            if s.tactic_id == event.tactic_id and s.tick_timestamp == event.tick_timestamp:
                print(f"[ENGINE] Duplicate signal skipped: tactic_id={event.tactic_id}, tick_timestamp={event.tick_timestamp}")
                return
        print(f"[ENGINE] Received signal from {event.tactic_name}")
        self.pending_signals.append(event)

    # ============================================================
    # 1) ZAPIS SYGNAŁÓW DO JSON
    # ============================================================
    def save_signals_to_json(self, filepath):
        data = []
        for s in self.pending_signals:
            entry_order = s.entry_order.__dict__ if hasattr(s.entry_order, "__dict__") else s.entry_order
            close_order = s.close_order.__dict__ if hasattr(s.close_order, "__dict__") else s.close_order
            close_escape_order = s.close_escape_order.__dict__ if hasattr(s.close_escape_order, "__dict__") else s.close_escape_order

            data.append({
                "tactic_group_name": s.tactic_group_name,
                "tactic_name": s.tactic_name,
                "tactic_id": s.tactic_id,
                "side": s.side,
                "symbol": s.symbol,
                "entry_type": s.entry_type,
                "price": s.price,
                "generation_timestamp": s.generation_timestamp,
                "generation_timestamp_dt": s.generation_timestamp_dt,
                "tick_timestamp": s.tick_timestamp,
                "tick_timestamp_dt": s.tick_timestamp_dt,
                "stake": s.stake,
                "take_profit": s.take_profit,
                "stop_loss": s.stop_loss,
                "wait_periods": s.wait_periods,
                "expiration_timestamp": s.expiration_timestamp,
                "expiration_timestamp_dt": s.expiration_timestamp_dt,
                "internal_id": s.internal_id,
                "signal_id": s.signal_id,
                "status": s.status,
                "entry_order": entry_order,
                "close_order": close_order,
                "close_escape_order": close_escape_order,
                "stage": s.stage,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ============================================================
    # 2) ODCZYT SYGNAŁÓW Z JSON
    # ============================================================
    def load_signals_from_json(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("[ENGINE] No saved signals found.")
            return

        self.pending_signals = []

        for item in data:
            # >>> TU dodaj odbudowę OrderEvent <<<
            from order_manager import OrderEvent

            entry_order = item.get("entry_order")
            if isinstance(entry_order, dict):
                try:
                    entry_order = OrderEvent(**entry_order)
                except Exception:
                    pass

            close_order = item.get("close_order")
            if isinstance(close_order, dict):
                try:
                    close_order = OrderEvent(**close_order)
                except Exception:
                    pass

            close_escape_order = item.get("close_escape_order")
            if isinstance(close_escape_order, dict):
                try:
                    close_escape_order = OrderEvent(**close_escape_order)
                except Exception:
                    pass

            # teraz tworzysz SignalEvent już z obiektami OrderEvent
            event = SignalEvent(
                tactic_group_name=item["tactic_group_name"],
                tactic_name=item["tactic_name"],
                tactic_id=item["tactic_id"],
                side=item["side"],
                symbol=item["symbol"],
                entry_type=item["entry_type"],
                price=item["price"],
                generation_timestamp=item["generation_timestamp"],
                generation_timestamp_dt=item["generation_timestamp_dt"],
                tick_timestamp=item["tick_timestamp"],
                tick_timestamp_dt=item["tick_timestamp_dt"],
                stake=item["stake"],
                take_profit=item["take_profit"],
                stop_loss=item["stop_loss"],
                wait_periods=item["wait_periods"],
                expiration_timestamp=item["expiration_timestamp"],
                expiration_timestamp_dt=item["expiration_timestamp_dt"],
                internal_id=item["internal_id"],
                signal_id=item["signal_id"],
                status=item["status"],
                entry_order=entry_order,
                close_order=close_order,
                close_escape_order=close_escape_order,
                stage=item["stage"],
            )
            self.pending_signals.append(event)

    # ============================================================
    # 3) USUWANIE SYGNAŁÓW, KTÓRE WYGASŁY
    # ============================================================
    def clear_expired_signals(self, current_timestamp):
        if current_timestamp < 1e12:
            current_timestamp = int(current_timestamp * 1000)

        before = len(self.pending_signals)
        expired_signals = [s for s in self.pending_signals if s.expiration_timestamp <= current_timestamp and s.status == "new"]
        self.pending_signals = [s for s in self.pending_signals if not (s.expiration_timestamp <= current_timestamp and s.status == "new")]

        removed = before - len(self.pending_signals)
        if removed > 0:
            print("\n==============================")
            print(f"[ENGINE] Removed {removed} expired signals (status NEW).")
            self._save_expired_signals_to_history(expired_signals, current_timestamp)

    def _save_expired_signals_to_history(self, expired_signals, current_timestamp):
        dt_utc = datetime.fromtimestamp(current_timestamp / 1000, tz=timezone.utc)
        date_str = dt_utc.strftime("%Y-%m-%d")
        filepath = f"signals_hist_{date_str}.json"

        data = []
        for s in expired_signals:
            entry_order = s.entry_order.__dict__ if hasattr(s.entry_order, "__dict__") else s.entry_order
            close_order = s.close_order.__dict__ if hasattr(s.close_order, "__dict__") else s.close_order
            close_escape_order = s.close_escape_order.__dict__ if hasattr(s.close_escape_order, "__dict__") else s.close_escape_order
            data.append({
                "tactic_group_name": s.tactic_group_name,
                "tactic_name": s.tactic_name,
                "tactic_id": s.tactic_id,
                "side": s.side,
                "symbol": s.symbol,
                "entry_type": s.entry_type,
                "price": s.price,
                "generation_timestamp": s.generation_timestamp,
                "generation_timestamp_dt": s.generation_timestamp_dt,
                "tick_timestamp": s.tick_timestamp,
                "tick_timestamp_dt": s.tick_timestamp_dt,
                "stake": s.stake,
                "take_profit": s.take_profit,
                "stop_loss": s.stop_loss,
                "wait_periods": s.wait_periods,
                "expiration_timestamp": s.expiration_timestamp,
                "expiration_timestamp_dt": s.expiration_timestamp_dt,
                "internal_id": s.internal_id,
                "signal_id": s.signal_id,
                "status": s.status,
                "entry_order": entry_order,
                "close_order": close_order,
                "close_escape_order":close_escape_order,
                "stage": s.stage
            })

        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                data = existing_data + data
            except Exception as e:
                print(f"[ENGINE] Error reading {filepath}: {e}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"[ENGINE] Saved {len(expired_signals)} expired signals to {filepath}")
        except Exception as e:
            print(f"[ENGINE] Error saving to {filepath}: {e}")


    # ============================================================
    # 4) SYNCHRONIZACJA ORDEROW DO SYGNALOW -------------------------ZMIENIC, BO NA RAZIE JEST TYLKO ENTRY
    # ============================================================
    def sync_orders(self, order_manager):
        for signal in self.pending_signals:
            if signal.signal_id:
                for order in order_manager.active_orders + order_manager.completed_orders:
                    if order.signal_id == signal.signal_id and order.stage == "entry_order":
                        signal.entry_order = order ###########################################  SYNCHRONIZACJA ENTRY
                    elif order.signal_id == signal.signal_id and order.stage == "close_order":
                        signal.close_order = order
                    elif order.signal_id == signal.signal_id and order.stage == "close_escape_order":
                        signal.close_escape_order = order




