import json
import time
from signal_generator import SignalEvent


class TradingEngine:
    def __init__(self, expiration_ticks, tick_seconds):
        self.expiration_ticks = expiration_ticks
        self.tick_seconds = tick_seconds
        # List of signals waiting for processing
        # (for now we only store them, no trading logic yet)
        self.pending_signals = []

    def add_signal(self, event):
        # Called when main.py detects a valid strategy signal
        print(f"[ENGINE] Received signal from {event.tactic_name}")

        # Store the signal for later processing
        self.pending_signals.append(event)

    # ============================================================
    # 1) ZAPIS SYGNAŁÓW DO JSON
    # ============================================================
    def save_signals_to_json(self, filepath):
        data = []

        for s in self.pending_signals:
            data.append({
                "tactic_group_name": s.tactic_group_name,
                "tactic_name": s.tactic_name,
                "tactic_id": s.tactic_id,
                "side": s.side,
                "entry_type": s.entry_type,
                "price": s.price,

                "generation_timestamp": s.generation_timestamp,
                "tick_timestamp": s.tick_timestamp,
                "stake": s.stake,
                "take_profit": s.take_profit,
                "stop_loss": s.stop_loss,
                "wait_periods": s.wait_periods,
                "expiration_time": s.expiration_time,

                "internal_id": s.internal_id,
                "signal_id": s.signal_id,
                "status": s.status,
            })

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # ============================================================
    # 2) ODCZYT SYGNAŁÓW Z JSON (np. po crashu)
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
            event = SignalEvent(
                tactic_group_name=item["tactic_group_name"],
                tactic_name=item["tactic_name"],
                tactic_id=item["tactic_id"],
                side=item["side"],
                entry_type=item["entry_type"],
                price=item["price"],
                generation_timestamp=item["generation_timestamp"],
                tick_timestamp=item["tick_timestamp"],
                stake=item["stake"],
                take_profit=item["take_profit"],
                stop_loss=item["stop_loss"],
                wait_periods=item["wait_periods"],
                expiration_time=item["expiration_time"],
                internal_id=item["internal_id"],
                signal_id=item["signal_id"],
                status=item["status"],
            )
            self.pending_signals.append(event)

    # ============================================================
    # 3) USUWANIE SYGNAŁÓW, KTÓRE WYGASŁY
    # ============================================================
    def clear_expired_signals(self, current_timestamp):
        """
        Usuwa sygnały, których expiration_time < current_timestamp.
        expiration_time to unix_timestamp (ms).
        """
        before = len(self.pending_signals)

        self.pending_signals = [
            s for s in self.pending_signals
            if s.expiration_time > current_timestamp
        ]

        after = len(self.pending_signals)
        removed = before - after

        if removed > 0:
            print(f"[ENGINE] Removed {removed} expired signals.")
