import json
import time
import os
from datetime import datetime, timezone
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
    # 3) USUWANIE SYGNAŁÓW, KTÓRE WYGASŁY. ARCHIWIZACJA
    # ============================================================
    def clear_expired_signals(self, current_timestamp):
        """
        Usuwa sygnały, których expiration_time < current_timestamp.
        Zapisuje usunięte sygnały do pliku signals_hist_[data_UTC].json
        expiration_time to unix_timestamp (ms).
        """
        # Normalize timestamp: accept seconds or milliseconds
        if current_timestamp < 1e12:
            # probably seconds -> convert to ms
            current_timestamp = int(current_timestamp * 1000)

        before = len(self.pending_signals)

        # Rozdziel na aktywne i wygasłe
        expired_signals = [
            s for s in self.pending_signals
            if s.expiration_time <= current_timestamp
        ]

        self.pending_signals = [
            s for s in self.pending_signals
            if s.expiration_time > current_timestamp
        ]

        after = len(self.pending_signals)
        removed = before - after

        if removed > 0:
            print(f"[ENGINE] Removed {removed} expired signals.")
            # Oznacz status w pamięci i zapisz wygasłe sygnały do pliku historii
            for s in expired_signals:
                s.status = "ARCHIVED"
            self._save_expired_signals_to_history(expired_signals, current_timestamp)

    def _save_expired_signals_to_history(self, expired_signals, current_timestamp):
        """
        Zapisuje wygasłe sygnały do pliku signals_hist_[data_UTC].json
        """
        # Konwertuj timestamp z ms na sekundy i utwórz datę UTC
        dt_utc = datetime.fromtimestamp(current_timestamp / 1000, tz=timezone.utc)
        date_str = dt_utc.strftime("%Y-%m-%d")
        
        filepath = f"signals_hist_{date_str}.json"

        # Przygotuj dane do zapisu
        data = []
        for s in expired_signals:
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

        # Jeśli plik już istnieje, dodaj do istniejącej listy
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                data = existing_data + data
            except Exception as e:
                print(f"[ENGINE] Error reading {filepath}: {e}")

        # Zapisz do pliku
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"[ENGINE] Saved {len(expired_signals)} expired signals to {filepath}")
        except Exception as e:
            print(f"[ENGINE] Error saving to {filepath}: {e}")