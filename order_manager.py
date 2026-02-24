# order_manager.py
import json
import os
from datetime import datetime, timezone
from order_event import OrderEvent


class OrderManager:
    """
    Zarządza pełnym cyklem życia orderów:
    - wystawianie BUY/SELL (market/limit)
    - monitorowanie statusów
    - obsługa partial fill
    - anulowanie po expiration_timestamp
    - wystawianie SELL LIMIT po BUY
    - wystawianie SELL MARKET po częściowym TP
    """

    def __init__(self, exchange_client):
        self.exchange = exchange_client
        self.active_orders = []      # lista OrderEvent
        self.completed_orders = []   # historia (opcjonalnie)

    # ============================================================
    # ZAPIS / ODCZYT ORDERÓW DO JSON
    # ============================================================

    def save_orders_to_json(self, filepath="orders.json"):
        data = []
        for o in self.active_orders + self.completed_orders:
            data.append(o.__dict__)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_orders_from_json(self, filepath="orders.json"):
        if not os.path.exists(filepath):
            print("[ORDER_MANAGER] No saved orders found.")
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ORDER_MANAGER] Corrupted {filepath} ({e}), resetting file.")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([], f)
            return

        self.active_orders = []
        self.completed_orders = []
        for item in data:
            try:
                order = OrderEvent(**item)
                if order.status in ("open", "partial"):
                    self.active_orders.append(order)
                else:
                    self.completed_orders.append(order)
            except Exception as e:
                print(f"[ORDER_MANAGER] Error rebuilding order: {e}")

    # ============================================================
    # PUBLICZNE API
    # ============================================================

    def create_order(self, side, symbol, order_type, amount, price=None, signal_id=None, expiration_timestamp=None, stage=None):
        """
        Główna funkcja do wystawiania orderów.
        Przyjmuje parametry i kieruje do odpowiedniej metody prywatnej.
        """
        if order_type == "market":
            return self._create_market_order(side, symbol, amount, signal_id, expiration_timestamp, stage)

        if order_type == "limit":
            return self._create_limit_order(side, symbol, amount, price, signal_id, expiration_timestamp, stage)

        raise ValueError(f"Unknown order_type: {order_type}")

    def check_open_orders(self, current_timestamp):
        """
        Wywoływana cyklicznie w pętli głównej.
        Aktualizuje statusy orderów i obsługuje:
        - partial fill
        - expiration
        - pełne fill
        """
        for order in list(self.active_orders):
            status, filled, remaining = self._fetch_order_status(order)

            order.filled = filled
            order.remaining = remaining
            order.status = status

            if status == "filled":
                self._handle_filled(order)

            elif status == "partial":
                self._handle_partial(order, current_timestamp)

            elif current_timestamp >= order.expiration_timestamp:
                self._handle_expired(order)


    # ============================================================
    # TWORZENIE ORDERÓW (PRYWATNE)
    # ============================================================

    def _create_market_order(self, side, symbol, amount, signal_id, expiration_timestamp, stage):
        """
        Wystawia zlecenie MARKET przez ccxt.
        Zwraca OrderEvent.
        """
        result = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount
        )
        order_id = result["id"]

        event = OrderEvent(
            order_id=order_id,
            signal_id=signal_id,
            side=side,
            symbol=symbol,
            order_type="market",
            price=None,
            amount=amount,
            filled=0,
            remaining=amount,
            status="open",
            creation_timestamp=self._now_ms(),
            expiration_timestamp=expiration_timestamp,
            stage=stage
        )

        self.active_orders.append(event)
        return event

    def _create_limit_order(self, side, symbol, amount, price, signal_id, expiration_timestamp, stage):
        """
        Wystawia zlecenie LIMIT przez ccxt.
        Zwraca OrderEvent.
        """
        result = self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price
        )
        order_id = result["id"]

        event = OrderEvent(
            order_id=order_id,
            signal_id=signal_id,
            side=side,
            symbol=symbol,
            order_type="limit",
            price=price,
            amount=amount,
            filled=0,
            remaining=amount,
            status="open",
            creation_timestamp=self._now_ms(),
            expiration_timestamp=expiration_timestamp,
            stage=stage
        )

        self.active_orders.append(event)
        return event

    # ============================================================
    # OBSŁUGA STATUSÓW ORDERÓW
    # ============================================================

    def _handle_filled(self, order):
        """
        Order w pełni zrealizowany.
        BUY → wystaw SELL LIMIT (TP)
        SELL → koniec cyklu
        """
        if order.side == "buy":
            tp_price = self._calculate_tp_price(order)
            self.create_order(
                side="sell",
                symbol=order.symbol,
                order_type="limit",
                amount=order.amount,
                price=tp_price,
                signal_id=order.signal_id,
                expiration_timestamp=order.expiration_timestamp
            )

        self._finalize_order(order)


    def _handle_partial(self, order, current_timestamp):
        """
        BUY partial:
            - anuluj pozostałą część
            - wystaw SELL LIMIT na część kupioną

        SELL partial:
            - resztę sprzedaj MARKET
        """
        if order.side == "buy":
            self._cancel_remaining(order)

            tp_price = self._calculate_tp_price(order)
            self.create_order(
                side="sell",
                symbol=order.symbol,
                order_type="limit",
                amount=order.filled,
                price=tp_price,
                signal_id=order.signal_id,
                expiration_timestamp=order.expiration_timestamp
            )

        elif order.side == "sell":
            self._sell_remaining_market(order)

        self._finalize_order(order)


    def _handle_expired(self, order):
        """
        BUY expired → anuluj pozostałą część
        SELL expired → resztę sprzedaj MARKET
        """
        if order.side == "buy":
            self._cancel_remaining(order)

        elif order.side == "sell":
            self._sell_remaining_market(order)

        self._finalize_order(order)


    # ============================================================
    # AKCJE NA ORDERACH
    # ============================================================

    def _cancel_remaining(self, order):
        try:
            self.exchange.cancel_order(order.order_id, order.symbol)
            order.status = "canceled"
        except Exception as e:
            print(f"Error canceling order {order.order_id}: {e}")
            order.status = "error"

    def _sell_remaining_market(self, order):
        """
        Sprzedaje pozostałą część SELL MARKET.
        """
        remaining = order.remaining
        if remaining > 0:
            self.create_order(
                side="sell",
                symbol=order.symbol,
                order_type="market",
                amount=remaining,
                signal_id=order.signal_id,
                expiration_timestamp=order.expiration_timestamp
            )


    def _finalize_order(self, order):
        """
        Przenosi order z active_orders do completed_orders.
        """
        if order in self.active_orders:
            self.active_orders.remove(order)
        self.completed_orders.append(order)


    # ============================================================
    # POMOCNICZE
    # ============================================================

    def _fetch_order_status(self, order):
        """
        Pobiera status orderu z giełdy przez ccxt.
        Zwraca: (status, filled, remaining)
        """
        try:
            result = self.exchange.fetch_order(order.order_id, order.symbol)

            filled = result.get("filled", 0)
            remaining = result.get("remaining", order.amount)

            # mapowanie statusów
            ccxt_status = result.get("status", "open")
            if filled > 0 and remaining > 0:
                status = "partial"
            elif ccxt_status == "closed":
                status = "filled"
            elif ccxt_status == "canceled":
                status = "canceled"
            else:
                status = "open"

            return status, filled, remaining

        except Exception as e:
            print(f"Error fetching order {order.order_id}: {e}")
            return "error", order.filled, order.remaining

    def _calculate_tp_price(self, order):
        """
        Oblicza cenę take-profit na podstawie sygnału.
        """
        # placeholder
        return order.price * 1.02


    def _now_ms(self):
        return int(datetime.now(timezone.utc).timestamp() * 1000)
