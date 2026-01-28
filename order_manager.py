# order_manager.py

from datetime import datetime, timezone
from order_event import OrderEvent


class OrderManager:
    """
    Zarządza pełnym cyklem życia orderów:
    - wystawianie BUY/SELL (market/limit)
    - monitorowanie statusów
    - obsługa partial fill
    - anulowanie po expiration_time
    - wystawianie SELL LIMIT po BUY
    - wystawianie SELL MARKET po częściowym TP
    """

    def __init__(self, exchange_client):
        self.exchange = exchange_client
        self.active_orders = []      # lista OrderEvent
        self.completed_orders = []   # historia (opcjonalnie)


    # ============================================================
    # PUBLICZNE API
    # ============================================================

    def create_order(self, side, order_type, amount, price=None, signal=None, expiration_time=None):
        """
        Główna funkcja do wystawiania orderów.
        Przyjmuje parametry i kieruje do odpowiedniej metody prywatnej.
        """
        if order_type == "market":
            return self._create_market_order(side, amount, signal, expiration_time)

        if order_type == "limit":
            return self._create_limit_order(side, amount, price, signal, expiration_time)

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

            elif current_timestamp >= order.expiration_time:
                self._handle_expired(order)


    # ============================================================
    # TWORZENIE ORDERÓW (PRYWATNE)
    # ============================================================

    def _create_market_order(self, side, amount, signal, expiration_time):
        """
        Wystawia zlecenie MARKET.
        Zwraca OrderEvent.
        """
        # tu będzie ccxt.create_order(...)
        order_id = None  # placeholder

        event = OrderEvent(
            order_id=order_id,
            signal_id=signal,
            side=side,
            order_type="market",
            price=None,
            amount=amount,
            filled=0,
            remaining=amount,
            status="open",
            creation_timestamp=self._now_ms(),
            expiration_time=expiration_time
        )

        self.active_orders.append(event)
        return event


    def _create_limit_order(self, side, amount, price, signal, expiration_time):
        """
        Wystawia zlecenie LIMIT.
        Zwraca OrderEvent.
        """
        # tu będzie ccxt.create_order(...)
        order_id = None  # placeholder

        event = OrderEvent(
            order_id=order_id,
            signal_id=signal,
            side=side,
            order_type="limit",
            price=price,
            amount=amount,
            filled=0,
            remaining=amount,
            status="open",
            creation_timestamp=self._now_ms(),
            expiration_time=expiration_time
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
                order_type="limit",
                amount=order.amount,
                price=tp_price,
                signal=order.signal_id,
                expiration_time=order.expiration_time
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
                order_type="limit",
                amount=order.filled,
                price=tp_price,
                signal=order.signal_id,
                expiration_time=order.expiration_time
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
        """
        Anuluje pozostałą część BUY.
        """
        # ccxt.cancel_order(...)
        pass


    def _sell_remaining_market(self, order):
        """
        Sprzedaje pozostałą część SELL MARKET.
        """
        remaining = order.remaining
        if remaining > 0:
            self.create_order(
                side="sell",
                order_type="market",
                amount=remaining,
                signal=order.signal_id,
                expiration_time=order.expiration_time
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
        Pobiera status orderu z giełdy.
        Zwraca: (status, filled, remaining)
        """
        # tu będzie ccxt.fetch_order(...)
        return "open", 0, order.amount  # placeholder


    def _calculate_tp_price(self, order):
        """
        Oblicza cenę take-profit na podstawie sygnału.
        """
        # placeholder
        return order.price * 1.02


    def _now_ms(self):
        return int(datetime.now(timezone.utc).timestamp() * 1000)
