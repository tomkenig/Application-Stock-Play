# order_event.py

class OrderEvent:
    """
    Reprezentuje pojedynczy order giełdowy.
    Order ma własny cykl życia, niezależny od sygnału.
    """

    def __init__(
        self,
        order_id,
        signal_id,
        side,               # "buy" / "sell"
        symbol,
        order_type,         # "market" / "limit"
        price,
        amount,
        filled,
        remaining,
        status,             # "new" / "open" / "partial" / "filled" / "canceled" / "error"
        creation_timestamp,
        expiration_time
    ):
        # ID nadawane przez giełdę (ccxt zwróci)
        self.order_id = order_id

        # ID sygnału, z którego order powstał
        self.signal_id = signal_id

        # BUY lub SELL
        self.side = side

        # MARKET lub LIMIT
        self.order_type = order_type

        # symbol
        self.symbol = symbol

        # Cena (dla LIMIT), None dla MARKET
        self.price = price

        # Ilość zlecona
        self.amount = amount

        # Ile zostało zrealizowane
        self.filled = filled

        # Ile pozostało do realizacji
        self.remaining = remaining

        # Status orderu
        self.status = status

        # Timestamp utworzenia orderu (ms)
        self.creation_timestamp = creation_timestamp

        # Timestamp wygaśnięcia orderu (ms)
        self.expiration_time = expiration_time


    def is_active(self):
        """
        Order jest aktywny, jeśli nie jest zakończony.
        """
        return self.status in ("new", "open", "partial")


    def is_finished(self):
        """
        Order jest zakończony (filled, canceled, error).
        """
        return self.status in ("filled", "canceled", "error")
