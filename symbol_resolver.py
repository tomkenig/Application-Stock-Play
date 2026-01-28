# symbol_resolver.py

def resolve_symbol(exchange_name, base, quote):
    """
    Zwraca symbol w formacie wymaganym przez giełdę.
    """
    exchange_name = exchange_name.lower()

    # KuCoin: BTC-USDT
    if exchange_name == "kucoin":
        return f"{base}-{quote}"

    # Binance: BTCUSDT
    if exchange_name == "binance":
        return f"{base}{quote}"

    # Bybit: BTCUSDT
    if exchange_name == "bybit":
        return f"{base}{quote}"

    # OKX: BTC-USDT
    if exchange_name == "okx":
        return f"{base}-{quote}"

    # Domyślnie: najpopularniejszy format
    return f"{base}{quote}"
