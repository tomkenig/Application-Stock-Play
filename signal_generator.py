class SignalGenerator:
    def __init__(self, indicator_functions):
        # Dictionary: {"get_indicator_rsi": function, "get_indicator_roc": function, ...}
        self.indicator_functions = indicator_functions

    def compute_indicators(self, close, tactic):
        """
        Computes all indicators required by the tactic (strategy).
        Strategy contains a list like:
            "indicator_functions": [
                ["get_indicator_rsi", 14],
                ["get_indicator_roc", 10]
            ]
        For each entry:
            func_name = "get_indicator_rsi"
            period = 14
        """
        indicator_values = {}

        # Loop through all indicator definitions in the tactic (strategy)
        for func_name, period in tactic["indicator_functions"]:
            func = self.indicator_functions[func_name]   # get actual function
            result = func(close, period)                 # compute indicator
            indicator_values.update(result)              # merge dict into indicator_values

        return indicator_values

    def check_tactic(self, indicator_values, tactic):
        """
        Checks up to 5 indicator conditions.
        Strategy contains fields like:
            indicator_1_name, indicator_1_value, indicator_1_operator
            indicator_2_name, ...
        If name == "none", the slot is ignored.
        """
        for i in range(1, 6):
            name = tactic[f"indicator_{i}_name"]
            value = tactic[f"indicator_{i}_value"]
            operator = tactic[f"indicator_{i}_operator"]

            # Skip empty indicator slots
            if name == "none":
                continue

            # Get indicator series (array)
            series = indicator_values[name]

            # Use the last value (current candle)
            current = float(series[-1])

            # If ANY condition fails → no signal
            if not self._check_condition(current, value, operator):
                return False

        # All conditions passed → signal
        return True

    def _check_condition(self, current, expected, operator):
        """
        Compares indicator value with expected value using operator.
        """
        if operator == "==": return current == expected
        if operator == "!=": return current != expected
        if operator == ">": return current > expected
        if operator == "<": return current < expected
        if operator == ">=": return current >= expected
        if operator == "<=": return current <= expected

        # If operator is unknown → configuration error
        raise ValueError(f"Unknown operator: {operator}")


class SignalEvent:
    """
    Simple container for a generated trading signal.
    main.py creates this object and passes it to TradingEngine.
    """
    def __init__(
        self,
        tactic_group_name,
        tactic_name,
        tactic_id,
        side,
        entry_type,
        price,
        generation_timestamp,
        tick_timestamp,
        stake,
        take_profit,
        stop_loss,
        wait_periods,
        expiration_time,
        internal_id,
        signal_id,
        status
    ):
        self.tactic_group_name = tactic_group_name
        self.tactic_name = tactic_name
        self.tactic_id = tactic_id
        self.side = side                        # buy / sell
        self.entry_type = entry_type            # market / limit
        self.price = price                      # current price

        # czas wygenerowania sygnału przez aplikację (ms, czas serwera giełdy)
        self.generation_timestamp = generation_timestamp

        # czas zamknięcia świecy giełdowej, na której powstał sygnał (ms)
        self.tick_timestamp = tick_timestamp

        self.stake = stake                      # amount in QUOTE currency
        self.take_profit = take_profit          # e.g. 0.025 = +2.5%
        self.stop_loss = stop_loss              # not used yet
        self.wait_periods = wait_periods        # number of candles to wait

        # docelowy timestamp wygaśnięcia sygnału (ms)
        self.expiration_time = expiration_time

        self.internal_id = internal_id          # internal identifier
        self.signal_id = signal_id              # signal identifier
        self.status = status                    # signal status
