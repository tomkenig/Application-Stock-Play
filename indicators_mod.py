import numpy as np
import indicators as indc


def get_indicator_roc(close_np: np.ndarray, period: int):
    """
    Zwraca słownik:
        {"roc_100": np.ndarray([...])}
    gdzie '100' to dynamiczny period.
    """
    key = f"roc_{period}"
    return {key: indc.roc(close_np, period)}


def get_indicator_rsi(close_np: np.ndarray, period: int):
    """
    Zwraca słownik:
        {"rsi_50": np.ndarray([...])}
    gdzie '50' to dynamiczny period.
    """
    key = f"rsi_{period}"
    return {key: indc.rsi(close_np, period)}
