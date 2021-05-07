from datetime import datetime, timedelta
import currency_converter
from currency_converter import CurrencyConverter
import logging

logger = logging.getLogger(__name__)

cc = CurrencyConverter()
symbolmap = {
    "XXBTC": "XXBT",
    "XBT": "XXBT",
    "XXDG": "XXDG",
    "ETH": "XETH",
    "BTC": "XXBT",
    "BCH": "XBCH",
    "GNO": "XGNO",
    "EOS": "XEOS",
    "STR": "XXLM",
    "SC": "XXSC",
    "EUR": "ZEUR",
    "USD": "ZUSD",
    "SEK": "ZSEK"
}


def next_weekday(date):
    if date.weekday() > 4:
        day_gap = 7 - date.weekday()
        return date + timedelta(days=day_gap)
    else:
        return date


def fiatconvert(amount: float, cfrom: str, cto: str, date: datetime, fallback=True):
    try:
        if fallback:
            cc = CurrencyConverter(fallback_on_wrong_date=fallback)
        return cc.convert(amount, cfrom, cto, date=next_weekday(date))
    except currency_converter.RateNotFoundError as e:
        logger.warn("Conversionrate not found, using fallback:", e)
        return fiatconvert(amount, cfrom, cto, date=date, fallback=True)


def canonical_symbol(symbol):
    return symbolmap.get(symbol, symbol)
