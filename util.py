from datetime import timedelta
from currency_converter import CurrencyConverter

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
    "USD": "ZUSD"
}


def next_weekday(date):
    if date.weekday() > 4:
        day_gap = 7 - date.weekday()
        return date + timedelta(days=day_gap)
    else:
        return date


def fiatconvert(amount, cfrom, cto, date):
    return cc.convert(amount, cfrom, cto, date=next_weekday(date))


def canonical_symbol(symbol):
    return symbolmap.get(symbol, symbol)
