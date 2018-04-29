from datetime import timedelta
from currency_converter import CurrencyConverter

cc = CurrencyConverter()


def next_weekday(date):
    if date.weekday() > 4:
        day_gap = 7 - date.weekday()
        return date + timedelta(days=day_gap)
    else:
        return date


def fiatconvert(amount, cfrom, cto, date):
    return cc.convert(amount, cfrom, cto, date=next_weekday(date))
