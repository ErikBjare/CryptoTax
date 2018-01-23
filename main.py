from typing import List, Dict, Any
from functools import reduce

import dateutil.parser

# import sqlite3
# connection = sqlite3.connect(':memory:')


class Table:
    def __init__(self, header, rows):
        self.header = header
        self.rows = rows


def _load_csv(filepath, delimiter=None) -> List[Dict[str, Any]]:
    import csv
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        return list(dict(zip(header, row)) for row in reader)


def _format_csv_from_kraken(trades_csv):
    "Format a CSV from a particular source into a canonical data format"
    for trade in trades_csv:
        trade["pair"] = (trade["pair"][:4], trade["pair"][4:])
        trade["time"] = dateutil.parser.parse(trade["time"])
        trade["price"] = float(trade["price"])
        trade["vol"] = float(trade["vol"])
        trade["cost"] = float(trade["cost"])
    return trades_csv


def _print_trades(trades, n=None):
    h = dict(zip(trades[0].keys(), trades[0].keys()))
    print(f"{h['time']:10}  {h['pair']:12.12}  {h['price']:12}  {h['vol']:9}  {h['cost']:12.10}")
    for d in (trades[:n] if n else trades):
        print(f"{d['time'].isoformat():.10}  {' / '.join(d['pair']):12}  {d['price']:12.6}  {d['vol']:9.6}  {d['cost']:12.6}")


def _sum_trades(t1, t2):
    # Price becomes volume-weighted average
    t1["price"] = (t1["price"] * t1["vol"] + t2["price"] * t2["vol"]) / (t1["vol"] + t2["vol"])
    # Volume becomes sum of trades
    t1["vol"] += t2["vol"]
    t1["cost"] += t2["cost"]
    return t1


def test_sum_trades():
    t1 = {"price": 1, "vol": 1, "cost": 1}
    t2 = {"price": 2, "vol": 1, "cost": 2}
    t3 = _sum_trades(t1, t2)
    assert t3["price"] == 1.5
    assert t3["vol"] == 2
    assert t3["cost"] == 3


def _reduce_trades(trades):
    """Merges consequtive trades in the same pair on the same day"""
    def r(processed, next):
        if len(processed) == 0:
            processed.append(next)
        else:
            last = processed[-1]
            if last["time"].date() == next["time"].date() and \
               last["pair"] == next["pair"]:
                processed[-1] = _sum_trades(last, next)
            else:
                processed.append(next)
        return processed
    return reduce(r, trades, [])


if __name__ == "__main__":
    trades_csv = _load_csv("data_private/kraken-trades.csv")
    trades = _format_csv_from_kraken(trades_csv)
    # print(trades[0].keys())
    trades = _reduce_trades(trades)
    _print_trades(trades)
