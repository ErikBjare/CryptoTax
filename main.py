from typing import List, Dict, Any

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
    return trades_csv


def _print_trades(trades):
    h = dict(zip(trades[0].keys(), trades[0].keys()))
    print(f"{h['time']:10.10}  {h['pair']:12.12}  {h['price']:10.10}  {h['cost']:10.10}")
    for d in trades[:3]:
        print(f"{d['time']:10.10}  {' / '.join(d['pair']):12.12}  {d['price']:10.10}  {d['cost']:10.10}")


if __name__ == "__main__":
    trades_csv = _load_csv("data_private/kraken-trades.csv")
    trades = _format_csv_from_kraken(trades_csv)
    _print_trades(trades)
