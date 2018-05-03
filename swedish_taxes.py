from collections import defaultdict
from main import get_trades
from util import fiatconvert
from typing import NamedTuple

import load_data


class Table:
    def __init__(self, header, _format=None):
        if _format:
            self._format = {k: v for k, v in zip(header, _format)}
        else:
            self._format = {k: "{x}" for k in header}
        self._header = header
        self._cols = {k: [] for k in header}

    def __setitem__(self, key, value):
        assert not self._cols[key][-1]
        self._cols[key][-1] = value

    def new_row(self):
        for k in self._header:
            self._cols[k].append(None)

    def _width(self, key):
        return max(len(self._format[key].format(x=v)) for v in self._cols[key])

    def __getitem__(self, key):
        return [self._cols[k][key] for k in self._header]

    def __str__(self):
        widths = [max(self._width(k), len(k)) for k in self._header]
        str_header = " | ".join(("{{:<{}}}".format(l)).format(h.upper()) for h, l in zip(self._header, widths))
        delim = '-' * len(str_header)
        rows = [delim, str_header, delim]
        for row in self:
            rows.append(" | ".join(("{{:>{}}}".format(l)).format(self._format[self._header[i]].format(x=r)) for (i, r), l in zip(enumerate(row), widths)))
        return '\n'.join(rows)


class ProfitLoss(NamedTuple):
    profit: float
    loss: float

    def __add__(self, other):
        if other > 0:
            return ProfitLoss(self.profit + other, self.loss)
        return ProfitLoss(self.profit, self.loss + other)


def pptable(title, table):
    delim = f"{'-'*(len(title))}"
    print()
    print(delim)
    print(f"{title}")
    print(table)


def iscrypto(asset):
    return asset[0] == 'X'


def _canonical_trade(trade):
    """Takes a trade that could be either a buy or a sell and returns a
    simplified canonical format
    """
    pair = trade["pair"]
    cost = trade["cost"]
    vol = trade["vol"]
    if trade["type"] == "sell":
        pair = reversed(pair)
        cost = trade["vol"]
        vol = trade["cost"]
    return (*pair, cost, vol)


def swedish_taxes(trades, deposits):
    ftblfmt = "{x:8.4f}"
    asset_cost = defaultdict(int)
    asset_vol = defaultdict(int)
    asset_sold = defaultdict(lambda: defaultdict(int))
    asset_profit = defaultdict(lambda: defaultdict(int))
    profits = defaultdict(lambda: ProfitLoss(0, 0))
    for deposit in filter(lambda d: not iscrypto(d['asset']), deposits):
        curr = deposit['asset']
        amount = deposit['amount']
        cost = fiatconvert(amount, curr[1:], "SEK", deposit["time"])
        asset_cost[curr] += cost
        asset_vol[curr] += amount

    d_table = Table(["asset", "volume", "cost"])
    for asset in asset_vol:
        d_table.new_row()
        d_table['asset'] = asset
        d_table['volume'] = asset_vol[asset]
        d_table['cost'] = asset_cost[asset]
    pptable("Deposited", d_table)

    t_table = Table(['time', 'unknown warn', 'negative warn', 'profit', 'type', 'vol', 'cost', 'price', 'avg_price', 'pair'],
                    _format=["{x}", "{x}", "{x}", "{x:+10.2f} SEK", "{x:<4}", ftblfmt, ftblfmt, ftblfmt, "{x[0]:10.2f} {x[1]}", "({x[0]} -> {x[1]})"])
    for trade in trades:
        to, fro, cost, vol = _canonical_trade(trade)
        known = fro in asset_vol

        # Calculate cost in SEK
        cost_sek = trade["cost_sek"]

        asset_vol[to] += vol
        asset_cost[to] += cost_sek

        # Calculate average price in SEK
        avg_price = asset_cost[fro] / (asset_vol[fro] or 1)

        profit = cost_sek - cost * max([0, avg_price])
        year = trade["time"].year
        asset_vol[fro] -= cost
        asset_sold[year][fro] += cost
        asset_cost[fro] -= cost_sek
        asset_profit[year][fro] += profit

        if iscrypto(fro):
            profits[year] += profit

        negative = asset_vol[fro] < 0
        if negative:
            asset_vol[fro] = 0
            asset_cost[fro] = 0

        t_table.new_row()
        t_table['unknown warn'] = ' ' if known else '?'
        t_table['negative warn'] = '!' if negative else ' '
        t_table['time'] = trade['time'].replace(microsecond=0)
        t_table['profit'] = profit
        t_table['type'] = trade['type']
        t_table['vol'] = trade['vol']
        t_table['cost'] = trade['cost']
        t_table['price'] = trade['price']
        t_table['avg_price'] = (avg_price, fro)
        t_table['pair'] = (fro[1:], to[1:])
    pptable("Trades", t_table)

    profit_table = Table(['year', 'profit', 'loss'])
    for (year, profit) in profits.items():
        profit_table.new_row()
        profit_table['year'] = year
        profit_table['profit'] = f"{profit[0]:>20.20f}"
        profit_table['loss'] = f"{profit[1]:>20.20f}"
    pptable("Total profits", profit_table)

    for (year, ap) in asset_profit.items():
        asset_table = Table(['asset', 'profit', 'sold_vol', 'final_vol', 'total_cost', 'avg_price'],
                            _format=["{x}", "{x:+10.2f} SEK", ftblfmt, ftblfmt, ftblfmt, ftblfmt])
        for (asset, profit) in ap.items():
            asset_table.new_row()
            asset_table['asset'] = asset
            asset_table['profit'] = profit
            asset_table['sold_vol'] = asset_sold[year][asset]
            asset_table['final_vol'] = asset_vol[asset]
            asset_table['total_cost'] = asset_cost[asset]
            asset_table['avg_price'] = 0 if not asset_cost[asset] else asset_cost[asset] / asset_vol[asset]
        pptable(f"Profits {year}", asset_table)


if __name__ == "__main__":
    trades = get_trades()
    deposits = load_data.load_deposits()
    swedish_taxes(trades, deposits)
