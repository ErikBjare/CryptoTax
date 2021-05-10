from typing import List, Dict, Any
from functools import reduce
from itertools import groupby
from collections import defaultdict
from datetime import datetime
from copy import copy, deepcopy
from math import isclose
import logging

import click
from tabulate import tabulate
from tqdm import tqdm

from .util import fiatconvert
from . import load_data
from .download_data import get_price, symbol2currencymap

logger = logging.getLogger(__name__)

Trade = Dict[str, Any]


def _print_trades(trades: List[Trade], n=None):
    print(
        tabulate(
            [
                [
                    t["time"].date(),
                    _pair_fmt(t["pair"]),
                    t["type"],
                    t["price"],
                    t["vol"],
                    t["cost"],
                    t["cost_usd"],
                ]
                for t in trades
            ],
            headers=["time", "pair", "type", "price", "vol", "cost", "cost_usd"],
        )
    )


def _sum_trades(t1: Trade, t2: Trade):
    assert t1["type"] == t2["type"]
    assert t1["pair"] == t2["pair"]
    # Price becomes volume-weighted average
    t1["price"] = (t1["price"] * t1["vol"] + t2["price"] * t2["vol"]) / (
        t1["vol"] + t2["vol"]
    )
    # Volume becomes sum of trades
    t1["vol"] += t2["vol"]
    t1["cost"] += t2["cost"]
    if "cost_usd" in t1:
        t1["cost_usd"] += t2["cost_usd"]
    return t1


def test_sum_trades():
    t1 = {"price": 1, "vol": 1, "cost": 1, "type": "buy", "pair": ("USD", "BTC")}
    t2 = {"price": 2, "vol": 1, "cost": 2, "type": "buy", "pair": ("USD", "BTC")}
    t3 = _sum_trades(t1, t2)
    assert t3["price"] == 1.5
    assert t3["vol"] == 2
    assert t3["cost"] == 3


def _reduce_trades(trades: List[Trade]):
    """Merges consequtive trades in the same pair on the same day"""

    def r(processed, next):
        if len(processed) == 0:
            processed.append(next)
        else:
            last = processed[-1]
            if (
                last["time"].date() == next["time"].date()
                and last["pair"] == next["pair"]
                and last["type"] == next["type"]
            ):
                processed[-1] = _sum_trades(last, next)
            else:
                processed.append(next)
        return processed

    return reduce(r, trades, [])


def _pair_fmt(pair):
    return f"{pair[0].ljust(4)}/{pair[1].rjust(4)}"


def _calc_cost_usd(trades: List[Trade]):
    logger.info("Processing trades...")
    for trade in tqdm(trades):
        date = trade["time"].date()
        val_cur = trade["pair"][1]
        if val_cur == "ZEUR":
            # Buy/sell something valued in EUR
            trade["cost_usd"] = fiatconvert(trade["cost"], "EUR", "USD", trade["time"])

        elif val_cur == "ZSEK":
            # Buy/sell something valued in EUR
            trade["cost_usd"] = fiatconvert(trade["cost"], "SEK", "USD", trade["time"])

        elif val_cur == "ZUSD":
            trade["cost_usd"] = trade["cost"]

        elif val_cur in symbol2currencymap.keys():
            price = get_price(symbol2currencymap[val_cur], date)
            trade["cost_usd"] = trade["cost"] * price

        elif val_cur == "XUNKNOWN":
            pass

        else:
            print(
                f"Could not calculate USD cost for pair: {trade['pair']}, add support for {trade['pair'][1]} by adding a mapping from symbol to currency in currency2symbolmap."
            )
            trade["cost_usd"] = None
            continue

        trade["cost_sek"] = fiatconvert(trade["cost_usd"], "USD", "SEK", trade["time"])
    return trades


def _cost_basis_per_asset(trades: List[Trade]) -> None:
    costbasis = defaultdict(float)  # type: Dict[str, float]
    vol = defaultdict(float)  # type: Dict[str, float]

    incoming = load_data._load_incoming_balances()
    if incoming:
        print("WARNING: Loaded incoming balances, setting cost to zero.")
        for r in incoming:
            costbasis[r["asset"]] += 0
            vol[r["asset"]] += float(r["amount"])

    for trade in trades:
        if trade["type"] == "buy" and trade["cost_usd"]:
            costbasis[trade["pair"][0]] += trade["cost_usd"]
            vol[trade["pair"][0]] += trade["vol"]

    print("")
    print(
        tabulate(
            [
                (
                    asset,
                    round(costbasis[asset]),
                    round(vol[asset], 3),
                    round(costbasis[asset] / vol[asset], 3),
                )
                for asset in costbasis
            ],
            headers=["asset", "costbasis", "vol", "cost/vol"],
        )
    )


def _filter_trades_by_time(trades: List[Trade], year: int) -> List[Trade]:
    return list(
        filter(
            lambda t: datetime(year, 1, 1) <= t["time"] < datetime(year + 1, 1, 1),
            trades,
        )
    )


def test_filter_trades_by_time():
    _t1 = {"time": datetime(2017, 12, 30, 23, 42)}
    _t2 = {"time": datetime(2018, 1, 1, 1, 42)}
    assert 1 == len(_filter_trades_by_time([_t1, _t2], 2017))
    assert 1 == len(_filter_trades_by_time([_t1, _t2], 2018))


def _flip_pair(t: Trade) -> Trade:
    t = copy(t)
    assert t["type"] in ["buy", "sell"]
    t["type"] = "buy" if t["type"] == "sell" else "sell"
    t["pair"] = tuple(reversed(t["pair"]))
    t["price"] = 1 / t["price"]
    t["vol"], t["cost"] = t["cost"], t["vol"]
    return t


def _normalize_trade_type(t: Trade) -> Trade:
    """
    Normalizes sell-trade t into a buy-trade by flipping the pair
    such that asset1 is always being bought.
    """
    t = copy(t)
    assert isclose(t["vol"], t["cost"] / t["price"], rel_tol=1e-4)
    if t["type"] == "sell":
        t = _flip_pair(t)
    assert isclose(t["vol"], t["cost"] / t["price"], rel_tol=1e-4)
    return t


def test_normalize_trade_type():
    t1 = {"pair": ("XETH", "ZEUR"), "type": "sell", "vol": 10, "price": 2, "cost": 20}
    t1norm = _normalize_trade_type(t1)
    assert t1norm["pair"] == tuple(reversed(t1["pair"]))
    assert t1norm["vol"] == 20
    assert t1norm["price"] == 0.5
    assert t1norm["cost"] == 10

    t2 = {"pair": ("XXBT", "XETH"), "type": "sell", "vol": 8, "price": 0.25, "cost": 2}
    t2norm = _normalize_trade_type(t2)
    assert t2norm["pair"] == tuple(reversed(t2["pair"]))
    assert t2norm["price"] == 4
    assert t2norm["cost"] == 8


def _calculate_delta(trades: List[Trade]) -> Dict[str, Dict[str, float]]:
    d = defaultdict(lambda: defaultdict(float))  # type: Dict[str, Dict[str, float]]
    for t in trades:
        t = _normalize_trade_type(t)
        d[t["pair"][0]]["balance"] += t["vol"]
        d[t["pair"][1]]["balance"] -= t["cost"]

        d[t["pair"][0]]["cost_usd"] += t["cost_usd"]
        d[t["pair"][1]]["cost_usd"] -= t["cost_usd"]

        d[t["pair"][0]]["cost_sek"] += t["cost_sek"]
        d[t["pair"][1]]["cost_sek"] -= t["cost_sek"]

    return d


def _calculate_inout_balances(
    trades: List[dict], balances: Dict[str, int] = None
) -> Dict[str, int]:
    if not balances:
        balances = defaultdict(lambda: 0)
    for t in trades:
        t = _normalize_trade_type(t)
        balances[t["pair"][0]] += t["vol"]
        balances[t["pair"][1]] -= t["cost"]

    return balances


def _aggregate_trades(trades: List[Trade]) -> List[Trade]:
    def keyfunc(t):
        return tuple(list(t["pair"]) + [t["type"]])

    trades = deepcopy(trades)
    grouped = groupby(sorted(trades, key=keyfunc), key=keyfunc)

    agg_trades = []
    for k, v in grouped:
        t = reduce(_sum_trades, v)
        del t["time"]
        agg_trades.append(t)

    return list(sorted(agg_trades, key=lambda t: t["pair"]))


def _print_balances(trades: List[Trade], year: int = None) -> None:
    if year:
        trades = _filter_trades_by_time(trades, year)
    print(f"\n# Balance diff {f'for {year}' if year else ''}")
    delta = _calculate_delta(trades)
    print(
        tabulate(
            [[k, d["balance"], d["cost_usd"], d["cost_sek"]] for k, d in delta.items()],
            headers=["asset", "∆balance", "cost_usd", "cost_sek"],
        )
    )


def _print_agg_trades(trades: List[Trade], year: int = None) -> None:
    if year:
        trades = _filter_trades_by_time(trades, year)
    print(f"\n# Aggregate trades {f'for {year}' if year else ''}")
    print(
        tabulate(
            [
                [
                    _pair_fmt(t["pair"]),
                    t["type"],
                    t["vol"],
                    t["cost"],
                    t["cost_usd"],
                    t["cost_usd"] / t["vol"],
                ]
                for t in trades
            ],
            headers=["pair", "type", "vol", "cost", "cost_usd", "cost_usd/unit"],
        )
    )


def get_trades() -> List[Trade]:
    trades = load_data.load_all_trades()
    trades = _reduce_trades(trades)
    trades = _calc_cost_usd(trades)
    return trades


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("test")


@main.command()
def trades() -> None:
    """Prints all trades"""
    trades = get_trades()
    _print_trades(trades)


@main.command()
def cost_basis() -> None:
    """Computed the cost-basis per asset"""
    trades = get_trades()
    _print_trades(trades)

    print("\n# Cost basis per asset")
    _cost_basis_per_asset(trades)


@main.command()
@click.argument("year", type=int)
def year(year: int) -> None:
    """Prints stats per year"""
    trades = get_trades()
    trades_for_year = _filter_trades_by_time(trades, year)
    _print_balances(trades_for_year, year)
    _print_agg_trades(trades_for_year, year)


@main.command()
def swedish_taxes() -> None:
    """Prints swedish tax summary"""
    from .swedish_taxes import main as swedish_taxes_main

    swedish_taxes_main()


@main.command()
def avanza() -> None:
    """Runs avanza code"""
    from .avanza import main as avanza_main

    avanza_main()


if __name__ == "__main__":
    main()
