import csv
import pickle
from typing import List, Dict, Any
from datetime import date
import dateutil.parser
from pathlib import Path

from .util import canonical_symbol


def _load_csv(filepath, delimiter=",") -> List[Dict[str, Any]]:
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        header = next(reader)
        return list(dict(zip(header, row)) for row in reader)


def _load_price_csv(symbol):
    """Returns a dict mapping from date to price"""
    with open(f"data_public/prices-{symbol}.csv", "r") as csvfile:
        price_by_date = {}
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)  # discard header
        for row in reader:
            price_by_date[row[0]] = float(row[1])
        return price_by_date


def _load_price_csv2(symbol):
    """Returns a dict mapping from date to price"""
    history = _load_pricehistory(symbol)
    return {k: v["open"] for k, v in history.items()}


def _load_pricehistory(symbol) -> Dict[date, Dict[str, float]]:
    """Returns a dict mapping from date to price"""
    with open(f"tmp/{symbol}-pricehistory.pickle", "rb") as f:
        return pickle.load(f)


def _load_incoming_balances() -> List[Dict[str, Any]]:
    p = Path("data_private/balances-incoming.csv")
    if p.exists():
        data = _load_csv(p.absolute())
        return data
    else:
        return []


def _format_csv_from_kraken(trades_csv):
    "Format a CSV from a particular source into a canonical data format"
    for trade in trades_csv:
        # Kraken has really weird pair formatting...
        if "XXBT" not in trade["pair"] and "XBT" in trade["pair"]:
            trade["pair"] = (trade["pair"][:-3], trade["pair"][-3:])
        elif "ZEUR" not in trade["pair"] and "EUR" in trade["pair"]:
            trade["pair"] = (trade["pair"][:-3], trade["pair"][-3:])
        else:
            pairlen = int(len(trade["pair"]) / 2)
            trade["pair"] = (trade["pair"][:pairlen], trade["pair"][pairlen:])
        trade["pair"] = tuple(map(canonical_symbol, trade["pair"]))

        trade["time"] = dateutil.parser.parse(trade["time"])
        trade["price"] = float(trade["price"])
        trade["vol"] = float(trade["vol"])
        trade["cost"] = float(trade["cost"])

        del trade["txid"]
        del trade["ordertxid"]
    return trades_csv


def _format_csv_from_ethplorer(trades_csv):
    "Format a CSV from a particular source into a canonical data format"
    trades_list = []
    for trade in trades_csv:
        t = dict()
        if trade["toAddress"] == "0x7a250d5630b4cf539739df2c5dacb4c659f2488d" and trade["tokenSymbol"] == "ETH":
            t["pair"] = tuple(map(canonical_symbol, ("ETH", "XUNKNOWN")))
            t["time"] = dateutil.parser.parse(trade["date"])
            t["price"] = float(trade["usdPrice"].replace(',', '.'))
            t["vol"] = float(trade["value"].replace(',', '.'))
            t["cost"] = t["price"] * t["vol"]
            t["cost_usd"] = t["price"] * t["vol"]
            t["type"] = "sell"
            trades_list.append(t)

    return trades_list


def _format_csv_from_bitstamp(trades_csv):
    "Format a CSV from a particular source into a canonical data format"
    trades_csv = filter(lambda t: t["Type"] == "Market", trades_csv)
    trades_list = []

    for trade in trades_csv:
        ordertype = "market"
        time = dateutil.parser.parse(trade["Datetime"])
        vol = float(trade["Amount"].split(" ")[0])
        tradetype = trade["Sub Type"].lower()
        curr1 = trade["Amount"].split(" ")[1]
        curr2 = trade["Value"].split(" ")[1]
        pair = (canonical_symbol(curr1), canonical_symbol(curr2))
        price = float(trade["Rate"].split(" ")[0])
        cost = float(trade["Value"].split(" ")[0])
        fee = float(trade["Fee"].split(" ")[0])

        trades_list.append(
            {
                "ordertype": ordertype,
                "time": time,
                "vol": vol,
                "type": tradetype,
                "pair": pair,
                "price": price,
                "cost": cost,
                "fee": fee,
                "margin": 0.0,
                "misc": "",
                "ledgers": None,
            }
        )
    return trades_list


def _format_csv_from_lbtc(trades_csv):
    trades_csv = [t for t in trades_csv if t[" TXtype"] == "Trade"]
    curr1 = "XXBT"
    curr2 = "ZUSD"
    price_history = _load_pricehistory("XXBT")
    trades_list = []
    for trade in trades_csv:
        time = dateutil.parser.parse(trade[" Created"]).replace(tzinfo=None)

        if trade[" Received"] != "":
            tradetype = "buy"
            vol = float(trade[" Received"])
        else:
            tradetype = "sell"
            vol = float(trade[" Sent"])

        price = price_history[time.strftime("%Y-%m-%d")]["high"]
        cost = price * vol

        pair = (curr1, curr2)

        trades_list.append(
            {
                "ordertype": "market",
                "time": time,
                "vol": vol,
                "type": tradetype,
                "pair": pair,
                "price": price,
                "cost": cost,
                "fee": None,
                "margin": None,
                "misc": "",
                "ledgers": None,
            }
        )
    return trades_list


def _format_csv_from_poloniex(trades_csv):
    "Format a CSV from a particular source into a canonical data format"
    for trade in trades_csv:
        trade["pair"] = trade.pop("Market").split("/")
        trade["pair"] = tuple(map(canonical_symbol, trade["pair"]))
        trade["type"] = trade.pop("Type").lower()

        trade["time"] = dateutil.parser.parse(trade.pop("Date"))
        trade["price"] = float(trade.pop("Price"))
        trade["vol"] = float(trade.pop("Amount"))
        trade["cost"] = float(trade.pop("Total"))

        del trade["Category"]
        del trade["Order Number"]
        del trade["Fee"]
        del trade["Base Total Less Fee"]
        del trade["Quote Total Less Fee"]
    return trades_csv


def _format_csv_from_generic(trades_csv):
    """Formats csv files in the format in examples/generic-trades.csv to trades."""
    for trade in trades_csv:
        trade["pair"] = trade["pair"].split("/")
        trade["pair"] = tuple(map(canonical_symbol, trade["pair"]))
        trade["time"] = dateutil.parser.parse(trade["time"])
        trade["cost"] = float(trade["cost"])
        trade["vol"] = float(trade["vol"])
        trade["price"] = trade["cost"] / trade["vol"]
    return trades_csv



def load_all_trades():
    """Loads all trades from the .csv files exported from exchanges. Currently supports Kraken and Poloniex formatted .csv files.
    """
    trades = []

    kraken_trades_filename = "data_private/kraken-trades.csv"
    if Path(kraken_trades_filename).exists():
        print("Found kraken trades!")
        trades_kraken_csv = _load_csv(kraken_trades_filename)
        trades.extend(_format_csv_from_kraken(trades_kraken_csv))

    polo_trades_filename = "data_private/poloniex-trades.csv"
    if Path(polo_trades_filename).exists():
        print("Found poloniex trades!")
        trades_polo_csv = _load_csv(polo_trades_filename)
        trades.extend(_format_csv_from_poloniex(trades_polo_csv))

    bitstamp_trades_filename = "data_private/bitstamp-trades.csv"
    if Path(bitstamp_trades_filename).exists():
        print("Found bitstamp trades!")
        trades_bitstamp_csv = _load_csv(bitstamp_trades_filename)
        trades.extend(_format_csv_from_bitstamp(trades_bitstamp_csv))

    lbtc_trades_filename = "data_private/lbtc-trades.csv"
    if Path(lbtc_trades_filename).exists():
        print("Found lbtc trades!")
        trades_lbtc_csv = _load_csv(lbtc_trades_filename)
        trades.extend(_format_csv_from_lbtc(trades_lbtc_csv))

    ethplorer_trades_filename = "data_private/ethplorer-trades.csv"
    if Path(ethplorer_trades_filename).exists():
        print("Found ethplorer trades!")
        trades_ethplorer_csv = _load_csv(ethplorer_trades_filename, delimiter=";")
        trades.extend(_format_csv_from_ethplorer(trades_ethplorer_csv))

    generic_trades_filename = "data_private/generic-trades.csv"
    if Path(generic_trades_filename).exists():
        print("Found generic trades!")
        trades_csv = _load_csv(generic_trades_filename)
        trades_csv = _format_csv_from_generic(trades_csv)
        trades.extend(trades_csv)

    return list(sorted(trades, key=lambda t: t["time"]))


def _format_deposits_kraken(ledger):
    deposits = filter(lambda x: x["type"] == "deposit", ledger)
    for deposit in deposits:
        deposit["time"] = dateutil.parser.parse(deposit["time"])
        deposit["amount"] = float(deposit["amount"])
        yield deposit


def _format_deposits_bitstamp(trades):
    deposits = filter(lambda x: "Deposit" in x["Type"], trades)
    for deposit in deposits:
        d = dict()
        d["time"] = dateutil.parser.parse(deposit["Datetime"])
        d["amount"] = float(deposit["Amount"].split(" ")[0])
        d["asset"] = canonical_symbol(deposit["Amount"].split(" ")[1])
        yield d


def load_deposits():
    deposits = []
    kraken_ledger_filename = "data_private/kraken-ledgers.csv"
    if Path(kraken_ledger_filename).exists():
        print("Found kraken ledgers!")
        ledger_kraken_csv = _load_csv(kraken_ledger_filename)
        deposits_kraken = _format_deposits_kraken(ledger_kraken_csv)
        deposits.extend(deposits_kraken)

    bitstamp_trades_filename = "data_private/bitstamp-trades.csv"
    if Path(bitstamp_trades_filename).exists():
        print("Found bitstamp deposits!")
        trades_bitstamp_csv = _load_csv(bitstamp_trades_filename)
        deposits_bitstamp = _format_deposits_bitstamp(trades_bitstamp_csv)
        deposits.extend(deposits_bitstamp)

    if not deposits:
        raise Exception("No deposits found, please import ledger data")

    return deposits
