from collections import defaultdict
from typing import Dict, Tuple, List, Any
from copy import deepcopy

from tabulate import tabulate
import pandas as pd
import pydash
import matplotlib.pyplot as plt

from .load_data import _load_csv

_typ_to_type = {
    "Insättning": "deposit",
    "Uttag": "withdrawal",
    "Köp": "buy",
    "Köp, rättelse": "buy",
    "Sälj": "sell",
    "Sälj, rättelse": "sell",
    "Räntor": "interest",
    "Överföring från Nordea": "buy",
    "Övf fr Nordea": "buy",
    "Utdelning": "dividend",
}


isin_registry = {}


def _load_data_avanza(filename="./data_private/avanza-transactions_utf8.csv") -> List[Dict]:
    data = _load_csv(filename, delimiter=";")
    for row in data:
        row["date"] = row.pop("Datum")
        ttype = row.pop("Typ av transaktion")
        if ttype in _typ_to_type:
            row["type"] = _typ_to_type[ttype]
        else:
            # print(type)
            row["type"] = ttype

        row["asset_name"] = row.pop("Värdepapper/beskrivning")
        row["asset_id"] = row.pop('ISIN')
        isin_registry[row["asset_id"]] = row["asset_name"]

        volume = row.pop("Antal")
        row["volume"] = abs(float(volume.replace(",", "."))) if volume != "-" else None

        price = row.pop("Kurs")
        row["price"] = float(price.replace(",", ".")) if price != "-" else None

        amount = row.pop("Belopp")
        row["amount"] = float(amount.replace(",", ".")) if amount != "-" else None

        row['currency'] = row.pop("Valuta")

        row.pop("Konto")
        row.pop("Courtage")

    for tx in data:
        if tx['type'] == "MERGER MED TESLA 1 PÅ 0,11":
            tx['type'] = 'sell'
            tx['price'] = 20.75
            tx['amount'] = tx['price'] * tx['volume']
        elif tx['type'] == "MERGER MED SOLARCITY 0,11 PÅ 1":
            tx['type'] = 'buy'
            tx['price'] = 20.75 / 0.11
            tx['amount'] = tx['price'] * tx['volume']

    #print(data[0].keys())
    return data


def compute_balances(trades) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    balances: Dict[str, float] = defaultdict(lambda: 0)
    costavg: Dict[str, float] = defaultdict(lambda: 0)
    profit: Dict[str, float] = defaultdict(lambda: 0)

    for t in sorted(trades, key=lambda t: t['date']):
        if t["type"] == "buy":
            prev_cost = balances[t['asset_id']] * costavg[t['asset_id']]
            balances[t['asset_id']] += t['volume']
            if balances[t['asset_id']] == 0:
                print(f"new balance was zero, something is wrong: \n - {t}\n - {t['volume']}\n - {balances[t['asset_id']]}")
            elif t['price']:
                new_cost = t['volume'] * t['price']
                costavg[t['asset_id']] = (prev_cost + new_cost) / balances[t['asset_id']]
            else:
                # Special case where stock was transfered without price attached
                pass
        elif t["type"] == "dividend":
            profit[t["asset_id"]] += t["amount"]
        elif t["type"] == "sell":
            #print(costavg[t['asset']])
            p = (t["price"] - costavg[t["asset_id"]]) * t["volume"]
            profit[t["asset_id"]] += p
            #print(f"Profited: {p}")
            balances[t['asset_id']] -= t['volume']
    return balances, costavg, profit


def print_table(table) -> None:
    from IPython.display import HTML, display
    from tabulate import tabulate
    display(HTML(tabulate(table, tablefmt='html')))


def display_performance(balances, costavg, profit):
    header = ["Asset", "Balance", "Profit", "Costavg"]
    table = []
    for asset in balances:
        table.append([
            asset,
            round(balances[asset], 2),
            round(profit[asset], 2),
            round(costavg[asset], 2)
        ])

    print_table([header] + sorted(table, key=lambda r: -r[2]))
    return table


def plot_holdings(txs: List[Dict]) -> None:
    for group, group_txs in pydash.group_by(txs, lambda tx: tx["asset_id"]).items():
        group_txs = [tx for tx in group_txs if tx["volume"] and tx["type"] in ["buy", "sell"]]
        if not group_txs:
            continue
        df = asset_txs_to_dataframe(group_txs)

        if round(df.iloc[-1]['balance']) != 0:
            plt.plot(df['holdings'], label=group_txs[0]['asset_name'])
            # print(df)

    plt.title("Value of holdings")
    plt.ylabel("SEK")
    plt.ylim(0)
    plt.xlim(pd.Timestamp("2016-01-01"), pd.Timestamp.now())
    plt.legend()
    plt.show()


def txs_to_dataframe(txs: List[Dict]) -> pd.DataFrame:
    index = [pd.Timestamp(tx['date']) for tx in txs]
    txs = deepcopy(txs)
    for tx in txs:
        tx.pop('date')
        tx['volume'] *= 1 if tx['type'] == "buy" else -1
    df = pd.DataFrame(txs, index=index)
    df.sort_index(inplace=True)
    return df


def asset_txs_to_dataframe(txs: List[Dict]) -> pd.DataFrame:
    df = txs_to_dataframe(txs)
    df['profit'] = 0.0
    df['costavg'] = df['price']
    df['balance'] = df['volume']
    for i, d in list(enumerate(df.index))[1:]:
        df.iloc[i, df.columns.get_loc('balance')] += df.iloc[i - 1]['balance']

        prev_cost = df.iloc[i - 1]['balance'] * df.iloc[i - 1]['costavg']
        if df.iloc[i]['type'] == "buy":
            new_cost = df.iloc[i]['price'] * df.iloc[i]['volume']
            df.iloc[i, df.columns.get_loc('costavg')] = (prev_cost + new_cost) / df.iloc[i]['balance']
        else:
            df.iloc[i, df.columns.get_loc('costavg')] = df.iloc[i - 1]['costavg']

        df.iloc[i, df.columns.get_loc('profit')] = df.iloc[i - 1]['profit']
        if df.iloc[i]['type'] == "sell":
            df.iloc[i, df.columns.get_loc('profit')] += -df.iloc[i]['volume'] * (df.iloc[i]['price'] - df.iloc[i]['costavg'])
    df['balance'] = df['balance'].round(2)
    df['holdings'] = (df['balance'] * df['price']).round(2)
    df['purchase_cost'] = (df['balance'] * df['costavg']).round(2)
    return df


def print_overview(txs):
    balances, costavg, profit = compute_balances(txs)
    rows: List[Any] = []
    for k in sorted(balances.keys(), key=lambda k: -balances[k] * costavg[k]):
        #if round(balances[k], 6) != 0:
            purchase_cost = round(balances[k] * costavg[k])
            rows.append([isin_registry[k], k, round(balances[k], 6), costavg[k], profit[k], purchase_cost])

    print(tabulate(rows, headers=["name", "id", "balance", "costavg", "profit", "purchase_cost"]))


def print_unaccounted_txs(txs):
    print("Unaccounted txs:")
    for tx in txs:
        if tx['type'] not in ["buy", "sell", "deposit", "withdrawal", "dividend", "interest"]:
            print(tx)
        if tx['currency'] != "SEK":
            print(tx)


if __name__ == "__main__":
    txs = _load_data_avanza()
    print_overview(txs)
    print_unaccounted_txs(txs)
    plot_holdings(txs)
