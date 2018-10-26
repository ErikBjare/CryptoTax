from collections import defaultdict
from typing import Dict, Tuple, List, Any
from copy import deepcopy
from datetime import datetime, timedelta

from deprecation import deprecated
from tabulate import tabulate
import pandas as pd
import pydash
import matplotlib.pyplot as plt

from .load_data import _load_csv

pd.set_option('display.expand_frame_repr', False)

# FIXME: Handle assets priced in USD/EUR correctly


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


isin_to_name = {}

with open('isin_to_avanza_ids.csv') as f:
    isin_to_avanza_ids = {row[0]: row[1] for row in [line.split(",") for line in f.readlines()]}


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
        isin_to_name[row["asset_id"]] = row["asset_name"]

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


def plot_holdings(txs: List[Dict]) -> None:
    # TODO: Create a different plot that shows the past performance of the portfolio **if it always had the current allocation**.
    df = all_txs_to_dataframe([tx for tx in txs if tx["volume"] and tx["type"] in ["buy", "sell"]])
    group = df.reset_index().set_index('date').groupby("asset_name")['holdings']
    for k, g in group:
        if g.iloc[-1] > 10000:
            g.plot(label=k)

    plt.title("Value of holdings")
    plt.ylabel("SEK")
    plt.xlim(pd.Timestamp("2017-01-01"), pd.Timestamp.now())
    plt.ylim(0)
    plt.legend()
    plt.show()


def plot_realized_profits():
    pass


def txs_to_dataframe(txs: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(txs)
    df.date = pd.to_datetime(df.date)
    df.volume *= [1 if otype == "buy" else -1 for otype in df['type']]
    df = df.set_index("date")
    return df.sort_index()


def _clean_price(df):
    asset_id = df.reset_index()['asset_id'].iloc[0]
    asset_name = df.reset_index()['asset_name'].iloc[0]
    try:
        calc_price = abs(df['amount'] / df['volume'])
        error_fac = (df['price'] / calc_price).dropna()
        e = 0.1
        if not asset_id[:2] == "US":
            assert error_fac.between(1 - e, 1 + e).all()
    except AssertionError:
        print(f"Price didn't pass consistency check, trying to infer... ({asset_name}, {asset_id})")
        # print(error_fac)
        # This fixes some issues, but might create others...
        df['price'] = -df['amount'] / df['volume']
    return df


def _calc_costavg(df: pd.DataFrame) -> pd.DataFrame:
    df['profit'] = 0.0
    df['costavg'] = df['price']
    df['balance'] = df['volume'].cumsum()
    for i, d in list(enumerate(df.index))[1:]:
        prev_cost = df.iloc[i - 1]['balance'] * df.iloc[i - 1]['costavg']
        if df.iloc[i]['type'] == "buy":
            new_cost = df.iloc[i]['price'] * df.iloc[i]['volume']
            df.iloc[i, df.columns.get_loc('costavg')] = (prev_cost + new_cost) / df.iloc[i]['balance']
        else:
            df.iloc[i, df.columns.get_loc('costavg')] = df.iloc[i - 1]['costavg']

        df.iloc[i, df.columns.get_loc('profit')] = df.iloc[i - 1]['profit']
        if df.iloc[i]['type'] == "sell":
            df.iloc[i, df.columns.get_loc('profit')] += -df.iloc[i]['volume'] * (df.iloc[i]['price'] - df.iloc[i]['costavg'])
    return df


def test_calc_costavg():
    df = pd.DataFrame([["2018-01-01", "buy", 1, 1],
                       ["2018-01-04", "buy", 1.5, 1],
                       ["2018-01-08", "sell", 2, -2]],
                      columns=["date", "type", "price", "volume"]).set_index("date")
    df['profit'] = 0.0
    df = _calc_costavg(df)
    assert all(df['costavg'] == [1, 1.25, 1.25])
    assert all(df['profit'] == [0, 0, 1.5])


def _fill_with_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    from .avanza_api import get_history
    df = df.reset_index().set_index('date')

    isin = df.iloc[0]['asset_id']
    if isin in isin_to_avanza_ids:
        pricehistory = dict((str(dt.date()), p) for dt, p in get_history(isin_to_avanza_ids[isin]))
    else:
        # print(f"Could not get price history for {df.iloc[0]['asset_name']} ({isin})")
        return df

    def _create_row(r, dt):
        dtstr = dt.isoformat()[:10]
        if dtstr not in pricehistory:
            return None
        return {
            "asset_name": r["asset_name"],
            "asset_id": r["asset_id"],
            "date": dt,
            "price": pricehistory[dtstr],
            "balance": r["balance"],
            "costavg": r["costavg"],
            "profit": r["profit"],
            "volume": 0,
        }
    rows = []
    for r1, r2 in [(df.iloc[i], df.iloc[i + 1]) for i in range(len(df) - 1)]:
        # The trick below is required because sometimes the passed dataframe will be
        # indexed on only 'date' and sometimes on ('date', 'asset_id').
        start = r1.name
        end = r2.name
        dt = start
        while dt + timedelta(days=1) < end:
            dt += timedelta(days=1)
            row = _create_row(r1, dt)
            if row:
                rows.append(row)
    if not rows:
        return df
    now = datetime.now()
    dt = end
    while dt + timedelta(days=1) < now:
        dt += timedelta(days=1)
        row = _create_row(r2, dt)
        if row:
            rows.append(row)

    # Why is `append(..., sort=False).sort_index()` not equivalent to `append(..., sort=True)`?
    df = df.append(pd.DataFrame(rows).set_index("date"), sort=False).sort_index()
    return df


def test_fill_with_daily_prices() -> None:
    txs = [
        {"date": pd.Timestamp("2018-09-01"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "buy", "price": 1, "volume": 1},
        {"date": pd.Timestamp("2018-09-25"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "sell", "price": 2, "volume": 1},
    ]
    for tx in txs:
        tx['amount'] = tx['price'] * tx['volume']
    df = all_txs_to_dataframe(txs)
    assert df.loc[("2018-09-04", "SE0011527613")]["price"]


def test_fillall_with_daily_prices() -> None:
    txs = [
        {"date": pd.Timestamp("2018-09-01"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "buy", "price": 1, "volume": 1},
        {"date": pd.Timestamp("2018-09-01"), "asset_id": "SE0000709123", "asset_name": "Swedbank Robur Ny Teknik", "type": "buy", "price": 1, "volume": 1},
        {"date": pd.Timestamp("2018-09-25"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "sell", "price": 2, "volume": 1},
        {"date": pd.Timestamp("2018-09-25"), "asset_id": "SE0000709123", "asset_name": "Swedbank Robur Ny Teknik", "type": "sell", "price": 2, "volume": 1},
    ]
    for tx in txs:
        tx['amount'] = tx['price'] * tx['volume']
    df = all_txs_to_dataframe(txs)
    assert df.loc[("2018-09-04", 'SE0011527613')]["price"]
    assert df.loc[("2018-09-04", 'SE0000709123')]["price"]


def all_txs_to_dataframe(txs: List[Dict]) -> pd.DataFrame:
    df = txs_to_dataframe(txs).reset_index().set_index(["date", "asset_id"])
    df = df.groupby('asset_id').apply(_clean_price)
    df = df.groupby('asset_id').apply(_calc_costavg)
    df = df.groupby(level='asset_id').apply(lambda df: _fill_with_daily_prices(df))
    # The below is required because the groupby-apply for fill_with_daily_prices leaves an extra non-index asset_id column for whatever reason
    df.pop('asset_id')

    # TODO: Handle duplicates correctly
    non_dups = ~df.index.duplicated()
    df = df[non_dups]
    # print(df.tail())

    df = df.unstack(['asset_id'])
    # FIXME: Fill price from external data before ffill
    for key in ['asset_name', 'price', 'balance', 'costavg', 'profit']:
        df[key] = df[key].ffill()
    df = df.stack(['asset_id'], dropna=False)
    df['balance'] = df['balance'].round(2)
    currency_factors = [9 if "US" in aid else 1 for aid in df.index.get_level_values('asset_id')]
    df['holdings'] = (df['balance'] * df['price']).round(2) * currency_factors
    df['purchase_cost'] = (df['balance'] * df['costavg']).round(2) * currency_factors
    df['profit_unrealized'] = df['holdings'] - df['purchase_cost']
    return df


def plot_total_holdings(txs):
    # Filter away Avanza -> Avanza transactions
    txs = [tx for tx in txs if 'Avanzakonto' not in tx['asset_name']]
    df = all_txs_to_dataframe(txs)

    holdings_assets = df.groupby("date")['holdings'].sum()
    holdings_sek = df.groupby("date")["amount"].sum().cumsum()

    for df in [holdings_assets, holdings_sek]:
        df[df < 0] = 0

    holdings = pd.concat([holdings_assets, holdings_sek], axis=1)
    holdings.plot.area()
    plt.xlim(datetime(2017, 1, 1), datetime.now())
    plt.ylim(0)
    plt.show()


def test_all_txs_to_dataframe():
    txs = [
        {"date": pd.Timestamp("2018-09-01"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "buy", "price": 53.1, "volume": 3},
        {"date": pd.Timestamp("2018-09-01"), "asset_id": "US0000000000", "asset_name": "test", "type": "buy", "price": 102.4, "volume": 2},
        {"date": pd.Timestamp("2018-09-25"), "asset_id": "SE0011527613", "asset_name": "Avanza Global", "type": "sell", "price": 76.2, "volume": 1},
    ]
    for tx in txs:
        tx['amount'] = tx['price'] * tx['volume']
    df = all_txs_to_dataframe(txs)
    print(df)
    assert df.loc[('2018-09-25', 'US0000000000'), 'holdings'].item()


def print_overview(txs) -> None:
    balances, costavg, profit = compute_balances(txs)
    rows: List[Any] = []
    for k in sorted(balances.keys(), key=lambda k: -balances[k] * costavg[k]):
        #if round(balances[k], 6) != 0:
            purchase_cost = round(balances[k] * costavg[k])
            rows.append([isin_to_name[k], k, round(balances[k], 6), costavg[k], profit[k], purchase_cost])

    print(tabulate(rows, headers=["name", "id", "balance", "costavg", "profit", "purchase_cost"]))


def print_unaccounted_txs(txs):
    print("Unaccounted txs:")
    for tx in txs:
        if tx['type'] not in ["buy", "sell", "deposit", "withdrawal", "dividend", "interest"]:
            print(tx)
        if tx['currency'] != "SEK":
            print(tx)


def print_overview_df(txs: List[Dict]) -> None:
    txs = [tx for tx in txs if tx["type"] in ["buy", "sell"]]
    df = all_txs_to_dataframe(txs)
    df = df.groupby('asset_id').agg('last')
    df = df[df['holdings'] > 0]

    summed = df.sum()
    summed['asset_name'] = '--- SUMMED ---'
    summed[['costavg', 'balance']] = None
    summed = summed.rename('-- SUMMED --')

    print(df.append(summed)[['asset_name', 'holdings', 'costavg', 'balance', 'profit_unrealized']])


def _load_crypto_trades():
    from .load_data import load_all_trades
    trades = load_all_trades()
    for t in trades:
        for k in ['ordertype', 'ledgers']:
            if k in t:
                del t[k]
    return trades


if __name__ == "__main__":
    txs = _load_data_avanza()

    if False:
        crypto_txs = _load_crypto_trades()
        print(crypto_txs)

    print_overview(txs)
    print_overview_df(txs)
    # print_unaccounted_txs(txs)
    plot_holdings(txs)
    plot_total_holdings(txs)
