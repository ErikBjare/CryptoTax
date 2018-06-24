from collections import defaultdict

from .load_data import _load_csv

_typ_to_type = {
    "Insättning": "deposit",
    "Uttag": "withdrawal",
    "Köp": "buy",
    "Köp, rättelse": "buy",
    "Sälj": "sell",
    "Sälj, rättelse": "sell",
    "Räntor": "interest",
}


def _load_data_avanza(filename="./data_private/avanza-transactions_utf8.csv"):
    data = _load_csv(filename, delimiter=";")
    for row in data:
        row["date"] = row.pop("Datum")
        ttype = row.pop("Typ av transaktion")
        if type in _typ_to_type:
            row["type"] = _typ_to_type[ttype]
        else:
            # print(type)
            row["type"] = ttype

        row["asset"] = row.pop("Värdepapper/beskrivning")

        volume = row.pop("Antal")
        row["volume"] = abs(float(volume.replace(",", "."))) if volume != "-" else None

        price = row.pop("Kurs")
        row["price"] = float(price.replace(",", ".")) if price != "-" else None
    #print(data[0].keys())
    return data


def compute_balances(trades):
    balances = defaultdict(lambda: 0)
    costavg = defaultdict(lambda: 0)
    profit = defaultdict(lambda: 0)

    for t in sorted(trades, key=lambda t: t['date']):
        #print(t['asset'], t['type'],  t['volume'])

        if t["type"] == "buy":
            prev_cost = balances[t['asset']] * costavg[t['asset']]
            new_cost = t['volume'] * t['price']
            new_bal = t['volume'] + balances[t['asset']]
            if new_bal == 0:
                print(f"new balance was zero, something is wrong: \n - {t}\n - {t['volume']}\n - {balances[t['asset']]}")
            else:
                costavg[t['asset']] = (prev_cost + new_cost) / new_bal

        if t["type"] == "sell":
            #print(costavg[t['asset']])
            p = (t["price"] - costavg[t["asset"]]) * t["volume"]
            profit[t["asset"]] += p
            #print(f"Profited: {p}")

        balances[t['asset']] += t['volume'] * (1 if t["type"] == "buy" else -1)
    return balances, costavg, profit


def print_table(table):
    from IPython.display import HTML, display
    from tabulate import tabulate
    display(HTML(tabulate(table, tablefmt='html')))


def display_performance(balances, costavg, profit):
    header = ["Asset", "Balance", "Profit", "Costavg"]
    table = []
    for asset in balances:
        #if round(balances[asset], 1) != 0:
        table.append([
            asset,
            round(balances[asset], 2),
            round(profit[asset]),
            round(costavg[asset], 2)
        ])
        #print(f"{asset}:")
        #print(f" - {balances[asset]}")
        # print(f" - {costavg[asset]}")
        #print(f" - profit: {profit[asset]}")

    print_table([header] + sorted(table, key=lambda r: -r[2]))
