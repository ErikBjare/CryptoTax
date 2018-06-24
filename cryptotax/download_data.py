import pickle
from pathlib import Path
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup


_coinmarketcap_data_filename = "tmp/{}-coinmarketcap-data.pickle"
_pricehistory_filename = "tmp/{}-pricehistory.pickle"


currency2symbolmap = {
    "bitcoin": "XXBT",
    "ethereum": "XETH",
    "stellar": "XXLM"
}


def get_data_from_coinmarketcap(currency):
    r = requests.get(f"https://coinmarketcap.com/currencies/{currency}/historical-data/?start=20140101&end=20180423")
    with open(_coinmarketcap_data_filename.format(currency), "wb") as f:
        pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)


def load_data(currency):
    filename = _coinmarketcap_data_filename.format(currency)
    if not Path(filename).exists():
        print(f"Didn't find data file for {currency}, downloading...")
        get_data_from_coinmarketcap(currency)
    with open(filename, "rb") as f:
        return pickle.load(f)


def parse_table(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    tables = soup.find_all("table")

    headers = [el.text.lower() for el in tables[0].find_all("th")]
    rows = []

    for row in tables[0].find_all("tr"):
        cells = [el.text for el in row.find_all("td")]
        if len(cells) == len(headers):
            rows.append({k: v for k, v in zip(headers, cells)})
        elif cells:
            print(f"Incomplete row: {cells}")

    d = {datetime.strptime(r["date"], "%b %d, %Y").date(): r for r in rows}
    for k, v in d.items():
        v.pop("date")
        v.pop("market cap")
        d[k] = {ohlc: float(d[k][ohlc].replace(",", "")) for ohlc in d[k]}
    return d


def _save_table(currency, data):
    with open(_pricehistory_filename.format(currency2symbolmap[currency]), "wb") as f:
        pickle.dump(data, f)
        print(f"Price history for {currency} saved!")


def test_everything():
    data = load_data("bitcoin")
    tablebtc = parse_table(data.text)

    assert all(k in tablebtc[date(2017, 1, 1)] for k in ["open", "high", "low", "close"])

    data = load_data("ethereum")
    tableeth = parse_table(data.text)

    assert tablebtc[date(2017, 1, 1)]["open"] != tableeth[date(2017, 1, 1)]["open"]


if __name__ == "__main__":
    # get_data("bitcoin")
    for currency in ["bitcoin", "ethereum", "stellar"]:
        print(f"Getting price history for {currency}...")
        data = load_data(currency)
        table = parse_table(data.text)
        _save_table(currency, table)

    # print(data.text)
