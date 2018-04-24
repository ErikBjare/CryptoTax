import pickle
from pathlib import Path
from datetime import datetime, date

import requests
from bs4 import BeautifulSoup


_coinmarketcap_data_filename = "tmp/{}-coinmarketcap-data.pickle"


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
    return d


def test_everything():
    data = load_data("bitcoin")
    tablebtc = parse_table(data.text)

    assert all(k in tablebtc[date(2017, 1, 1)] for k in ["open", "high", "low", "close"])

    data = load_data("ethereum")
    tableeth = parse_table(data.text)

    assert tablebtc[date(2017, 1, 1)]["open"] != tableeth[date(2017, 1, 1)]["open"]


if __name__ == "__main__":
    # get_data("bitcoin")
    data = load_data("bitcoin")
    table = parse_table(data.text)

    assert table[date(2017, 1, 1)]["open"]
    assert table[date(2017, 1, 1)]["high"]
    assert table[date(2017, 1, 1)]["low"]
    assert table[date(2017, 1, 1)]["close"]

    # print(data.text)
