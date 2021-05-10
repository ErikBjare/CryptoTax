import pickle
import json
import iso8601
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


_coinmarketcap_data_filename = "tmp/{}-coinmarketcap-data.pickle"
_pricehistory_filename = "tmp/{}-pricehistory.pickle"


currency2symbolmap = {
    "bitcoin": "XXBT",
    "ethereum": "XETH",
    "stellar": "XXLM",
    "eos": "XEOS",
    "bitcoin-cash": "XBCH",
    "gnosis": "XGNO",
    "monero": "XXMR",
    "dogecoin": "XXDG",
    "ethereum-classic": "XETC",
    "siacoin": "XXSC",
    "havven": "SNX",
    "chainlink": "LINK",
    "polkadot": "DOT",
    "uniswap": "UNI",
    "omg-network": "OMG",
}

symbol2currencymap = {v: k for k, v in currency2symbolmap.items()}


def get_data_from_coinmarketcap(currency):
    d_str = datetime.now().strftime("%Y%m%d")
    r = requests.get(
        f"https://coinmarketcap.com/currencies/{currency}/historical-data/?start=20140101&end={d_str}"
    )
    p = Path(_coinmarketcap_data_filename.format(currency))

    # Ensure directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("wb") as f:
        pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)


def get_price(currency: str, date: date) -> float:
    """
    Return the historical price of currency at day represented by date from coingecko API.
    Currency is a string representing a currency in the Coingecko API.
    """
    d_str = date.strftime("%d-%m-%Y")
    while True:
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{currency}/history?date={d_str}&localization=false"
            )
            q = json.loads(r.content)["market_data"]["current_price"]["usd"]
            return q
        except Exception as e:
            if r.status_code == 429:
                time_out = int(r.headers["Retry-After"]) + 5
            else:
                time_out = 10
            print(f"Error occured downloading data: {e}")
            print(f"Retrying in {time_out} seconds.")
            print(f"status: {r.status_code}")
            time.sleep(time_out)


def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def get_data_from_coingecko(currency):
    # FIXME: Don't hardcode
    date_ranges = [
        (date(2016, 10, 1), date(2016, 10, 15)),
        # (date(2020, 11, 1), date(2020, 12, 31)),
    ]

    ticks = {}
    for start, end in date_ranges:
        for _date in tqdm(_daterange(start, end), total=int((end - start).days)):
            ticks[_date] = get_price(currency, _date)

    return ticks


def load_data(currency):
    filename = _coinmarketcap_data_filename.format(currency)
    if not Path(filename).exists():
        print(f"Didn't find data file for {currency}, downloading...")
        get_data_from_coinmarketcap(currency)

    with open(filename, "rb") as f:
        return pickle.load(f)


def parse_json(doc) -> Dict[date, Dict[str, Any]]:
    soup = BeautifulSoup(doc, "html.parser")
    data = json.loads(soup.find("script", {"id": "__NEXT_DATA__"}).decode_contents())

    state = data["props"]["initialState"]
    _id = list(state["cryptocurrency"]["info"]["data"].keys())[0]
    quotes = state["cryptocurrency"]["ohlcvHistorical"][_id]["quotes"]
    ticks = {}
    for q in quotes:
        q = q["quote"]["USD"]
        ticks[iso8601.parse_date(q["timestamp"]).date()] = q
    return ticks


def _save_table(currency, data):
    with open(_pricehistory_filename.format(currency2symbolmap[currency]), "wb") as f:
        pickle.dump(data, f)
        print(f"Price history for {currency} saved!")


def test_everything():
    data = load_data("bitcoin")
    btc = parse_json(data.text)

    d = date(2017, 1, 1)
    assert all(k in btc[d] for k in ["open", "high", "low", "close"])

    data = load_data("ethereum")
    eth = parse_json(data.text)

    assert btc[d]["open"] != eth[d]["open"]


def main():
    # get_data("bitcoin")
    for currency in currency2symbolmap:
        print(f"Getting price history for {currency}...")
        data = load_data(currency)
        table = parse_json(data.text)
        _save_table(currency, table)


if __name__ == "__main__":
    main()
