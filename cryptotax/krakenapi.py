import os
import time
import krakenex
import pykrakenapi


if __name__ == "__main__":
    apikey = os.environ["KRAKEN_APIKEY"]
    privatekey = os.environ["KRAKEN_SECRET"]

    api = krakenex.API(key=apikey, secret=privatekey)
    # api._nonce = lambda: 1_000_000 * time.time()
    kraken = pykrakenapi.KrakenAPI(api)
    print(kraken.get_ledgers_info()[0])
    print(kraken.get_closed_orders())
    print(kraken.get_trades_history()[0])
