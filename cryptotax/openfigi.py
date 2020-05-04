import requests
import json
import os
import logging
from pprint import pprint

import pytest

_amazon_isin = "US0231351067"


logger = logging.getLogger(__name__)


def get_by_isin(isin, exchcode="US"):
    data = [{"idType": "ID_ISIN", "idValue": isin, "exchCode": exchcode}]
    headers = {"Content-Type": "text/json"}
    if "OPENFIGI_APIKEY" in os.environ:
        headers["X-OPENFIGI-APIKEY"] = os.environ["OPENFIGI_APIKEY"]
    r = requests.post(
        "https://api.openfigi.com/v1/mapping", json.dumps(data), headers=headers
    )
    if b"Too Many Requests" in r.content:
        logger.error(
            "Ratelimited by OpenFIGI, put an API key in the 'OPENFIGI_APIKEY' environment variable"
        )
        return None
    return r.json()[0] if "error" not in r.json()[0] else None


@pytest.mark.skipif(
    "OPENFIGI_APIKEY" not in os.environ, reason="No OPENFIGI_APIKEY env var set"
)
def test_get_by_isin():
    assert get_by_isin(_amazon_isin)
    assert get_by_isin("US2312312321313") is None


if __name__ == "__main__":
    pprint(get_by_isin(_amazon_isin))
