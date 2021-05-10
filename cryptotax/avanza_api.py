from datetime import datetime, timedelta
import json

import requests
from joblib import Memory

cachedir = ".joblibcache"
memory = Memory(cachedir, verbose=0)


@memory.cache
def get_history(oid: int, start=None, end=None, cache=False):
    now = datetime.now()
    r = requests.post(
        "https://www.avanza.se/ab/component/highstockchart/getchart/orderbook",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "orderbookId": oid,
                "chartType": "AREA",
                "widthOfPlotContainer": 558,
                "chartResolution": "DAY",
                "navigator": True,
                "percentage": False,
                "volume": False,
                "owners": False,
                "start": (now - timedelta(days=180)).isoformat() + "Z",
                "end": now.isoformat() + "Z",
                "ta": [],
            }
        ),
    )
    r.raise_for_status()
    data = r.json()
    if "navigatorPoints" in data:
        data = data["navigatorPoints"]
    elif "marketMakerBidPoints" in data:
        data = data["marketMakerBidPoints"]
    else:
        raise Exception(data)
    for point in data:
        point[0] = datetime.fromtimestamp(point[0] / 1000)
    data = [p for p in data if p[1]]
    # print(data.keys())
    # print(data["navigatorPoints"][0])
    # print(data["navigatorPoints"][-1])
    # print(data["lastPrice"])
    # print(data["allowedResolutions"])
    return data


def test_get_history():
    now = datetime.now()
    h = get_history(463161, now - timedelta(days=30), now)
    print(h)
    assert h


if __name__ == "__main__":
    get_history(463161)
