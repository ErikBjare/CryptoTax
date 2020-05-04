from datetime import timedelta

import pandas as pd


def annualized_ror(rate, duration):
    return (1 + rate)**(timedelta(days=365) / duration) - 1


def timevalue(txs):
    """
    I want to know the time-value of my different investments.

    Proposed method:
     - Value a timeperiod by the annualized RoR
     - Weight that timeperiod by its duration
    """
    for t1, t2 in zip(txs[:-1], txs[1:]):
        td = t2["date"] - t1["date"]
        ror = (t2["price"] - t1["price"]) / t1["price"]
        aror = annualized_ror(ror, td)
        print(aror)
        print(t1, t2)


def test_timevalue():
    txs = [
        {"date": pd.Timestamp("2018-01-01"), "price": 1.0, "volume": 1.0},
        {"date": pd.Timestamp("2018-02-01"), "price": 1.1, "volume": 1.0}
    ]
    timevalue(txs)
    assert False


def test_annualized_ror():
    assert 0.12 < annualized_ror(0.01, timedelta(days=30)) < 0.13
    assert 0.12 < annualized_ror(0.5, timedelta(1275)) < 0.125
    assert 0.78 < annualized_ror(0.05, timedelta(days=30)) < 0.82
