from collections import defaultdict
from main import get_trades, load_deposits
from util import fiatconvert


def swedish_taxes(trades, deposits):
    asset_cost = defaultdict(int)
    asset_vol = defaultdict(int)
    profits = defaultdict(lambda: [0, 0])
    for deposit in deposits:
        curr = deposit['asset']
        if curr[0] == 'X':
            continue
        amount = deposit['amount']
        cost = fiatconvert(amount, curr[1:], "SEK", deposit["time"])
        asset_cost[curr] += cost
        asset_vol[curr] += amount
    print(asset_vol)
    print(asset_cost)

    for trade in trades:
        pair = trade["pair"]
        cost = trade["cost"]
        if trade["type"] == "sell":
            pair = reversed(pair)
            cost = trade["vol"]
        c1, c2 = pair

        # Calculate cost in SEK
        cost_sek = fiatconvert(trade["cost_usd"], "USD", "SEK", trade["time"])

        asset_vol[c1] += trade["vol"]
        asset_cost[c1] += cost_sek

        # Calculate average price in SEK
        if c2 not in asset_vol:
            print(f"No prior knowledge of asset: {c2}")
        avg_price = asset_cost[c2] / (asset_vol[c2] or 1)

        profit = cost_sek - cost * avg_price
        print(f"profit: {profit:>5.2f} SEK",
              "\t{:<4}".format(trade["type"].upper()),
              f"vol: {'{:4.4}'.format(trade['vol']):>10}",
              ",{:>15}".format(f"cost: {trade['cost']:>4.4}"),
              f"price: {trade['price']:>4.4}, avg_price[{c2}]: {avg_price:>4.4}",
              f"(buy {c1}/{c2} | sell {c2}/{c1})")
        asset_vol[c2] -= trade["vol"]
        asset_cost[c2] -= cost_sek
        year = trade["time"].year
        if profit > 0:
            profits[year][0] += profit
        else:
            profits[year][1] += profit
        if asset_vol[c2] < 0:
            print(f"Negative volume of asset: {c2}")
            asset_vol[c2] = 0
            asset_cost[c2] = 0
    print('-'*80)
    for (year, profit) in profits.items():
        print(f"{year}: ", "Profit: {:>20.20f} Loss: {:>20.20f}".format(*profit))


if __name__ == "__main__":
    trades = get_trades()
    deposits = load_deposits()
    swedish_taxes(trades, deposits)
