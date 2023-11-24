# Simulate but without using the Simulation class as that can be the reason for it to be innefficient

import pandas as pd
from datetime import date
import json
from dateutil.relativedelta import relativedelta

from trading.func_brokers import get_assets
from trading import Asset
from trading.func_aux import timing

from metaheuristics import *

def get_asset(symbol):
    asset = Asset(
        symbol = symbol,
        fiat = "MX",
        broker = "gbm",
        frequency = "1m",
        start = date(1990,1,1),
        end = date(2023,1,1),
        source="db"
    )
    asset.df.index = pd.to_datetime( asset.df.index )

    asset.df = asset.df[ asset.df.index.map(lambda x : x.day == 1) ]

    # Insert macros

    return asset

@timing
def main():
    gen = 100
    
    predictions = { symbol:metaheuristic_integer_batch( inst = get_asset(symbol), gen = gen, verbose = False ) for symbol in get_assets()["gbm"] }

    with open( f"results/batch/DE_{gen}.json", "w" ) as fp:
        json.dump( predictions, fp )

def get_retuns_dfs():
    gen = 20
    with open( f"results/batch/DE_{gen}.json", "r" ) as fp:
        predictions = json.load( fp )

    predictions = pd.DataFrame().from_dict( predictions , orient = "index")

    dates = sorted([ date(2023,1,1) - relativedelta(months= i) for i in range(  len(predictions.columns)  ) ])

    predictions.columns = dates

    predictions = predictions.T

    close= []
    symbols = []
    for symbol in predictions.columns:
        asset = get_asset(symbol)
        if asset is None or asset.df is None or len(asset.df) == 0:
            continue
        symbols.append(symbol)
        close.append(asset.df["close"])

    real_retuns = pd.concat(close, axis = 1)
    real_retuns.columns = symbols

    real_retuns = real_retuns.pct_change()

    return predictions, real_retuns


def simple_past_returns():
    predictions, real_retuns = get_retuns_dfs()

    tmp = real_retuns.shift(1)

    tmp[ tmp > 0 ]

if __name__ != "__main__":
    main()
    