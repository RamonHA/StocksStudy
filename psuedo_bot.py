# Simulate but without using the Simulation class as that can be the reason for it to be innefficient
# First Update Stock info weekly

import pandas as pd
import numpy as np
from datetime import date
import json
from dateutil.relativedelta import relativedelta

from trading.func_brokers import get_assets
from trading import Asset
from trading.func_aux import timing

from trading.optimization import Optimization
from trading.func_brokers import historic_download


from metaheuristics import *

no_care_assets = ["EDOARDOB", "VINTE", "BSMXB", "ICHB"]

def get_asset(symbol, freq = "1m"):
    if symbol in no_care_assets:
        return None
    
    asset = Asset(
        symbol = symbol,
        fiat = "MX",
        broker = "gbm",
        frequency = freq,
        start = date(1980,1,1),
        end = date.today(),
        source="yahoo"
    )

    if asset.df is None or len(asset.df) < 10:
        return None
    
    asset.df.index = pd.to_datetime( asset.df.index )

    if freq == "1m":
        asset.df = asset.df[ asset.df.index.map(lambda x : x.day == 1) ]

    if (date.today() - asset.df.index[-1].date()).days > 60:
        print(symbol,asset.df.index[-1].date() )
        return None 

    # Insert macros

    return asset

def filter_assets():
    assets = {}
    data = []
    
    for symbol in get_assets()["gbm"]:
        asset = get_asset(symbol, freq="1w")
        if asset is None or len(asset.df) == 0:
            continue

        assets[symbol] = asset
 
        d = asset.df.iloc[-56:]
        data.append( [ symbol, len(d[ d["volume"] == 0 ]), d["volume"].mean() ] )

    df = pd.DataFrame(data, columns = ["symbol", "zeros", "mean"])
    # df.sort_values(by = "zeros", ascending = True, inplace = True)
    df.reset_index(drop = True, inplace = True)

    df = df[ (df["zeros"] <= 5) & (df["mean"] >= 10000 )]

    return df["symbol"].tolist()

@timing
def main():
    gen = 40

    predictions = {                                                             
        symbol:metaheuristic_integer_batch( 
            inst = get_asset(symbol), 
            gen = gen, 
            verbose = False,
            train_size=0.85
       ) for symbol in filter_assets()
    }

    # with open(f"results/batch/Bot_DE_{gen}.json", "r") as fp:
    #     predictions = json.load(fp)["predictions"]

    df = pd.DataFrame.from_dict(predictions, orient = "index")
    cols = len(df.columns)
    cols = [ date(2019,1,1) + relativedelta(months=d) for d in range(cols) ]
    df.columns = cols    
    df = df.T

    selected_assets = df[df > 0].iloc[-1].dropna().sort_values(ascending=False)
    
    if len(selected_assets[ selected_assets > 0.01 ]) > 30:
        selected_assets = selected_assets[ selected_assets > 0.01 ]

    # return_selected = df[ selected_assets.columns ]
    opt = Optimization(
        assets=list(selected_assets.to_frame().T.columns),
        
        start = date.today() - relativedelta(weeks=80),
        end = date.today(),
        frequency = "1w",

        exp_returns=selected_assets.to_frame(),
        
        risk = "efficientsemivariance",
        objective="efficientreturn",
        target_return = 0.03,
        broker = "gbm",
        fiat = "MX",
        source = "yahoo",

        interpolate=True, # If missing data
        verbose = 2,
    )

    portafolio = 8337.92

    allocation, qty, pct = opt.optimize( portafolio )

    def clean(d):
        for i,v in d.items():
            d[i] = float(v)
        
        return d

    data = {
        "predictions":predictions,
        "allocation":clean(allocation),
        "qty":clean(qty),
        "pct":clean(pct)
    }

    with open( f"results/batch/Bot_DE_{gen}.json", "w" ) as fp:
        json.dump(data,fp)

    with open( f"results/batch/Bot_DE_{gen}_{date.today()}.json", "w" ) as fp:
        json.dump(data,fp)


if __name__ == "__main__":
    historic_download(
        broker="gbm",
        fiat = "MX",
        frequency="1w",
        start=date(1980,1,1),
        source="yahoo",
        verbose=True
    )
    
    main()