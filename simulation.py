import warnings
warnings.filterwarnings("ignore")

from trading.processes import Simulation

import multiprocessing as mp
from datetime import date
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor

from features import *
from metaheuristics import *

def rf(asset):

    asset = features_2(asset)

    df = asset.df# .drop(columns = ["target"]).iloc[-1:].replace( [ np.nan, np.inf, -np.inf ], 0 )

    if len(df) == 0:
        return None

    train = df.iloc[:-1]
    test = df.iloc[-1:]

    rf = RandomForestRegressor()

    train = train.replace( [ np.nan, np.inf, -np.inf ], 0 ).dropna()
    test = test.replace( [ np.nan, np.inf, -np.inf ], 0 ).dropna()

    rf.fit( train.drop(columns = ["target"]), train[["target"]] )

    p = rf.predict( test.drop(columns = [ "target" ]) )

    return p[-1]

def impact(asset):
    df = asset.df
    df.index = pd.to_datetime(df.index)

    df["month"] = df.index.map(lambda x: x.month)
    df["year"] = df.index.map(lambda x: x.year)

    df["pct"] = df["close"].pct_change()

    dfp = pd.pivot_table( df, values = "pct", index = "month", columns = "year"  )
    dfp.dropna(axis = 1, inplace = True)

def simple(asset):
    return True

def single_exec(freq, periods, pct, freq_rsi, pct_rsi, lenght, smoth, slope, ema_lenght, ema_slope, ema_pct, freq_ema):

    def emaslope(asset):
        v = asset.ema_slope( ema_lenght, ema_slope ).iloc[-1]
        return v if v > 0 else None

    def rsislopes(asset):
        r = asset.rsi(lenght).iloc[-1]

        if r > 68:
            return None

        v = asset.rsi_smoth_slope( lenght, smoth, slope ).iloc[-1]
        return v if v > 0 else None 

    def momentum(asset):
        m = asset.momentum(periods)
        return m.iloc[ -1 ]

    sim = Simulation(
        broker="gbm", 
        fiat = "MX", 
        commission=0, 
        end = date(2022,1,1), 
        simulations=36,
        verbose = 1,
        parallel=True
    )

    sim.analyze(
        frequency="1m",

        test_time=1,
        analysis={
            # "MinVols":{
            #     "function":simple,
            #     "time":10,
            #     # "type":"filter"
            # }
            # "LowestMomentum_{}_{}".format( periods, pct):{
            #     "function":momentum,
            #     "time":20,
            #     "type":"filter",
            #     "filter":"lowest",
            #     "filter_qty":pct,
            #     "frequency":freq
            # },
            # "RSISmonthSlope_{}_{}_{}_{}".format(lenght, smoth, slope, pct_rsi):{
            #     "function":rsislopes,
            #     "time":50,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":pct_rsi,
            #     "frequency":freq_rsi
            # },
            # "EMASlope_{}_{}_{}".format(ema_lenght, ema_slope, ema_pct):{
            #     "function":emaslope,
            #     "time":120,
            #     "type":"filter",
            #     "filter":"highest",
            #     "filter_qty":ema_pct,
            #     "frequency":freq_ema
            # },
            # "RF2":{
            #     "function":rf,
            #     "time":360,
            #     "type":"prediction",
                
            # },
            "DE6":{
                "type":"prediction",
                "time":360,
                "function":metaheuristic
            }

        },
        run = True,
        cpus = 4
    )

    # for r, o in [ ("efficientsemivariance", "minsemivariance") ]: #  ("efficientfrontier", "minvol"),
    #     for t in [ 10]:
    #         sim.optimize(
    #             balance_time = t,
    #             risk = r,
    #             objective = o,
    #             run = True
    #         )
    for t in [ 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04 ]:
        sim.optimize(
                    balance_time = 10,
                    risk = "efficientsemivariance",
                    objective = "efficientreturn",
                    target_return = t,
                    run = True
                )
    

    df_indv = sim.results_compilation()

    if len(df_indv) == 0:
        return pd.DataFrame()

    return df_indv

def main():

    params_list = []
    for freq in [ "1m"]:
        for periods in [4]:
            for pct in [0.5]:
                for freq_rsi in [ "14d"]:
                    for pct_rsi in [ 0.5]:
                        for lenght, smoth, slope in [ (7,7, 3)]: # (11, 7 ,3), (14, 7, 3)

                            for ema_length in [ 10]:
                                for freq_ema in [ "7d" ]:
                                    for ema_slope in [ 3 ]:

                                        params_list.append( (freq, periods, pct, freq_rsi, pct_rsi, lenght, smoth, slope, ema_length, ema_slope, 0.5, freq_ema) )
    

    # grid = {
    #     "freq": ["1m", "14d"],
    #     "periods":[4],
    #     "pct":[0.4, 0.3],
    #     "freq_rsi":["1m", "14d"],
    #     "pct_rsi":[0.3],
    #     ""
    # }

    # params_list = [ tuple(list(i.values())) for i in list(ParameterGrid(grid))]


    print(f"A total of {len(params_list)} params are being tested.")
    print(params_list)

    with mp.Pool( 4 ) as pool:
        resulting_dataframes = pool.starmap( single_exec, params_list )

    # resulting_dataframes = single_exec( *params_list[0] )

    df = pd.concat( resulting_dataframes, axis = 0 )

    df.to_csv("results/lowermomentum_rsismothslope_emaslope_DE_6_simpletunning.csv", index=False)
    

    df = df.sort_values(by = "acc", ascending=False).reset_index(drop = True)

    print(df.head())
    print("\n")
    for i in range(5):
        print(df["route"].iloc[i])

if __name__ == "__main__":
    main()