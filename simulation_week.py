import warnings
warnings.filterwarnings("ignore")

from trading.processes import Simulation

import multiprocessing as mp
from datetime import date
import pandas as pd

from sklearn.model_selection import ParameterGrid

def simple(asset):
    return True

def single_exec():

    # def emaslope(asset):
    #     v = asset.ema_slope( ema_lenght, ema_slope ).iloc[-1]
    #     return v if v > 0 else None

    # def rsislopes(asset):
    #     v = asset.rsi_smoth_slope( lenght, smoth, slope ).iloc[-1]
    #     return v if v > 0 else None

    # def momentum(asset):
    #     m = asset.momentum(periods)
    #     return m.iloc[ -1 ]

    sim = Simulation(
        broker="gbm", 
        fiat = "MX", 
        commission=0, 
        end = date(2022,1,1), 
        simulations=52*3,
        verbose = 0
    )

    sim.analyze(
        frequency="1w",
        test_time=1,
        analysis={
            "MinVols":{
                "function":simple,
                "time":10,
                # "type":"filter"
            }
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

        },
        run = False
    )

    for r, o in [ ("efficientsemivariance", "minsemivariance") ]: #  ("efficientfrontier", "minvol"),
        for t in [ 10, 20, 40, 80]:
            sim.optimize(
                balance_time = t,
                risk = r,
                objective = o,
                run = False
            )
    
    df_indv = sim.results_compilation()

    if len(df_indv) == 0:
        return pd.DataFrame()

    return df_indv

def main():

    # params_list = []
    # for freq in [ "1m", "14d"]:
    #     for periods in [4]:
    #         for pct in [0.4, 0.3]:
    #             for freq_rsi in ["1m", "14d"]:
    #                 for pct_rsi in [ 0.3]:
    #                     for lenght, smoth, slope in [ (7,7, 3), (11, 7 ,3), (14, 7, 3)]:

    #                         for ema_length in [10, 20, 40, 80]:
    #                             for freq_ema in [ "1d", "7d" ]:
    #                                 for ema_slope in [ 3, 4 ]:

    #                                     params_list.append( (freq, periods, pct, freq_rsi, pct_rsi, lenght, smoth, slope, ema_length, ema_slope, 0.5, freq_ema) )
    

    # grid = {
    #     "freq": ["1m", "14d"],
    #     "periods":[4],
    #     "pct":[0.4, 0.3],
    #     "freq_rsi":["1m", "14d"],
    #     "pct_rsi":[0.3],
    #     ""
    # }

    # params_list = [ tuple(list(i.values())) for i in list(ParameterGrid(grid))]


    # print(f"A total of {len(params_list)} params are being tested.")
    # print(params_list)

    # with mp.Pool( 4 ) as pool:
    #     resulting_dataframes = pool.starmap( single_exec, params_list )

    resulting_dataframes = single_exec(  ) # *params_list[0]
    df = resulting_dataframes

    # df = pd.concat( resulting_dataframes, axis = 0 )

    df.to_csv("results/weekly.csv", index=False)
    
    df = df.sort_values(by = "acc", ascending=False).reset_index(drop = True)

    print(df.head())
    print("\n")
    l = 5 if 5 <= len(df) else len(df)
    for i in range(l):
        print(df["route"].iloc[i])

if __name__ == "__main__":
    main()