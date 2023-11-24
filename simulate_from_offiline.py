import pandas as pd
import os
import json
from datetime import datetime, date
import numpy as np
from itertools import combinations
import operator
import functools
from multiprocessing.pool import ThreadPool
from dateutil.relativedelta import relativedelta

from trading.func_brokers import get_assets
from trading import Asset
from trading.func_aux import timing

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.de import DE

def read_from_simulation():

    analysis = "DE_5_svm_integer"

    pwd = r"C:\Users\ramon\Documents\Trading\WorkingDir\gbm\results\mx\1m_1\{}\360_".format(analysis)

    files = [i for i in os.listdir( pwd ) if "resume" not in i]

    all_data = {}
    for file in files:
        with open( pwd + "/" + file, "r") as fp:
            data = json.load( fp )

        all_data[datetime.strptime( file.split("_")[1], "%Y-%m-%d %H-%M-%S" )] = data 


    de = pd.DataFrame().from_dict(all_data).T

def read_from_batch():
    with open( "results/batch/DE_20.json", "r" ) as fp:
        data = json.load(fp)

    df = pd.DataFrame.from_dict(data, orient = "index")
    cols = len(df.columns)
    cols = [ date(2019,1,1) + relativedelta(months=d) for d in range(cols) ]
    df.columns = cols    
    df = df.T
    df = df.loc[ :date(2021,12,1) ]

    return df

# de = read_from_simulation()
de = read_from_batch()


# Generar las matrices de las demas.
# Se puede lanzar la simulacion con la clase que se tiene o hacerlo manual para tener
# las matrices

returns = {}
for symbol in list(get_assets()["gbm"].keys()):
    a = Asset(
        symbol=symbol,
        fiat = "MX",
        broker = "gbm",

        frequency="1m",
        start = date(2019,1,1),
        end = date(2022, 3,1),

        source="db"
    )

    if a is not None and a.df is not None and "close" in a.df.columns:
        a.df.index= pd.to_datetime( a.df.index )
        a.df = a.df[ a.df.index.map( lambda x : x.day == 1 ) ]
        
        if len(a.df) == 0 or "close" not in a.df:
            continue

        returns[ symbol ] = a.df["close"].pct_change().shift(-1)

returns = pd.DataFrame(returns)
# Clean outliers
returns.where( abs(returns) < 1, 0 , inplace = True)


assets = {}
for symbol in get_assets()["gbm"]:
    a = Asset(
        symbol=symbol,
        fiat = "MX",
        broker = "gbm",

        frequency="1w",
        start = date(2019,1,1),
        end = date(2022, 3,1),

        source="db"
    )

    if a is not None and a.df is not None and "close" in a.df.columns:
        a.df.index= pd.to_datetime( a.df.index )
        assets[ symbol ] = a

def resampling(df):
    df = df.resample( "1MS" ).last()
    return df

# # First Test
# just_positive = (1 + ((de > 0) * returns).replace(np.nan, 0).mean(axis = 1)   ).prod()
# list_returns = {}
# list_returns["just_positive"] = just_positive

# ema_close = resampling(pd.DataFrame({ a.symbol:((a.ema(30) >= a.df["close"]).rolling(4).sum() == 0) for a in assets.values()}))

# ema_slope = resampling(pd.DataFrame({ a.symbol:( a.ema(15) > 0) for a in assets.values()}))

# momentum = resampling(pd.DataFrame({ a.symbol:a.momentum(4) for a in assets.values()}))

# rsi_rsi_smoth = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth(7, 7) < a.rsi_smoth(7, 3) ) for a in assets.values()}))

# rsi_slope = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth_slope(10, 7, 2) > 0 ) for a in assets.values()}))

# rsi_std = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth(7,7).rolling(15).std() > 1.5 ) for a in assets.values()}))

# rsi_thr = resampling(pd.DataFrame({ a.symbol:( (a.rsi( 7 ) > 68).rolling(20).sum() == 0 ) for a in assets.values()}))

# # metrics = [ema_close , ema_slope , rsi_rsi_smoth , rsi_slope , rsi_std , rsi_thr]
# metrics = {
#     "ema_close":ema_close , 
#     "ema_slope":ema_slope , 
#     "rsi_rsi_smoth":rsi_rsi_smoth , 
#     "rsi_slope":rsi_slope , 
#     "rsi_std":rsi_std , 
#     "rsi_thr":rsi_thr
# }

# list_combinations = []
# for i in range(1, len(metrics) + 1):
#     for lista in combinations( metrics, i ):
#         new_list = [ metrics[ elem ] for elem in lista ]
#         final_test_df = functools.reduce(lambda x,y: x & y, new_list)

#         # test1 = ema_close & ema_slope & rsi_rsi_smoth & rsi_slope & rsi_std & rsi_thr
#         test1 = (1 + (final_test_df * returns).replace(np.nan, 0).mean(axis = 1)   ).prod()
#         # print(test1)
#         list_returns["-".join(lista)] = test1
    
# list_returns = {k: v for k, v in sorted(list_returns.items(), key=lambda item: item[1], reverse = True)}


class MetricTesting(ElementwiseProblem):
    def __init__(self, assets, returns, meta = None, **kwargs):
        self.assets = assets
        self.returns = returns
        self.meta = meta
        super().__init__(n_var=kwargs["n_var"],
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=kwargs["xl"], # np.array([7,3, 40]),
                         xu=kwargs["xu"], #np.array([21,14, 90])
                         vtype=int
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        
        def rounding(i, x):
            # return round(x)
            if i in [0, 3, 5, 10, 14, 18]:
                return 1 if x > 0.5 else 0
            else:
                return round(x)

        x = [ rounding(i, xx) for i, xx in enumerate(x) ]

        out["F"] = [self.my_obj_func( x )]
    
    def my_obj_func(self, x):
        tables_to_add = []
        i = 0
        if x[i]:
            ema_close = resampling(pd.DataFrame({ a.symbol:((a.ema(x[i+1]) >= a.df["close"]).rolling(x[i+2]).sum() == 0) for a in self.assets.values()}))
            tables_to_add.append(ema_close)
        
        i = 3
        if x[i]:
            ema_slope = resampling(pd.DataFrame({ a.symbol:( a.ema(x[i+1]) > 0) for a in self.assets.values()}))
            tables_to_add.append(ema_slope)

        # i = 5
        # if x[i]:
        #     momentum = resampling(pd.DataFrame({ a.symbol:a.momentum(x[i+1]) for a in self.assets.values()}))
        #     tables_to_add.append(momentum)

        i = 5
        if x[i]:
            rsi_rsi_smoth = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth(x[i+1], x[i+2]) < a.rsi_smoth(x[i+3], x[i+4]) ) for a in self.assets.values()}))
            tables_to_add.append(rsi_rsi_smoth)

        i = 10
        if x[i]:
            rsi_slope = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth_slope(x[i+1], x[i+2], x[i+3]) > 0 ) for a in self.assets.values()}))
            tables_to_add.append(rsi_slope)

        i = 14
        if x[i]:
            rsi_std = resampling(pd.DataFrame({ a.symbol:( a.rsi_smoth(x[i+1],x[i+2]).rolling(x[i+3]).std() > 1.5 ) for a in self.assets.values()}))
            tables_to_add.append(rsi_std)

        i = 18
        if x[i]:
            rsi_thr = resampling(pd.DataFrame({ a.symbol:( (a.rsi( x[i+1] ) > x[i+2]).rolling( x[i+3] ).sum() == 0 ) for a in self.assets.values()}))
            tables_to_add.append(rsi_thr)

        de_pos = self.meta > 0
        tables_to_add.append( de_pos )

        if len(tables_to_add) == 0:
            return np.inf

        tables_to_add = functools.reduce(lambda x,y: x & y, tables_to_add)

        if tables_to_add is None or len(tables_to_add) == 0:
            return np.inf
        
        acc = (1 + (tables_to_add * self.returns).replace(np.nan, 0).mean(axis = 1)   ).prod()

        if acc == 0:
            return np.inf
        else:
            return 1/acc

@timing
def main():
    gen = 10

    n_threads = 5
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = MetricTesting(
        assets = assets,   
        returns = returns,     
        meta = de,                                           
        n_var = 22,
        xl = [0, 5, 3, 0, 5, 0, 5, 2, 5, 2, 0, 5, 2, 2, 0, 5, 2, 3, 0, 5, 45, 3],
        xu = [2, 70, 30, 2, 70, 2, 30, 15, 30, 15, 2, 30, 15, 6, 2, 30, 15, 20, 2, 30, 80, 20],
        elementwise_evaluation=True,
        elementwise_runner=runner,
    )

    algorithm = DE(
        pop_size=80,
        sampling=IntegerRandomSampling(), # LHS(),
        variant="DE/best/2/bin",
        CR=0.2,
        dither="vector",
        jitter=False
    )

    termination = get_termination("n_gen", gen)

    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

    X = res.X
    F = res.F

    def extract(x):
        x = x[0]
        if isinstance( x, float ):
            return x
        
        return extract( x )

    best_acc = extract( F )

    print("This is acc ", 1/best_acc)

    X = X.tolist()

    if isinstance( X[0], list ):
        print(X[0])
    
    else:
        print(X)
    

    data = {
            "acc": 1 / best_acc,
            "param":X
        }

    with open( f"results/offline_simulations/Batch_DE_5_{gen}.json", "w" ) as fp:
        json.dump( data, fp )
    
    pool.close()

if __name__ == "__main__":
    main()