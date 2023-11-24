import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import time
from datetime import date
import pandas as pd
import numpy as np

# from pymoo.algorithms.so_de import DE
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from trading.metaheuristics.ta_tunning import TATunningProblem
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import StarmapParallelization


from multiprocessing.pool import ThreadPool


from trading.func_aux import dropna, min_max

GEN = 3

class TATunningProblem(ElementwiseProblem):

    def __init__(
            self,
            asset,
            regr,
            n_var = 15,
            xu = 50,
            xl = 3,
            places = 1,
            normalize = True,
            seasonality = False,
            error = "relative",
            verbose = 0,
            n_obj = 1,
            n_constr = 0,
            type_var = np.int,
            train_size = None,
            **kwargs
        ):
        """  
            func (callable): function to minimize 
        """

        self.places = places
        self.normalize = normalize
        self.seasonality = seasonality
        self.verbose = verbose

        self.asset = asset
        self.regr = regr
        self.error_func = error

        self.train_size = train_size

        super().__init__(
            n_var=n_var, 
            n_obj=n_obj,
            n_constr=n_constr, 
            xl=xl, 
            xu=xu, 
            type_var=type_var ,
            elementwise_evaluation=True, 
            **kwargs
        )

    @property
    def error_func(self):
        return self._error_func
    
    @error_func.setter
    def error_func(self, value):
        if isinstance(value, str):
            self._error_func = {
                "relative":self.relative
            }[ value ]
        elif callable(value):
            self._error_func = value

    @property
    def asset(self):
        return self._asset
    
    @asset.setter
    def asset(self, value):
        from trading.assets import Asset
        assert issubclass( type(value), Asset ), "Asset must be an Assert type"

        if "target" not in value.df.columns:
            value.df["target"] = value.df[ "close" ].pct_change(periods = self.places).shift(-self.places)

        self._asset = value

    def _evaluate(self, x, out, *args, **kwargs):
        if self.verbose > 0: print( "- Evaluation" )
        x = x.astype(int)
        # out["F"] = [ self.objective_function(i) for i in x]    
        out["F"] = [self.objective_function(x)]

    def objective_function(self, vector):
        if self.verbose > 0: print("- Objective Function")
        self.update_ta(vector)

        predict = self.predict()
        if predict is None: 
            if self.verbose > 1: print("-- Predict is None")
            return np.inf

        error = self.error(predict) 

        if self.verbose > 1: print("-- Error is ", error)

        return error if error is not None else np.inf

    def update_ta(self, vector):
        """ Modified the asset object from the new vector parameters """

        if self.verbose > 0: print("- Update TA")

        try:
            self._update_ta(vector=vector)
        except Exception as e:
            if self.verbose > 1: print("Update ta exception: {} \n  Vector: {} \n {}".format( e, vector.shape, vector ))

    def _update_ta(self, vector):

        self.asset.df["ema"] = self.asset.ema( vector[0] ).pct_change(2)

        self.asset.df["wma"] = self.asset.wma( vector[1] ).pct_change(2)

        self.asset.df["dema"] = self.asset.ema( vector[2] ).pct_change(2)

        self.asset.df["cci"] = self.asset.cci( vector[3] )

        self.asset.df["sma"] = self.asset.sma( vector[4] ).pct_change(2)

        self.asset.df["stoch"], _ = self.asset.stoch( vector[5], 3 )

        self.asset.df["williams"] = self.asset.william( vector[6] )

        self.asset.df["force_index"] = self.asset.force_index( vector[7] )

        self.asset.df["rsi"] = self.asset.rsi_smoth( vector[8], vector[9] )

        self.asset.df["rsi_slope"] = self.asset.rsi_smoth_slope(vector[10], vector[11], 3)

        self.asset.df[ "rsi_std" ] = self.asset.rsi_smoth(vector[12],vector[13]).rolling(vector[14]).std()

        resistance, support = self.asset.support_resistance( vector[15] )
        self.asset.df["sr"] = (self.asset.df["close"] - support)/( resistance - support )

    def train_test(self,df ):

        df = dropna(df)

        if 0 in df.shape: return None, None

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) < ( 20 ): return None, None

        # self.test_size = int( len(df)*0.2 )      # self.places
        # self.train_size = len(df) - self.test_size   # self.places

        if self.train_size is None:
            train = df.loc[ :date(2019, 1, 1,) ]
            test = df.loc[ date(2019, 1, 1,): ]
        else:
            train_size = int( len(df) * self.train_size )
            train = df.iloc[:train_size]
            test = df.iloc[ train_size: ]

        return train, test

    def train_predict(self,df ):
        aux = df.tail(1)

        if len( aux.drop(columns = ["target"]).replace([ np.inf, -np.inf ], np.nan).dropna() ) == 0: return None, None

        train, _ = self.train_test(df)
        return train, aux

    def predict(self, for_real = False):
        if self.verbose > 0: print("- Predict")

        if self.normalize:
            df = min_max( self.asset.df, exception=["target"] )
        else:
            df = self.asset.df

        if df is None or len(df) == 0:
            return None

        if self.verbose > 1: print("-- Len of df.dropna in Predict: ", len(df.dropna()))

        self.train, self.test = self.train_test(df) if not for_real else self.train_predict(df)

        if self.train is None or len(self.train) == 0 or len(self.test) == 0: return None

        if self.verbose > 1: print("Len of train {} - Len of test {}".format( len(self.train), len(self.test) ))

        self.regr = self.regr.fit(self.train.drop(columns = "target"), self.train["target"])
        
        predict = self.regr.predict(self.test.drop(columns = "target"))

        return predict

    def rollback(self, predicted):

        real = pd.merge( self.test[["target"]], self.asset.df, left_index=True, right_index=True, how = "left" )[["close"]].reset_index(drop = True)

        predicted += 1
        
        predicted = pd.DataFrame(predicted)
        predicted.columns = ["close"]

        predicted = pd.concat([pd.DataFrame(real.head(1)), predicted], axis = 0)
        predicted = predicted.cumprod()
        predicted = predicted.iloc[1:]
        predicted.reset_index(drop = True, inplace = True)
        
        return  predicted

    def error(self, predicted):
        if self.verbose > 0: print("- Error")
        y_true = pd.merge( self.test[["target"]], self.asset.df, left_index=True, right_index=True, how = "left" )[["close"]].reset_index(drop = True)
        
        predicted = pd.DataFrame( self.rollback( predicted ) )
        predicted = predicted.iloc[ :-1 ]  
        predicted.reset_index(drop = True, inplace = True)

        y_true = y_true.iloc[1:]
        y_true.reset_index(drop = True, inplace = True)

        if (len(y_true) != len(predicted)) or len(y_true) == 0:
            return None

        return self.error_func( y_true, predicted )

    def relative(self, y_true, y_pred):
        if self.verbose > 1: print( "--- Relative Error. \nThe y_true is: {}\nAnd y_predict: {} ".format( y_true, y_pred ) )
        aux = abs( y_true - y_pred ) / y_true
        aux = aux.sum() / len(aux)
        return aux[ 0 ]

class SimpleTunning(TATunningProblem):

    def _update_ta(self, vector):

        self.asset.df["ema"] = self.asset.ema( vector[0] ).pct_change(2)

        self.asset.df["wma"] = self.asset.wma( vector[1] ).pct_change(2)

        self.asset.df["dema"] = self.asset.ema( vector[2] ).pct_change(2)

        self.asset.df["cci"] = self.asset.cci( vector[3] )

        self.asset.df["sma"] = self.asset.sma( vector[4] ).pct_change(2)

        self.asset.df["stoch"], _ = self.asset.stoch( vector[5], 3 )

        self.asset.df["williams"] = self.asset.william( vector[6] )

        self.asset.df["force_index"] = self.asset.force_index( vector[7] )

        self.asset.df["rsi"] = self.asset.rsi_smoth( vector[8], vector[9] )

        self.asset.df["rsi_slope"] = self.asset.rsi_smoth_slope(vector[10], vector[11], 3)

        self.asset.df[ "rsi_std" ] = self.asset.rsi_smoth(vector[12],vector[13]).rolling(vector[14]).std()


def metaheuristic(inst):
    
    inst.df["vpt"] = inst.vpt()
    inst.df["buy_wf"] = inst.william_fractals(3, shift=True)
    inst.df["sell_wf"] = inst.william_fractals(3, shift=True, order = "sell").rolling(3).sum()
    inst.df["oneside_gaussian_filter_slope"] = inst.oneside_gaussian_filter_slope(4,3)

    inst.df.dropna(inplace = True)

    if len(inst.df) == 0: return None

    algorithm = DE(
        pop_size = 80,
        variant="DE/best/2/bin",
        CR = 0.8,
        F = 0.1,
        dither = "vector",
    )

    problem = SimpleTunning(
        asset = inst,
        regr = SVR(),# RandomForestRegressor(n_estimators = 150),
        xl = [3, 3, 3, 7, 3, 7, 2, 2, 7, 2, 7, 2, 5, 2, 2 ],
        xu = [36, 36, 36, 36, 36, 36, 8, 20, 29, 29, 29, 29, 29, 10, 10],
        verbose = 0,
        n_var = 15,
        normalize = True,
        error = mean_squared_error
    )

    # try:
    res = minimize(
        problem,
        algorithm,
        ("n_gen", GEN),
        seed = 1,
        verbose = False
    )
    # except Exception as e:
    #     print("{} with exception {}".format( inst, e ))
    #     return None

    problem.update_ta( res.X.astype(int) )

    predict = problem.predict(for_real = True)

    aux = predict[-1] if predict is not None else None

    return aux

def metaheuristic_integer(inst):
    
    inst.df["vpt"] = inst.vpt()
    inst.df["buy_wf"] = inst.william_fractals(3, shift=True)
    inst.df["sell_wf"] = inst.william_fractals(3, shift=True, order = "sell").rolling(3).sum()
    inst.df["oneside_gaussian_filter_slope"] = inst.oneside_gaussian_filter_slope(4,3)

    inst.df.dropna(inplace = True)

    if len(inst.df) == 0: return None

    algorithm = DE(
        pop_size = 80,
        variant="DE/best/2/bin",
        CR = 0.8,
        F = 0.1,
        dither = "vector",
        sampling=IntegerRandomSampling()
    )

    problem = SimpleTunning(
        asset = inst,
        regr = SVR(),# RandomForestRegressor(n_estimators = 150),
        xl = [3, 3, 3, 7, 3, 7, 2, 2, 7, 2, 7, 2, 5, 2, 2 ],
        xu = [36, 36, 36, 36, 36, 36, 8, 20, 29, 29, 29, 29, 29, 10, 10],
        verbose = 0,
        n_var = 15,
        normalize = True,
        error = mean_squared_error
    )

    # try:
    res = minimize(
        problem,
        algorithm,
        ("n_gen", GEN),
        seed = 1,
        verbose = False
    )
    # except Exception as e:
    #     print("{} with exception {}".format( inst, e ))
    #     return None

    problem.update_ta( res.X.astype(int) )

    predict = problem.predict(for_real = True)

    aux = predict[-1] if predict is not None else None

    return aux

def metaheuristic_integer_batch(inst, gen, verbose = True, train_size = None):
    print(inst)
    if inst is None or len(inst.df) == 0 or "close" not in inst.df.columns:
        print("No info")
        return []

    inst.df["vpt"] = inst.vpt()
    inst.df["buy_wf"] = inst.william_fractals(3, shift=True)
    inst.df["sell_wf"] = inst.william_fractals(3, shift=True, order = "sell").rolling(3).sum()
    inst.df["oneside_gaussian_filter_slope"] = inst.oneside_gaussian_filter_slope(4,3)

    inst.df.dropna(inplace = True)

    if len(inst.df) == 0: return []

    # n_threads = 6
    # pool = ThreadPool(n_threads)
    # runner = StarmapParallelization(pool.starmap)

    algorithm = DE(
        pop_size = 80,
        variant="DE/best/2/bin",
        CR = 0.8,
        F = 0.1,
        dither = "vector",
        sampling=IntegerRandomSampling()
    )

    problem = TATunningProblem(
        asset = inst,
        regr = SVR(),# RandomForestRegressor(n_estimators = 150),
        xl = [3, 3, 3, 7, 3, 7, 2, 2, 7, 2, 7, 2, 5, 2, 2, 5 ],
        xu = [36, 36, 36, 36, 36, 36, 8, 20, 29, 29, 29, 29, 29, 10, 10, 50],
        verbose = 0,
        n_var = 16,
        normalize = True,
        error = mean_squared_error,
        train_size=train_size
        # elementwise_runner=runner,
    )

    # try:
    res = minimize(
        problem,
        algorithm,
        ("n_gen", gen),
        seed = 1,
        verbose = verbose
    )
    # except Exception as e:
    #     print("{} with exception {}".format( inst, e ))
    #     return None

    problem.update_ta( res.X.astype(int) )

    predict = problem.predict(for_real = True)

    return predict.tolist() if predict is not None else []