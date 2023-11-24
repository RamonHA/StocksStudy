import warnings
warnings.filterwarnings("ignore")

from trading.func_aux import timing
from trading.processes import Simulation

import multiprocessing as mp
from datetime import date
import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor

from features import *
from metaheuristics import *

@timing
def main():
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
           
            "DE_5_svm_integer":{
                "type":"prediction",
                "time":360,
                "function":metaheuristic_integer,
                "offline": True,
            }

        },
        run = True,
        cpus = 6
    )


if __name__ == "__main__":
    main()