import pandas as pd

def batch_analysis(assets):
    

    df = pd.DataFrame()
    for asset in assets:
        df = pd.concat([ df, asset.df[ "buy" ] ], axis = 1)

    
def test(broker = "gbm"):
    
    from trading.func_brokers import get_assets

    

    return 