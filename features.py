from trading.func_aux import min_max

def normalize(df, cols):

    for col in cols:
        try:
            df[col] = ( df[col] - df[col].min() ) / ( df[col].max() - df[col].min() )
        except:
            df[col] = 0

    return df

def features_1(asset, clf = False):
 
    ori_cols = asset.df.drop(columns = ["volume"]).columns

    for i in [5, 10, 15]:
        asset.df[ f"ema_{i}"] = asset.ema(i)
        asset.df[ f"roc_{i}" ] = asset.roc(i)

        for j in range(2, 9, 3):
            asset.df[ f"ema_{i}_slope_{j}" ] = asset.df[ f"ema_{i}" ].pct_change( j )
        
        for c in ["close", "high", "volume"]:
            asset.df["std{}_{}".format(c, i)] = asset.df[c].rolling(i).std()

    for i in [7, 14, 21]:
        asset.df[ f"rsi_{i}"] = asset.rsi_smoth(i, 2)
        
        for j in range(2,7, 2):
            asset.df[ f"rsi_{i}_slope_{j}" ] = asset.df[ f"rsi_{i}" ].pct_change( j )
    
    for i in [2,3,4,5,6]:
        asset.df[f"momentum_{i}"] = asset.momentum(i)
        asset.df[f"momentum_ema_{i}"] = asset.momentum(i, target = "ema_10")
        asset.df[f"momentum_rsi_{i}"] = asset.momentum(i, target = "rsi_7")

    asset.df["hl"] = asset.df["high"] - asset.df["low"]
    asset.df["ho"] = asset.df["high"] - asset.df["open"]
    asset.df["lo"] = asset.df["low"] - asset.df["open"]
    asset.df["cl"] = asset.df["close"] - asset.df["low"]
    asset.df["ch"] = asset.df["close"] - asset.df["high"]

    asset.df["buy_wf"] = asset.william_fractals(2, shift=True)
    for i in [2,3,4]:
        for j in [2,3,4]:
            asset.df[f"oneside_gaussian_filter_slope_{i}_{j}"] = asset.oneside_gaussian_filter_slope(i,j)

    asset.df["obv"] = asset.obv()

    for i in [10, 20, 30]:
        s, r = asset.support_resistance(i)
        asset.df[ f"support_{i}" ] = ( s / asset.df["close"] ) - 1
        asset.df[ f"resistance_{i}" ] = ( r / asset.df["close"] ) - 1

    # Normalization
    n_cols = list( set(asset.df.columns) - set(ori_cols) )
    
    asset.df = normalize(asset.df, cols = n_cols)

    asset.df["engulfing"] = asset.engulfing()
    asset.df["william_buy"] = asset.william_fractals(2, order = "buy").apply(lambda x : 1 if x == True else 0).rolling(5).sum()
    asset.df["william_sell"] = asset.william_fractals(2, order = "sell").apply(lambda x : 1 if x == True else 0).rolling(5).sum()

    if clf:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1).apply(lambda x: 1 if x > 0 else 0)
    else:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1)

    asset.df.drop(columns = ori_cols, inplace = True)

    return asset

def features_2(asset, clf = False):
 
    ori_cols = asset.df.drop(columns = ["volume"]).columns

    for i in [5, 10, 15]:
        asset.df[ f"ema_{i}"] = asset.ema(i)
        asset.df[ f"roc_{i}" ] = asset.roc(i)

        for j in range(2, 6, 3):
            asset.df[ f"ema_{i}_slope_{j}" ] = asset.df[ f"ema_{i}" ].pct_change( j )
        
        for c in ["close", "high", "volume"]:
            asset.df["std{}_{}".format(c, i)] = asset.df[c].rolling(i).std()

    for i in [7, 14]:
        asset.df[ f"rsi_{i}"] = asset.rsi_smoth(i, 2)
        
        for j in range(2,5, 2):
            asset.df[ f"rsi_{i}_slope_{j}" ] = asset.df[ f"rsi_{i}" ].pct_change( j )
    
    for i in [2,3,4]:
        asset.df[f"momentum_{i}"] = asset.momentum(i)
        asset.df[f"momentum_ema_{i}"] = asset.momentum(i, target = "ema_10")
        asset.df[f"momentum_rsi_{i}"] = asset.momentum(i, target = "rsi_7")

    asset.df["hl"] = asset.df["high"] - asset.df["low"]
    asset.df["ho"] = asset.df["high"] - asset.df["open"]
    asset.df["lo"] = asset.df["low"] - asset.df["open"]
    asset.df["cl"] = asset.df["close"] - asset.df["low"]
    asset.df["ch"] = asset.df["close"] - asset.df["high"]

    asset.df["buy_wf"] = asset.william_fractals(3, shift=True)
    for i in [2,3]:
        for j in [2,3]:
            asset.df[f"oneside_gaussian_filter_slope_{i}_{j}"] = asset.oneside_gaussian_filter_slope(i,j)

    asset.df["obv"] = asset.obv()

    for i in [10, 20]:
        s, r = asset.support_resistance(i)
        asset.df[ f"support_{i}" ] = ( s / asset.df["close"] ) - 1
        asset.df[ f"resistance_{i}" ] = ( r / asset.df["close"] ) - 1

    # Normalization
    n_cols = list( set(asset.df.columns) - set(ori_cols) )
    
    asset.df = normalize(asset.df, cols = n_cols)

    asset.df["engulfing"] = asset.engulfing()
    asset.df["william_buy"] = asset.william_fractals(3, order = "buy").apply(lambda x : 1 if x == True else 0).rolling(3).sum()
    asset.df["william_sell"] = asset.william_fractals(3, order = "sell").apply(lambda x : 1 if x == True else 0).rolling(3).sum()

    if clf:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1).apply(lambda x: 1 if x > 0 else 0)
    else:
        asset.df["target"] = asset.df["close"].pct_change().shift(-1)

    asset.df.drop(columns = ori_cols, inplace = True)

    return asset