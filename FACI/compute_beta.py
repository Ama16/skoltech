import numpy as np


def findBeta(recentScores, curScore, epsilon=0.001):
    top = 1
    bot = 0
    mid = (top + bot) / 2
    while top - bot > epsilon:
        if np.quantile(recentScores, 1-mid) > curScore:
            bot = mid
            mid = (top + bot) / 2
        else:
            top = mid
            mid = (top + bot) / 2
    return mid


def computeBetasByGeoByTime(scores, lookback=100, epsilon=0.001, geosToUse=None, col2use="mae"):
    dates = sorted(np.unique(scores["timestamp"]))
    geovals = np.unique(scores["segment"])
    if geosToUse is None:
        geosToUse = geovals
    
    betaSeqMat = np.zeros((len(geosToUse), len(dates)-lookback))
    for t in range(lookback, len(dates)):
        prevScores = scores[scores["timestamp"] < dates[t]][col2use].values
        for i in range(len(geosToUse)):
            newScore = scores[(scores["timestamp"] == dates[t]) & (scores["segment"] == geosToUse[i])][col2use].values
            if len(newScore) > 0:
                betaSeqMat[i][t-lookback] = findBeta(prevScores, newScore, epsilon)
            else:
                betaSeqMat[i][t-lookback] = None
    return betaSeqMat


def computeConfInt(scores, alphas, col2use="mae"):
    alphas = [1 - max(0, i) for i in alphas]
    lookback = len(scores) - len(alphas)
    dates = sorted(np.unique(scores["timestamp"]))
    
    ConfScore = np.zeros((2, len(dates)-lookback))
    for t in range(lookback, len(dates)):
        prevScores = scores[scores["timestamp"] < dates[t]][col2use].values

        quntile_value = np.quantile(prevScores, q=alphas[t-lookback])
        forecast_now = scores[scores["timestamp"] == dates[t]]["forecast"].values[0]
        if col2use == "mae":
            ConfScore[0][t-lookback] = forecast_now - quntile_value
            ConfScore[1][t-lookback] = forecast_now + quntile_value
        elif col2use == "mape":
            if quntile_value < 1:
                ConfScore[0][t-lookback] = forecast_now / (1 + quntile_value)
                ConfScore[1][t-lookback] = forecast_now / (1 - quntile_value)
            else:
                ConfScore[0][t-lookback] = forecast_now / (1 + quntile_value)
                ConfScore[1][t-lookback] = np.inf
        else:
            raise "NotImplementedError"
            
    return ConfScore