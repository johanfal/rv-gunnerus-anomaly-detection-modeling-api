import numpy as np
from sklearn.preprocessing import StandardScaler
import progressbar
import pandas as pd

def create(data, trainPct):
    X, y, timesteps = transform(data, trainPct)

    X_train, y_train = reshapeData(X.head(10000), y.head(10000), timesteps)
    X_test, y_test = reshapeData()

def transform(data, trainPct=0.8):
    """Description."""
    # Create training and testing sets
    testPct = 1 - trainPct
    trainSize = int(np.ceil(data.shape[0] * trainPct))
    testSize = int(np.floor(data.shape[0] * testPct))
    dfTrain, dfTest = data.iloc[0:trainSize], data.iloc[trainSize:]
    print(dfTrain.shape, dfTest.shape)
    # Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(dfTrain[dfTrain.columns])

    arrTrain = scaler.transform(dfTrain)
    arrTest = scaler.transform(dfTest)
    colCounter = 0
    for col in dfTrain.columns:
        dfTrain[col] = [i[colCounter] for i in arrTrain]
        dfTest[col] = [i[colCounter] for i in arrTest]
        colCounter += 1

    timeSteps = 100
    return dfTrain[list(dfTrain.columns)], dfTrain, timeSteps

    # XTrain, yTrain = reshapeData(
    #     dfTrain[list(dfTrain.columns)],
    #     dfTrain,
    #     timeSteps
    # )

def reshapeData(X, y, timeSteps = 1):
    """Description."""

    Xs, ys = [], [] # empty placeholders

    rangeMax = len(X)-timeSteps # iteration range

    bar = getBar(rangeMax).start() # progress bar

    for i in range(rangeMax):
        bar.update(i+1)
        v = X.iloc[i:(i + timeSteps)].values
        Xs.append(v)
        ys.append(y.iloc[i + timeSteps])
    bar.finish()
    return np.array(Xs), np.array(ys)


def getBar(rangeMax):
    return progressbar.ProgressBar(maxval=rangeMax, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

if __name__ == '__main__':
    pbar()
    import sys
    sys.exit('Run from manage.py, not model.')