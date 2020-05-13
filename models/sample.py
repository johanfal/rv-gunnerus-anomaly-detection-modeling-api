import numpy as np
from sklearn.preprocessing import StandardScaler

def create(data, trainPct=0.8):
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

    timeSteps = 100

    XTrain, yTrain = createSubsequence(
        dfTrain,
    )

    print("Hello world!")

def createSubsequence(X, y, timeSteps = 1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)



if __name__ == '__main__':
    import sys
    sys.exit('Run from manage.py, not model.')