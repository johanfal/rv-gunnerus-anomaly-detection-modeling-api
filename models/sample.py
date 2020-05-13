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

    dfTrain2 = dfTrain.filter(['ME1_EngineSpeed'])
    dfTest2 = dfTest.filter(['ME1_EngineSpeed'])

    colCounter = 0
    scaler2 = StandardScaler()
    for col in dfTrain.columns:
        scaler3 = scaler2.fit(dfTrain[[col]])

        cmparrTrain = np.array([[i[colCounter]] for i in arrTrain])
        cmparrTest = np.array([[i[colCounter]] for i in arrTest])
        arrTrain3 = scaler3.transform(dfTrain[[col]])
        arrTest3 = scaler3.transform(dfTest[[col]])

        print(col, 'train', np.array_equal(cmparrTrain, arrTrain3))
        print(col, 'test', np.array_equal(cmparrTest, arrTest3))
        colCounter += 1

    scaler2 = scaler.fit(dfTrain2[['ME1_EngineSpeed']])
    arrTrain2 = scaler.transform(dfTrain2[['ME1_EngineSpeed']])
    arrTest2 = scaler.transform(dfTest2[['ME1_EngineSpeed']])

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