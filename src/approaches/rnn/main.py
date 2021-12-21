import torch
from castingcurvemodel import LstmRNN

def ReadData():
    datasetFolder = "../../../Datasets"


# hz
sampleRate = 4


if __name__ == '__main__':
    
    print ('now __name__ is %s' %__name__)

    timeLimitForEveryStep = list(1,2,3,4,5,6)

    # attr for every stage LSTM.
    attrs = list(
        list(1,2,4,5),
        list(1,2,3,4),
        list(1,2,4,5),
        list(4,5,6,7),
        list(1,2,3,4),
        list(1,2,3,4)
    )
    
    # first layer
    for stepLimit in timeLimitForEveryStep:
        attrs.append(list(1,2,3,4))
    

    # second layer
    attrs.append(list(1,2,3,4))

    # third layer
    lstm_model = LstmRNN(4, 3200, 2)

    # pipeline


