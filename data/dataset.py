import torchxrayvision as xrv
import pandas as pd
from tools import *


def GetsubData(dataset, path):
    subset = pd.read_csv(path)
    
    s = subset['Image Index'].tolist()
    a = dataset.csv['Image Index'].tolist()
    idx = np.where([i in s for i in a])
    idx = idx[0]
        
    return xrv.datasets.SubsetDataset(dataset, idx)
    

def GetNIHData(config, transform=None, isTrain=True):
    dataset = xrv.datasets.NIH_Dataset(imgpath=config.DATA.DATA_PATH, csvpath=config.DATA.TRAIN_SET, views=["PA","AP"], unique_patients=False, transform=transform)

    if isTrain:
        return GetsubData(dataset=dataset, path=config.DATA.TRAIN_CSV)
    
    validset = GetsubData(dataset=dataset, path=config.DATA.VALID_CSV)
    testset = GetsubData(dataset=dataset, path=config.DATA.TEST_CSV)

    return validset, testset

def GetCheXData(config, transform=None, isTrain=True, value=0):
    if isTrain:
        dataset = xrv.datasets.CheX_Dataset(imgpath=config.DATA.TRAIN_SET, 
                                            csvpath=config.DATA.TRAIN_CSV, 
                                            views=["*"],
                                            unique_patients=False,
                                            transform=transform)
        dataset.labels = fillna(dataset.labels, value)
        return dataset
    else:
        dataset = xrv.datasets.CheX_Dataset(imgpath=config.DATA.VALID_SET, 
                                            csvpath=config.DATA.VALID_CSV, 
                                            views=["*"],
                                            unique_patients=False,
                                            transform=transform)
        dataset.labels = fillna(dataset.labels, value)
    return (dataset, dataset)


