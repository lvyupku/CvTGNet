from .model import *

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'CvtGNet':
        m = getCvtGNet(config=config)
    elif model_type == 'CvtG':
        m = getCvtG(config=config)
    elif model_type == 'Cvt':
        m = getCvt(config=config)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return m
