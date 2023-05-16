import numpy as np

def split(data, target, rate):
    mask = np.random.uniform(size=data.shape[0])
    return data[mask <= rate], target[mask <= rate], data[mask > rate], target[mask > rate]