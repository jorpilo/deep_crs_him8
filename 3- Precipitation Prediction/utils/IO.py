import numpy as np
import os

def load_dataset(path, layers, normalize=True):

    res = []
    for layer in layers:
        fullpath = os.path.join(path, "b"+str(layer)+"_30.npy")
        b = np.load(fullpath)
        if normalize:
            b = (b - b.mean()) / b.std()
        res.append(b)
    res = np.stack(res, axis=3)
    return res