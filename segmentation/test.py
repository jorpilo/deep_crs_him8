import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation
import xarray as xr

if __name__=="__main__":
    ds = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")
    print(ds)
    print(len(ds))
    print(len(ds['B7']))
    print(ds['B7'].shape)