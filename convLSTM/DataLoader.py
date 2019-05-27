import numpy as np
import xarray as xr
import os
class DataLoader():
    def __init__(self, datapath=None, filename="H8_Flow.nc", result="TP_2017.nc", prediction="GPM_2017.nc"):
        datapath = "../../dataset/sat_pre_video2/"
        filename="H8_Flow.nc"
        result="TP_2017.nc"
        prediction="GPM_2017.nc"
        # Load base dataset
        self.dataset = xr.open_dataset(os.path.join(datapath, filename))
        data_times = self.dataset.time[:].data
        # If we are trainign we can also load the results
        self.TP = xr.open_dataset(os.path.join(datapath, result))
        prec_times = self.TP.time[:].data

        # Find common dates
        self.times = np.intersect1d(data_times, prec_times)
        # Create geopotential normalised stack

        self.dataset = self.dataset.sel(time=self.times)
        self.TP = self.TP.sel(time=self.times)

    def __process_data(self, data):
        f


    def getRawData(self):
        return self.dataset, self.TP
    def getNumpyData(self):
        return None
    def getTimes(self):
        return self.times

    def loadAndIntercept(filepath):
        data = xr.open_dataset(filepath)
        return data.sel(time=self.times)


dl = DataLoader()
dataset, predict = dl.getRawData()

predict['crs']
predict['tp'].values.shape
results = []
for key in predict.keys():
    if key != 'crs':
        results.append(predict[key].values)
results2 = np.vstack(results)
#%%[]
if len(results2.shape) == 3:
    results2 = result2[:, np.newaxis]
