import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr


fig = plt.figure()
ds = xr.open_dataset("../../sat_precip/H8_Flow.nc")
print(ds)
print(ds.time)
#print(ds["B7"][:,:,:].min())
#print(ds["B7"][:,:,:].max())
#sys.exit()
data = np.asarray(ds["B7"])

data.shape
ims = []
for i in range(len(ds["B7"])):
    im = plt.imshow(ds["B7"][i,:,:], animated=True)
    ims.append([im])


ani.save('cloudsB7.mp4')
