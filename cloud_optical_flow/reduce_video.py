import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr


#fig = plt.figure()
ds = xr.open_dataset("../../dataset/HIM8_2017.nc")

#ds = xr.open_dataset("sort.nc")
#ds

ds = ds.drop(['B8','B10', 'B12', 'B13', 'B14','B15'])
print(ds)

comp = dict(zlib=True, complevel=9)
encoding = {var: comp for var in ds.data_vars}


ds.to_netcdf('sort.nc',mode='w', format='NETCDF4', engine='h5netcdf',encoding=encoding)
"""
ds
print(ds.time)
#print(ds["B7"][:,:,:].min())
#print(ds["B7"][:,:,:].max())
#sys.exit()

ims = []
for i in range(len(ds["B7"])):
    im = plt.imshow(ds["B7"][i,:,:], animated=True)
    ims.append([im])
print(len(ims))
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

ani.save('cloudsB7.mp4')
"""
