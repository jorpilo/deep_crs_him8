import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr


fig = plt.figure()
ds = xr.open_dataset("../../dataset/sat_pre_video2/sortH8_2017.nc")

ds
#ds = xr.open_dataset("sort.nc")
#ds


ds
print(ds.time)
#print(ds["B7"][:,:,:].min())
#print(ds["B7"][:,:,:].max())
#sys.exit()

ims = []
for i in range(100):
    im = plt.imshow(ds["B7"][i,:,:], animated=True)
    ims.append([im])
    print(i)
print(len(ims))
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

ani.save('cloudsB7.mp4')
