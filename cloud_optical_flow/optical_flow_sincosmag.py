"""
Created on Tue May 14 15:09:12 2019
@author: RMC
"""
import numpy as np
import cv2
import xarray as xr

# The video feed is read in as a VideoCapture object
def main():
    # Load video1
    #cap = xr.open_dataset("../dataset/sat_pre_video1/H8_Flow.nc")  # H8_Flow.nc


    # Load video2
    cap = xr.open_dataset("../dataset/sat_pre_video2/HIM8_2017.nc")  # H8_Flow.nc
    # We need to intercept frames if using video2
    cap2 = xr.open_dataset("../dataset/sat_pre_video2/TP_2017.nc")  # H8_Flow.nc
    cap_time = cap.time[:].data
    cap2_time = cap2.time[:].data
    times = np.intersect1d(cap_time, cap2_time)

    a = cap.B11.sel(time=times)[:].data

    lkParams = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    dataset = np.round(255 * (a - np.min(a)) / np.ptp(a).astype(int)).astype(np.uint8)

    gray = dataset[0, :, :]
    # gray = cv2.fastNlMeansDenoising(gray,None,2,7)

    sinarray = []
    cosarray = []
    magarray = []
    print (dataset.shape)
    for i in range(1, len(dataset)):
        frameGray = dataset[i, :, :]

        frameGray = cv2.fastNlMeansDenoising(frameGray,None,2,7)

        flow = cv2.calcOpticalFlowFarneback(gray, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        theta = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        mag = np.sqrt(np.power(flow[:, :, 1], 2) + np.power(flow[:, :, 0], 2))

        sinarray.append(sinTheta)
        cosarray.append(cosTheta)
        magarray.append(mag)

        gray = frameGray

    # cap.release()
    return np.stack(sinarray), np.stack(cosarray), np.stack(magarray)


if __name__ == "__main__":
    filename = "../dataset/sat_pre_video2/opticalflow.nc"

    sin, cos, mag = main()
    ds = xr.Dataset({})

    ds['sin'] = xr.DataArray(sin, dims=['time', 'width', 'height'])
    ds['cos'] = xr.DataArray(sin, dims=['time', 'width', 'height'])
    ds['mag'] = xr.DataArray(sin, dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}

    ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)


    print(ds)