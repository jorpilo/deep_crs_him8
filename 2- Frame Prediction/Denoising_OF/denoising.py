import cv2
import numpy as np
import xarray as xr
from multiprocessing import Pool
import argparse


# implement fastmeansdenoising
# source link1: https://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html
# source link2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# We choose a small number for the size of template winodw here to reserve the details of the clouds
# while implementing the denoising
def denoise(img):
    dst = cv2.fastNlMeansDenoising(img,None,3,7)
    return dst

# implememt erosion on denoised image for data enhancement
# source link: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    return erosion

def denoise_pool(dataset):
    print("Starting threads")
    with Pool(4) as p:
        result = p.map(denoise, dataset)
    return result


def denoise_many(ds):
    dx = xr.Dataset({})
    for number in ds:
        if number != "crs":
            a = ds[number].values
            dataset = np.round(255 * (a - np.min(a)) / np.ptp(a).astype(int)).astype(np.uint8)
            result = denoise_pool(dataset)
            print("Processes done for " + number)
            result = np.array(result, dtype='uint8')
            dx[number] = xr.DataArray(result, dims=['time', 'width', 'height'])
    return dx


## Main
if __name__ == "__main__":
    # Arguments:
    parser = argparse.ArgumentParser(description='Implement denoising on a dataset')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    args = parser.parse_args()

    # Load dataset
    dataset = xr.open_dataset(args.filename)

    # Perform denoise
    dx = denoise_many(dataset)

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dx.data_vars}
    dx.to_netcdf(args.output, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
