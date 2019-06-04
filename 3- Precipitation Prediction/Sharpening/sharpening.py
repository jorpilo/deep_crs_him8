import cv2
import numpy as np
import xarray as xr
from multiprocessing import Pool
import argparse

def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Taken from https://github.com/soroushj/python-opencv-numpy-example/blob/master/unsharpmask.py
    Return a sharpened version of the image, using an unsharp mask.
    """
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def sharpen_batch(batch):
    with Pool(8) as p:
        result = p.map(sharpen(), batch)
    result = np.asarray(result)

    return result


def sharpen_many(input_filename, output_filename):
    ds = xr.open_dataset(input_filename)
    dx = xr.Dataset({})
    for number in ds:
        if number != "crs":
            a = np.asarray(ds[number])
            dataset = np.round(255 * (a - np.min(a)) / np.ptp(a).astype(int)).astype(np.uint8)
            print("Starting threads")
            with Pool(8) as p:
                result = p.map(sharpen, dataset)
            print("Processes done for " + number)
            result = np.array(result, dtype='uint8')
            dx[number] = xr.DataArray(result, dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dx.data_vars}
    dx.to_netcdf(output_filename, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)


## Main
if __name__ == "__main__":
    # Arguments:
    parser = argparse.ArgumentParser(description='Implement sharpening on a dataset')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    args = parser.parse_args()

    # Load dataset
    dataset = xr.open_dataset(args.input, args.output)

    # Perform sharpening
    sharpen_many(dataset)
