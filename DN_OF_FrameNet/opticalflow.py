"""
    opticalflow.py

    Functions to calculate the optical flow and warping of images
"""
import numpy as np
import cv2
import xarray as xr
import argparse
from denoising import denoise_pool, denoise

def normalised_image(img):
    print(np.ptp(img))
    return np.round(255 * ((img - np.amin(img)) / (np.ptp(img)+1e-10))).astype(np.uint8)

def optical_flow_2frames(prev_frame, next_frame):
    base = normalised_image(prev_frame)
    frameGray = normalised_image(next_frame)
    base = denoise(base)
    frameGray = denoise(frameGray)
    flow = cv2.calcOpticalFlowFarneback(base, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow *= -2
    return flow

# warm image according to optical flow
def warp_flow(img, flow):

    h, w = flow.shape[:2]
    flow[:, :, 0] *= 2.0
    flow[:, :, 1] *= 2.0
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return res

# Calculate the optical flow for each layer and creates an image wrapped.
def warp_image_all_layers(prev,next):
    channels = next.shape[-1]
    result = np.zeros(next.shape)
    for i in range(channels):
        flow = optical_flow_2frames(prev[:,:,i],next[:,:,i])
        result[:,:,i]= warp_flow(next[:,:,i], flow)
    return result


def draw_flow(img, flow, step=8):
    """
    Taken from: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    :param img: Frame to draw the flow
    :param flow: Flow vectors
    :param step: Number of pixels each vector represents
    :return: visualisation
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_, _) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def optical_flow(a, silence, denoising):
    '''
    Structure Taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    To create dense optical on all the pixels
    '''
    # read in data

    dataset = np.round(255 * (a - np.min(a)) / np.ptp(a).astype(int)).astype(np.uint8)
    if denoising:
        dataset = denoise_pool(dataset)
    gray = base = dataset[0, :, :]
    # apply denoising before creating optical flow
    # gray = cv2.fastNlMeansDenoising(gray,None,2,7)
    height, width = gray.shape
    if not silence:
        cv2.imshow('frame', gray)
        cv2.waitKey()

    hsv = np.zeros_like(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
    hsv[..., 1] = 255

    # initialize array lists to store sin, cos, magnitude form the optical flow result on each pixel
    x = []
    y = []

    for i in range(1, len(dataset)):
        frameGray = dataset[i, :, :]
        x, y = frameGray.shape

        flow = cv2.calcOpticalFlowFarneback(gray, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow *= -2

        x.append(flow[:, :, 0])
        y.append(flow[:, :, 1])

        # display the result
        img = draw_flow(cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR), flow)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        numpy_horizontal = np.hstack((cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR), img, rgb))

        if not silence:
            cv2.imshow('frame', numpy_horizontal)
            cv2.waitKey()

        gray = frameGray

    # cap.release()
    cv2.destroyAllWindows()
    return x,y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optical Flow algorithm')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    parser.add_argument('layer',
                        help='layer to perform the optical flow, should be in the input dataset ex(B7)', type=str)
    parser.add_argument("-s", "--silence", help="Show the images ", action="store_true", dest="verbose")
    parser.add_argument("-d", "--denoising", help="perform denosing ", action="store_true", dest="denoising")

    args = parser.parse_args()

    cap = xr.open_dataset("../../dataset/H8_Flow.nc")  # H8_Flow.nc

    a = np.asarray(cap[args.layer])

    x, y = optical_flow(a, args.verbose, args.denoising)
    # Save x,y

    dx = xr.Dataset({})
    dx['x'] = xr.DataArray(x, dims=['time', 'width', 'height'])
    dx['y'] = xr.DataArray(y, dims=['time', 'width', 'height'])

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dx.data_vars}
    dx.to_netcdf(args.output, mode='w', format='NETCDF4', engine='h5netcdf', encoding=encoding)
