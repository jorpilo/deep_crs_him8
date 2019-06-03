# Optical flow implementation for 2 frames

import cv2
import numpy as np


def optical_flow_2frames(prev_frame, next_frame):
    base = normalised_image(prev_frame)
    frameGray = normalised_image(next_frame)
    base = cv2.fastNlMeansDenoising(base, None, 2, 7)
    frameGray = cv2.fastNlMeansDenoising(frameGray, None, 2, 7)
    flow = cv2.calcOpticalFlowFarneback(base, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow *= -2
    return flow

def normalised_image(img):
    print(np.ptp(img))
    return np.round(255 * ((img - np.amin(img)) / (np.ptp(img)+1e-10))).astype(np.uint8)

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