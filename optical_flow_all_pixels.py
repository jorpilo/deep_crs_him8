"""
Created on Tue May 14 15:09:12 2019
@author: RMC
"""
import numpy as np
import cv2
import xarray as xr


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

def main():
    '''
    Structure Taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    To create dense optical on all the pixels 
    '''
    # read in data 
    cap = xr.open_dataset("../../dataset/H8_Flow.nc") #H8_Flow.nc
   
    a = np.asarray(cap['B7'])
    lkParams=dict( winSize =(15,15),
                      maxLevel=2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    dataset = np.round(255*(a - np.min(a))/np.ptp(a).astype(int)).astype(np.uint8)

    gray = base = dataset[0,:,:]
    # apply denoising before creating optical flow 
    #gray = cv2.fastNlMeansDenoising(gray,None,2,7)
    height, width = gray.shape

    cv2.imshow('frame',gray)
    cv2.waitKey()

    hsv = np.zeros_like(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
    hsv[..., 1] = 255
    
    # initialize array lists to store sin, cos, magnitude form the optical flow result on each pixel 
    sinarray = []
    cosarray = []
    magarray = []

    for i in range(1,len(dataset)):
        frameGray = dataset[i,:,:]
        x, y= frameGray.shape    
    
        flow = cv2.calcOpticalFlowFarneback(gray, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow *= -2
        
        # compute cos, sin and magnitude for the optical flow on each pixel 
        theta = np.arctan2(flow[:,:,1], flow[:,:,0])
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)
        mag = np.sqrt(np.power(flow[:,:,1],2)+np.power(flow[:,:,0],2))

        sinarray.append(sinTheta)
        cosarray.append(cosTheta)
        magarray.append(mag)
        
         #display the result 
        img = draw_flow(cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR), flow)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        numpy_horizontal = np.hstack((cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR),img, rgb))

        cv2.imshow('frame',numpy_horizontal)
        cv2.waitKey()

        gray = frameGray

    #cap.release()
    cv2.destroyAllWindows()
    return np.stack(sinarray), np.stack(cosarray), np.stack(magarray)

if __name__ == "__main__":
   sin, cos, mag = main()
   print(sin.shape)