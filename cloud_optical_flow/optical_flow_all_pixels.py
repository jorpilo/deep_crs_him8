"""
Created on Tue May 14 15:09:12 2019

@author: RMC
"""
import numpy as np
import cv2
import xarray as xr


# The video feed is read in as a VideoCapture object
cap = xr.open_dataset("D:/dataset/H8_Flow.nc")

a = np.asarray(cap['B7'])
lkParams=dict( winSize =(15,15),
                  maxLevel=2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
dataset = np.round(255*(a - np.min(a))/np.ptp(a).astype(int)).astype(np.uint8)

gray = base = dataset[0,:,:]
height, width = gray.shape

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
cv2.imshow('frame',gray)
cv2.waitKey()

mask = np.zeros_like(gray)
for i in range(1, len(dataset)):
    frameGray = dataset[i,:,:]
    
    all_pixels = np.nonzero(frameGray)[::-1]
    all_pixels = tuple(zip(*all_pixels))
    all_pixels = np.vstack(all_pixels).reshape(-1, 1, 2).astype("float32")
       
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray, frameGray, all_pixels, None, **lkParams)
    all_pixels = all_pixels.reshape(height, width, 2)
    p1 = p1.reshape(height, width, 2)
    
    #img = cv2.add(frame,mask)

    # Flow vector calculated by subtracting new pixels by old pixels
    flow = p1 - all_pixels
    step=8
    
    h, w = frameGray.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(frameGray, lines, 0, (0, 255, 0))
    for (x1, y1), (_, _) in lines:
        frame = cv2.circle(frameGray, (x1, y1), 1, (0, 255, 0), -1)
        
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    cv2.waitKey()
    
    gray = frameGray
    mask = np.zeros_like(gray)

#cap.release()
cv2.destroyAllWindows()
