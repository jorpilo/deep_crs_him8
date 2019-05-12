import numpy as np
import cv2
import xarray as xr

def main():
    ds = xr.open_dataset("../../sat_precip/H8_Flow.nc")
    # Set the Corner detection parameters
    cornerParams= dict(maxCorners=100,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7 )


    # Set the Lucas Kanade Optical Flow Parameters
    lkParams=dict( winSize =(15,15),
                  maxLevel=2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



    # The Video Needs to be captured
    a = np.asarray(ds['B7'])

    dataset = np.round(255*(a - np.min(a))/np.ptp(a).astype(int)).astype(np.uint8)

    gray = dataset[0,:,:]
    #%%
    cv2.imshow('frame',gray)
    cv2.waitKey()
    # First Frame


    p0 = cv2.goodFeaturesToTrack(gray, mask = None, **cornerParams)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(gray)

    # Random Colours
    colour=np.random.randint(0,255,(100,3))

    for i in range(1, len(dataset)):
        frameGray = dataset[i,:,:]

         # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray, frameGray, p0, None, **lkParams)

         # Select good points
        goodPointNew = p1[st==1]
        goodPointOld = p0[st==1]
         # draw the tracks
        for i,(new,old) in enumerate(zip(goodPointNew,goodPointOld)):
             a,b = new.ravel()
             c,d = old.ravel()
             mask = cv2.line(mask, (a,b),(c,d), colour[i].tolist(), 2)
             frame = cv2.circle(gray,(a,b),5,colour[i].tolist(),-1)
        img = cv2.add(gray,mask)

        cv2.imshow('frame',img)
        cv2.waitKey()

     # Now update the previous frame and previous points
        gray = frameGray
        p0 = goodPointNew.reshape(-1,1,2)

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()