import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import cv2


'''
fig = plt.figure()
ds = xr.open_dataset("../../dataset/sat_pre_video2/sortH8_2017.nc")


for i in range(100):
    im = plt.imshow(ds["B11"][i,:,:], animated=True)
    #ims.append([im])
    print(i)
print(len(ims))
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)

ani.save('cloudsB7.mp4')
'''

fig = plt.figure()
#ofv = []


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

# The video feed is read in as a VideoCapture object
def main():
    cap = xr.open_dataset("../dataset/sat_pre_video1/sortH8_Flow.nc")

    a = np.asarray(cap['B11'])
    print("read")
    lkParams=dict( winSize =(15,15),
                      maxLevel=2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    dataset = np.round(255*(a - np.min(a))/np.ptp(a).astype(int)).astype(np.uint8)

    gray = base = dataset[0,:,:]
    height, width = gray.shape

    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    #cv2.imshow('frame',gray)
    #cv2.waitKey()

    hsv = np.zeros_like(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
    hsv[..., 1] = 255
    video=cv2.VideoWriter('test.mp4',-1,10,(width,height))

    for i in range(1, len(dataset)):
        frameGray = dataset[i,:,:]

        """
        all_pixels = np.nonzero(gray)[::-1]
        all_pixels = tuple(zip(*all_pixels))
        all_pixels = np.vstack(all_pixels).reshape(-1, 1, 2).astype("float32")
        """

        #p1, st, err = cv2.calcOpticalFlowPyrLK(gray, frameGray, None, None, **lkParams)
        #all_pixels = all_pixels.reshape(height, width, 2)
        #p1 = p1.reshape(height, width, 2)

        #img = cv2.add(frame,mask)

        # Flow vector calculated by subtracting new pixels by old pixels
        #flow = p1 - all_pixels

        flow = cv2.calcOpticalFlowFarneback(gray, frameGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow *= -2
        img = draw_flow(cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR), flow)
        
        '''
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        numpy_horizontal = np.hstack((cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR),img, rgb))
        '''
        #im = plt.imshow( img, animated=True)
        
        
        #ofv.append(img)

        #cv2.imshow('frame',numpy_horizontal)
        #cv2.waitKey()
        video.write(img)

        gray = frameGray
        

    #cap.release()
   # cv2.destroyAllWindows()
    #ani = animation.ArtistAnimation(fig, ofv, interval=50, blit=True, repeat_delay=1000)
    #ani.save('ofvB11mm.mp4')
    


if __name__ == "__main__":
    main()
