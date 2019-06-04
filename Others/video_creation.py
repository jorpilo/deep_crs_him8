"""
    video_creation.py

    Creates a video of a layer of a dataset
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video creating tool')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output file of the dataset', type=str)
    parser.add_argument('band',
                        help='band or layer to output', type=str)

    args = parser.parse_args()
    fig = plt.figure()
    ds = xr.open_dataset(args.input)
    print(ds)
    print(ds.time)
    ims = []
    time = ds[args.band].shape[0]
    for i in range(time):
        im = plt.imshow(ds[args.band][i,:,:], animated=True, vmin=200, vmax=400)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    ani.save(args.output)
