"""
    image_creation.py

    Save the images of a layer of a dataset into a foder
"""

import matplotlib.pyplot as plt
import xarray as xr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video creating tool')
    parser.add_argument('input',
                        help='Input file of the dataset', type=str)
    parser.add_argument('output',
                        help='Output folder of the dataset', type=str)
    parser.add_argument('band',
                        help='band or layer to output', type=str)

    args = parser.parse_args()
    fig = plt.figure()
    ds = xr.open_dataset(args.input)
    print(ds)
    print(ds.time)
    data = ds[args.band].values
    time = data.shape[0]
    for i in range(time):
        image = data[i]
        plt.imshow(image)
        plt.savefig(args.output+"/"+args.band+"_"+str(i) + ".png")
