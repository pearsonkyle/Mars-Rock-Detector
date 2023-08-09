import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--indir", type=str, default="training/",
        help="Directory where individual training data is saved")

    parser.add_argument("-o", "--outdir", type=str, default="training/",
            help="Directory where combined training data is saved")

    parser.add_argument("-ws", "--windowsize", type=int, default=11,
            help="size of training sample output in px") # size of training data
    
    parser.add_argument("-f", "--fraction", type=float, default=0.1,
            help="fraction of data to use for testing")

    return parser.parse_args()

if __name__ == "__main__":

    # read args from cmd line
    args = parse_args()

    # find all csv files in indir
    csv = glob.glob(os.path.join(args.indir, f"*/*_{args.windowsize**2}.csv"))

    # combine all csv files
    df = pd.concat([pd.read_csv(f) for f in csv], ignore_index=True)

    # split into train and test
    train, test = train_test_split(df, test_size=args.fraction)

    # save to csv
    fname = os.path.join(args.outdir,f'training_data_{args.windowsize**2}.csv')
    train.to_csv(fname, index=False)
    print(f"Train Data {train.shape} saved to {fname}")

    fname = os.path.join(args.outdir,f'testing_data_{args.windowsize**2}.csv')
    test.to_csv(fname, index=False)
    print(f"Test Data {test.shape} saved to {fname}")