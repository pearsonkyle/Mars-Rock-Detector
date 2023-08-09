# run this script after running label_maker.py on a set of images
# this script will export the training samples into a csv file
import os
import argparse
import pandas as pd
import numpy as np
import glymur

from label_maker import LabelMaker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ws", "--windowsize", type=int, default=11,
            help="size of training sample output in px") # size of training data

    parser.add_argument("-o", "--outdir", type=str, default="training/",
            help="Directory where training csv is saved")

    parser.add_argument("-i", "--image", type=str, default="images/PSP_001410_2210_RED_A_01_ORTHO.JP2",
            help="Choose a JPEG2000 image to decode")

    parser.add_argument("-r", "--res", type=int, default=0,
            help="Resolution level to decode (0 is highest resolution)")

    parser.add_argument("-t", "--threads", default=4, type=int,
            help="number of threads for background class")

    return parser.parse_args()

if __name__ == "__main__":

    # read args from cmd line
    args = parse_args()

    # directory of data
    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    # Load data from label maker
    LM = LabelMaker(["rock", "background"], outdir)

    # check if labels are the same as window size
    if LM.check_labels(args.windowsize):
        # if not, remake labels
        print("Remaking labels...")
        idata = glymur.Jp2k(args.image).read(rlevel=args.res).astype(np.float32)
        LM.reset_labels(idata, args.windowsize)

    rocks = np.array(LM.data['rock']).reshape(len(LM.data['rock']), -1)
    background = np.array(LM.data['background']).reshape(len(LM.data['background']), -1)

    # create csv file
    df = pd.DataFrame(np.concatenate((rocks, background), axis=0))
    # create label column
    df['label'] = np.concatenate((np.ones(len(rocks)), np.zeros(len(background))), axis=0)
    # save to csv
    fname = os.path.join(outdir,f'training_data_{args.windowsize**2}.csv')
    df.to_csv(fname, index=False)
    print(f"Data saved to {fname}")
