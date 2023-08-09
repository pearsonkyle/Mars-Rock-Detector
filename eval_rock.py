import os
import cv2
import glymur
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.util.shape import view_as_windows
from skimage.transform import resize

from trainer import train_ensembler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, default="images/PSP_001410_2210_RED_A_01_ORTHO.JP2", help="Choose a JPEG2000 image to decode")

    parser.add_argument("-r", "--res", type=int, default=0, help="Resolution level to decode (0 is highest resolution)")

    parser.add_argument("-th", "--threads", help="number of threads for background class", default=8, type=int)

    parser.add_argument("-tr", "--train", help="training data for rocks", default="training/training_data_121.csv", type=str)

    parser.add_argument("-te", "--test", help="testing data for rocks", default="training/testing_data_121.csv", type=str)

    parser.add_argument("-o", "--outdir", type=str, default="output/", help="Directory to save output images")

    parser.add_argument("-p", "--plot", help="plot results", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":

    # read args from cmd line
    args = parse_args()

    glymur.set_option('lib.num_threads', args.threads)

    print("Training ensemble...")
    ensemble = train_ensembler(args.train, args.test)
    print(ensemble.acc)

    print('Reading image...')
    image = glymur.Jp2k(args.image).read(rlevel=0).astype(np.float32)

    # set up output
    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    # make dirs
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # 11x11 window size
    ws = int(ensemble.xdim**0.5)

    # create boolean mask same size as image
    mask = np.zeros(image.shape, dtype=bool)

    # break into 512x512 windows for evaluation
    #BI = view_as_windows(image, (512, 512), 512)
    #BIR = BI.reshape(-1, 512, 512) # reshape to 1D array
    #for i in tqdm(range(BIR.shape[0])):
    eval_size = 512

    # create double for loop for slicing image into 512x512 windows
    for i in tqdm(range(0, image.shape[0], eval_size)):
        for j in range(0, image.shape[1], eval_size):

            # slice image into 512x512 window
            window = image[i:i+eval_size, j:j+eval_size]

            # skip if window is all zeros
            if window.max() == 0:
                continue

            # break into 11x11 windows
            BI_sub = view_as_windows(window, (ws, ws), 1)
            BIR_sub = BI_sub.reshape(-1, ws**2)

            # compute some size stuff for reconstruction
            patch_image = np.array([BI_sub.shape[0], BI_sub.shape[1]])
            pad = [int((window.shape[0] - patch_image[0])*0.5), int((window.shape[1] - patch_image[1])*0.5)]

            # preprocess each sample, scale between 0 - 1
            patches_flat = BIR_sub.reshape(BIR_sub.shape[0], -1)
            patches_pre = (patches_flat - patches_flat.mean(axis=1, keepdims=True)) / (1+patches_flat.std(axis=1, keepdims=True))

            # apply scaler as generic filter to image
            center_pixel = int((ws**2)/2)
            image_pre = patches_pre[:,center_pixel].reshape(BI_sub.shape[0], BI_sub.shape[1])

            # get predictions
            y_prob, y_std = ensemble.predict(BIR_sub, prob=True)
            #y_prob = ensemble.predict_best(BIR_sub, prob=True)

            # reshape to 512x512ish
            y_prob = y_prob[:,1].reshape(BI_sub.shape[0], BI_sub.shape[1])
            y_prob = resize(y_prob, patch_image, order=0, preserve_range=True, anti_aliasing=False)

            # pad to get to 512x512
            y_prob = np.pad(y_prob, ((pad[0],pad[0]), (pad[1],pad[1])), 'constant', constant_values=0)

            # save y_prob back to mask
            mask[i:i+eval_size, j:j+eval_size] = y_prob > 0.6 # TODO parameterize threshold?

            if args.plot:
                # create plot
                fig, ax_local = plt.subplots(3,2, figsize=(12,12))
                ax_local[0,0].imshow(window, cmap='gray')
                ax_local[0,1].imshow(image_pre, cmap='gray')
                ax_local[1,1].imshow(window, cmap='gray') # rock labels
                ax_local[1,0].imshow(window, cmap='gray') # probability map

                im = ax_local[1,0].imshow(y_prob, cmap='jet', alpha=0.5)
                cbar = fig.colorbar(im, ax=ax_local[1,0], fraction=0.046, pad=0.04)
                cbar.set_label('Probability of Rock', rotation=270, labelpad=20)
                # scale cbar down
                ax_local[0,0].set_title(f"{os.path.basename(args.image).split('.')[0]}", fontsize=16)
                ax_local[1,0].set_title(f"Probability of Rock using RF", fontsize=16)
                ax_local[1,1].set_title(f"Predicted Rock", fontsize=16)
                ax_local[0,1].set_title(f"Preprocessed Image", fontsize=16)

                # draw circles on rock labels
                labels, ngroups = label(y_prob>0.6)
                sizes = []
                probs = []
                # for each group draw a rectangle
                for g in range(1,ngroups+1):
                    rock_mask = labels==g
                    # skip if smaller than a pixel
                    if np.sum(rock_mask) < 2:
                        continue

                    # get bounding box
                    y,x = np.where(rock_mask)

                    radius = np.sqrt((x.max()-x.min()+1)*(y.max()-y.min()+1))

                    # estimate size + probability
                    size = np.sum(rock_mask)
                    sizes.append(size)
                    probs.append(np.max(y_prob[y,x]))

                    # estimate color based on probability
                    color_scale = np.max(y_prob[y,x])
                    color = plt.cm.jet(color_scale)

                    # plot circle based on size of bounding box
                    ax_local[1,1].plot((x.min()+x.max())*0.5, (y.min()+y.max())*0.5, 'o', color=color, markersize=radius, alpha=0.5, fillstyle='none')

                # collect data into bins of probability
                pbins = np.linspace(0.5,1,6) # bins of probability
                size_bins = [[] for ii in range(len(pbins)-1)]
                # loop over sizes and put into probability bins
                for ii in range(len(sizes)):
                    for jj in range(len(pbins)-1):
                        if probs[ii] >= pbins[jj] and probs[ii] < pbins[jj+1]:
                            size_bins[jj].append(sizes[ii])
                            break

                # create stacked histogram with colors for intervals of probability
                ax_local[2,0].hist(size_bins, bins=np.arange(20), stacked=True, 
                                    color=plt.cm.jet(np.linspace(0.5,1,len(pbins)-1)), 
                                    label=[f"{pbins[ii]:.2f} - {pbins[ii+1]:.2f}" for ii in range(len(pbins)-1)])

                ax_local[2,0].legend()
                ax_local[2,0].set_xlabel('Size of Rock Shadow [px]', fontsize=14)
                ax_local[2,0].set_ylabel('Count', fontsize=14)
                ax_local[2,0].set_title(f"Histogram of Rock Sizes", fontsize=16)

                # remove last axis
                ax_local[2,1].axis('off')
                plt.tight_layout()
                fname = os.path.join(outdir, f"{i}_{j}_prediction.png")
                plt.savefig(fname, dpi=300)
                plt.close()
                print(f"Saved {fname}")

    # save final rock mask
    fname = os.path.join(outdir, f"rock_mask_{args.res}.png")
    cv2.imwrite(fname, mask.astype(np.uint8)*255)
    print(f"Saved {fname}")