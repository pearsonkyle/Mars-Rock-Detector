#run label_maker.py -i training/data/brain_coral/PSP_010366_2155/PSP_010366_2155_RED.jp2 -r 0
from matplotlib.backend_bases import MouseButton
from skimage.transform import downscale_local_mean, resize
from skimage.util.shape import view_as_windows
from sklearn.model_selection import train_test_split
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse
import shutil
import glymur
import pickle
import glob
import os

# row/col index of tile
ri = 0
ci = 0
last_label_pos = (0,0) # position of last label in global view plot

# iterator for view_as_windows
def iterate_index(VI):
    global ci, ri
    ci += 1
    if ci >= VI.shape[1]:
        ci = 0
        ri += 1
    if ri >= VI.shape[0]:
        ri = 0

class LabelMaker():
    # class to keep track of labels + samples
    def __init__(self, classes, outdir):
        self.classes = classes
        self.data = {}
        self.pos = {} # position of data in original image
        self.class_idx = 0
        self.outdir = outdir

        clf_path = os.path.join(self.outdir, "classifier.pkl")
        if os.path.exists(clf_path):
            self.clf = pickle.load(open(clf_path,"rb"))
        else:
            #self.clf = SVC(kernel='rbf', probability=True)
            self.clf = RandomForestClassifier()

        for c in classes:
            self.data[c] = []
            self.pos[c] = []

            if not os.path.exists(os.path.join(outdir, c)):
                os.makedirs(os.path.join(outdir, c))
            else:
                # read pickle file to preserve orignal data type
                if os.path.exists(os.path.join(outdir, c, "data.pkl")):
                    self.data[c] = pickle.load(open(os.path.join(outdir, c, "data.pkl"),"rb"))
                if os.path.exists(os.path.join(outdir, c, "pos.pkl")):
                    self.pos[c] = pickle.load(open(os.path.join(outdir, c, "pos.pkl"),"rb"))
                # read samples in from disk to continue
                #for f in glob.glob(os.path.join(outdir, c, "*.jpg")):
                #    self.data[c].append(plt.imread(f)) # only use red channel
                #    self.pos[c].append(tuple(os.path.basename(f).split('.')[0].split('_')))
                #print(f"Loaded {len(self.data[c])} samples for class {c}")

    def reset_labels(self, idata, size):
        # create new labels that are all the same size
        for c in self.classes:
            # loop through images in each class
            for i, d in enumerate(self.data[c]):
                if (d.shape != (size,size)):
                    # extract patch from original image
                    pos = self.pos[c][i]
                    patch = idata[int(pos[1]-size/2):int(pos[1]+size/2),int(pos[0]-size/2):int(pos[0]+size/2)]
                    self.data[c][i] = patch

            # save pickle file
            pickle.dump(self.data[c], open(os.path.join(self.outdir, c, "data.pkl"),"wb"))

    def check_labels(self, size):
        # for each class check which images are not the same size
        check = False
        for c in self.classes:
            # loop through images in each class
            print(f"Checking class {c}, {len(self.data[c])} samples")
            for i, d in enumerate(self.data[c]):
                if (d.shape != (size,size)):
                    print(f"Class {c} image {i} has shape {d.shape} but should be ({size},{size})")
                    check = True
        return check

    def next_class(self):
        # move to next class
        self.class_idx += 1
        if self.class_idx >= len(self.classes):
            self.class_idx = 0

    def save_patch(self, patch, pos, disk=True):
        self.data[self.class_key].append(patch)
        self.pos[self.class_key].append(pos)
        if disk:
            # save pickle file
            pickle.dump(self.data[self.class_key], open(os.path.join(self.class_dir, "data.pkl"),"wb"))
            pickle.dump(self.pos[self.class_key], open(os.path.join(self.class_dir, "pos.pkl"),"wb"))

            # save image
            fname = os.path.join(self.class_dir, f"{int(pos[0])}_{int(pos[1])}.jpg")
            plt.imsave(fname, patch, cmap='gray', vmin=0, vmax=1024)
            print(f"Saved {fname}")

    def delete_last_patch(self, disk=True):
        self.data[self.class_key].pop()
        pos = self.pos[self.class_key].pop()
        if disk:
            fname = os.path.join(self.class_dir, f"{int(pos[0])}_{int(pos[1])}.jpg")
            if os.path.exists(fname):
                os.remove(fname)
                print(f"Deleted {fname}")

            # update pickle file
            pickle.dump(self.data[self.class_key], open(os.path.join(self.class_dir, "data.pkl"),"wb"))
            pickle.dump(self.pos[self.class_key], open(os.path.join(self.class_dir, "pos.pkl"),"wb"))


    @property
    def class_key(self):
        return self.classes[self.class_idx]

    @property
    def class_dir(self):
        return os.path.join(self.outdir, self.class_key)

    @property
    def class_count(self):
        return len(self.data[self.class_key])

    @property
    def dataset(self):
        X = []; y = []; w = []
        for i,c in enumerate(self.classes):
            X.extend(self.data[c])
            y.extend([i]*len(self.data[c]))
            print(f"{c}: {len(self.data[c])}")

        X = np.array(X)
        y = np.array(y)
        return X,y

    def train(self):
        print("Training classifier...")

        X,y = self.dataset
        Xf = X.reshape(X.shape[0], -1)
        #w = np.array(w)

        # whiten each sample
        #Xfp = (Xf - Xf.mean(axis=1, keepdims=True)) / (1+Xf.std(axis=1, keepdims=True))
        Xfp = Xf/1024.
        X_train, X_test, y_train, y_test = train_test_split(Xfp, y, test_size=0.15, random_state=42)
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        # TODO class weight
        self.clf = RandomForestClassifier(class_weight='balanced')# SVC(kernel='rbf', probability=True)
        self.clf.fit(X_train, y_train)
        print(f"Test Accuracy: {self.clf.score(X_test, y_test)*100:.2f}")
        print(f"Train Accuracy: {self.clf.score(X_train, y_train)*100:.2f}")

        # save classifier
        pickle.dump(self.clf, open(os.path.join(self.outdir, "classifier.pkl"),"wb"))

    def predict(self, image):
        # break image into windows
        step = 1#int(self.data[self.class_key][0].shape[0]/2)
        patches_og = view_as_windows(image, self.data[self.class_key][0].shape, step=step)
        patches = patches_og.reshape(-1, *self.data[self.class_key][0].shape)
        patch_image = np.array([patches_og.shape[0]*step, patches_og.shape[1]*step])
        pad = [1+int((image.shape[0] - patch_image[0])*0.5), 1+int((image.shape[1] - patch_image[1])*0.5)]

        # preprocess each sample, scale between 0 - 1
        Xf = patches.reshape(patches.shape[0], -1)
        #Xfp = (Xf - Xf.mean(axis=1, keepdims=True)) / (1+Xf.std(axis=1, keepdims=True))
        Xfp = Xf/1024.

        # predict
        y_pred = self.clf.predict(Xfp)
        y_prob = self.clf.predict_proba(Xfp)

        # reshape
        y_pred = y_pred.reshape(patches_og.shape[0],patches_og.shape[1], 1)
        y_prob = y_prob.reshape(patches_og.shape[0],patches_og.shape[1], len(self.classes))

        # resize to original image size
        y_pred = resize(y_pred, patch_image, order=0, preserve_range=True, anti_aliasing=False)
        y_prob = resize(y_prob, patch_image, order=0, preserve_range=True, anti_aliasing=False)

        # pad 
        y_pred = np.pad(y_pred, ((pad[0],0),(pad[1],0),(0,0)), 'constant', constant_values=0)
        y_prob = np.pad(y_prob, ((pad[0],0),(pad[1],0),(0,0)), 'constant', constant_values=0)

        return y_pred, y_prob

        

colors = ['limegreen', 'orange', 'cyan', 'purple', 'red', 'black', 'white']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ws", "--window", help="size of local viewing window in px", type=int, default=256)

    parser.add_argument("-os", "--outsize", help="size of training sample output in px", type=int, default=11) # size of training data

    parser.add_argument("-o", "--outdir", type=str, default="training/",
            help="Directory to save training samples to")

    parser.add_argument("-n", "--name", type=str, default="rock_{}.jpg",
            help="Name of output images")

    parser.add_argument("-i", "--image", type=str, default="images/ESP_016287_2205_RED_A_01_ORTHO.JP2",
            help="Choose a JPEG2000 image to decode")

    parser.add_argument("-r", "--res", type=int, default=0,
            help="Resolution level to decode (0 is highest resolution)")

    parser.add_argument("-t", "--threads", help="number of threads for background class", default=8, type=int)

    parser.add_argument("-re", "--reset", help="delete existing labels", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    glymur.set_option('lib.num_threads', args.threads)

    # set up output directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.reset:
        # recursive delete of directories
        for f in os.listdir(outdir):
            # delete every file in each class directory
            shutil.rmtree(os.path.join(outdir, f),ignore_errors=True)

    # set up utility for labelling
    LM = LabelMaker(["rock", "background"], outdir)

    # read in image
    print(f"Reading {args.image}...")
    idata = glymur.Jp2k(args.image).read(rlevel=args.res).astype(np.float32)

    # check if labels are the same as window size
    if LM.check_labels(args.outsize):
        # if not, remake labels
        print("Remaking labels...")
        LM.reset_labels(idata, args.outsize)

    # resize image for visualizer
    rdata = downscale_local_mean(idata, (5,5))

    # break into windows
    BI = view_as_windows(idata, args.window, step=int(args.window))
    BIR = BI.reshape(-1, args.window, args.window)

    # scroll to nonblack image
    iterate_index(BI)
    #while (np.median(BI[ri,ci]) < 1024):
    #    iterate_index(BI)

    # set up plot
    fig = plt.figure(figsize=(16,10))
    ax_local = plt.subplot2grid((4,6), (0,0), colspan=4, rowspan=4) # local view
    ax_global = plt.subplot2grid((4,6), (0,4), colspan=2, rowspan=2) # global view
    ax_sample = plt.subplot2grid((4,6), (2,4), colspan=2, rowspan=2) # rock view
    ax_global.set_title(os.path.basename(outdir))
    ax_global.imshow(rdata, cmap="gray",vmin=0,vmax=1024)
    ax_sample.set_title("Rock")

    im = ax_local.imshow(BI[ri,ci], cmap='gray',vmin=0,vmax=1024)
    plt.colorbar(im, ax=ax_local)
    ax_local.set_title(f"Local View ({ri},{ci})")

    # plot patch on global view
    patch = plt.Rectangle((ci*args.window/5, ri*args.window/5), args.window/5, args.window/5, linewidth=1,edgecolor='r',facecolor='none')
    ax_global.add_patch(patch)
    ax_global.axis('off')

    # for each position in label maker plot point on global view
    for c in LM.classes:
        for p in LM.pos[c]:
            ax_global.plot(float(p[0])/5, float(p[1])/5,'.', c=colors[LM.classes.index(c)],ms=1,alpha=0.75)


    def on_move(event):
        # get the x and y pixel coords
        x, y = event.x, event.y
        if event.inaxes == ax_local:

            # plot square around click
            x, y = event.xdata, event.ydata

            # extract patch in original image and show in ax_sample
            ax_sample.cla()
            xi = ci*args.window + x
            yi = ri*args.window + y
            patch = idata[int(yi-args.outsize/2):int(yi+args.outsize/2),int(xi-args.outsize/2):int(xi+args.outsize/2)]

            # preprocess patch - looks wierd
            #patch -= np.mean(patch)
            #patch /= (1+np.std(patch))

            ax_sample.imshow(patch, cmap='gray')#,vmin=-1,vmax=1)
            ax_sample.set_title(f"{LM.class_key} ( ) @ [{int(xi)}, {int(yi)}]" )

            fig.canvas.draw_idle()
            #ax = event.inaxes  # the axes instance


    def on_key(event):
        global ri, ci, LM

        print('you pressed', event.key, event.xdata, event.ydata)
        if event.key == 'n': # to go next tile

            iterate_index(BI)
            while (np.var(BI[ri,ci]) < 100):
                iterate_index(BI)

            ax_local.clear()
            ax_local.imshow(BI[ri,ci], cmap='gray')

            for p in ax_global.patches:
                p.remove()

            ax_global.add_patch(plt.Rectangle((ci*args.window/5, ri*args.window/5), 
                                args.window/5, args.window/5, fill=False, 
                                edgecolor='r',alpha=0.75, lw=1))

            # clear all patches in local view
            for p in ax_local.patches:
                p.remove()

        elif event.key == 'd':
            ax_local.patches.pop()
            ax_sample.clear()
            ax_sample.set_title(LM.class_key)
            # TODO draw x over last patch in global view

            # remove last patch from data
            LM.delete_last_patch()
        elif event.key == 'c':
            # change class
            LM.next_class()
            ax_sample.clear()
            ax_sample.set_title(LM.class_key)
        elif event.key == 't':
            LM.train()
        elif event.key == 'p':
            pred,prob = LM.predict(BI[ri,ci])
            ax_local.imshow(BI[ri,ci], cmap='gray')
            #ax_local.imshow(prob[:,:,0], cmap='jet',vmin=0, vmax=1, alpha=0.15)

            labels, ngroups = label(prob[:,:,0]>0.5)
            # for each group draw a rectangle
            for g in range(1,ngroups+1):
                # skip if smaller than a pixel
                if np.sum(labels==g) < 2:
                    continue
                # get bounding box
                y,x = np.where(labels==g)
                rect = plt.Rectangle((x.min(),y.min()),x.max()-x.min()+1,y.max()-y.min()+1,linewidth=1,edgecolor='r',facecolor='none',alpha=0.75)
                ax_local.add_patch(rect)
        elif event.key == '+':
            # add predictions to training data
            pred,prob = LM.predict(BI[ri,ci])
            # TODO
        elif event.key == 'i':
            ax_local.cla()
            ax_local.imshow(BI[ri,ci], cmap='gray')

        fig.canvas.draw_idle()

    def on_click(event):
        global ri, ci, LM

        # check if xdata and ydata are not None
        if event.xdata is not None and event.ydata is not None:

            if event.button is MouseButton.LEFT:
                #print('data coords %f %f' % (event.xdata, event.ydata))
                #plt.disconnect(binding_id)

                # check which axis was clicked
                if event.inaxes == ax_local:

                    # plot square around click
                    x, y = event.xdata, event.ydata
                    rect = plt.Rectangle((x-args.outsize/2,y-args.outsize/2),args.outsize,args.outsize,fill=False,color=colors[LM.class_idx],label=LM.class_key)
                    ax_local.add_patch(rect)

                    # extract patch in original image and show in ax_sample
                    ax_sample.cla()
                    xi = ci*args.window + x
                    yi = ri*args.window + y
                    patch = idata[int(yi-args.outsize/2):int(yi+args.outsize/2),int(xi-args.outsize/2):int(xi+args.outsize/2)]
                    # save patch to diskq
                    LM.save_patch(patch, (xi, yi))

                    # mark global view + show patch
                    ax_global.plot(xi/5, yi/5, '.', color=colors[LM.class_idx],ms=1,alpha=0.75)
                    ax_sample.imshow(patch, cmap='gray')
                    ax_sample.set_title(f"{LM.class_key} ({LM.class_count}) @ [{int(xi)}, {int(yi)}]" )

                elif event.inaxes == ax_global:
                    # convert x,y to row,col of BI
                    ri = int(event.ydata*5/args.window)
                    ci = int(event.xdata*5/args.window)

                    #import pdb; pdb.set_trace()
                    ax_local.clear()
                    ax_local.imshow(BI[ri,ci], cmap='gray',vmin=0,vmax=1024)

                    for p in ax_global.patches:
                        p.remove()

                    ax_global.add_patch(plt.Rectangle((ci*args.window/5, ri*args.window/5), 
                                        args.window/5, args.window/5, fill=False, 
                                        edgecolor='r',alpha=0.75, lw=1))

                    # clear all patches in local view
                    for p in ax_local.patches:
                        p.remove()
                fig.canvas.draw_idle()

    # print commands
    print("Commands:")
    print("  n: next tile")
    print("  d: delete last patch")
    print("  c: change class")
    print("  t: train model")
    print("  p: predict")
    print("  i: clear patches")
    print("  left click: label patch")

    # interactions
    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_key)
    plt.connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()