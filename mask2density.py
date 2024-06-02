import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import pickle
import glymur
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation, binary_closing, binary_erosion, label

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, default="images/ESP_036762_1845_RED_A_01_ORTHO.JP2",
            help="Choose a JPEG2000 image to decode")

    parser.add_argument("-o", "--outdir", type=str, default="output/", help="Directory to save outputs")

    parser.add_argument("-th", "--threads", default=4, type=int,
            help="number of threads for reading in JP2000 image")

    return parser.parse_args()


if __name__ == "__main__":
    # open image
    args = parse_args()

    # image loading options
    glymur.set_option('lib.num_threads', args.threads)

    # set up output
    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    # read in boolean mask for rocks
    mask = cv2.imread(os.path.join(outdir,"rock_mask_0.png"), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(bool)

    # open jp2k image
    image = glymur.Jp2k(args.image).read(rlevel=0).astype(np.float32)
    non_zero_count = np.count_nonzero(image)

    # clean mask with binary operations
    mask = binary_closing(mask)
    mask = mask & binary_dilation(binary_erosion(mask),iterations=3)
    # Even just a single binary erosion operation eliminates any cluster 
    # of points smaller than 2x3 and hence why we see the density drop off 
    # so drastically. Binary dilation in combination with a logical and on 
    # the original mask preserves the boundary shape of smaller segments. 
    # This is converse to just doing an erosion followed by a dilation, which 
    # tends to inflate the boundary uniformly in all directions and tends to 
    # make the rock shapes more homogeneous at smaller sizes since you lose 
    # information about the edges and boundary. Dilating the mask too much 
    # will start to produce diamond shape masks as well. 

    # save cleaned mask to output directory
    cv2.imwrite(os.path.join(outdir,"rock_mask_cleaned.png"), mask.astype(np.uint8)*255)

    # start counting rocks
    label_image, ngroups = label(mask)
    regions = regionprops(label_image)

    # lists to store size of each rock
    rock_data = {
        "pixel_sizes": [],    # number of pixels in rock
        "ellipse_area": [],   # area of ellipse that fits rock
        "rock_locations": [], # location of rock in pixels
    }

    # compute size of each rock
    for region in tqdm(regions):

        # filter out small rocks
        if region.area <= 2:
            continue

        # compute area of ellipse
        area = region.axis_major_length * region.axis_minor_length # * np.pi
        # The addition of pi is too much, the sizes don't make sense...

        # save data to dict
        rock_data["pixel_sizes"].append(region.area)
        rock_data["ellipse_area"].append(area)
        rock_data["rock_locations"].append(region.centroid)

    # cast as numpy arrays
    pixel_sizes = np.array(rock_data["pixel_sizes"])
    ellipse_area = np.array(rock_data["ellipse_area"])
    rock_locations = np.array(rock_data["rock_locations"])

    # compute some metrics
    area_per_pixel = 0.3**2 # m^2
    total_area = non_zero_count * area_per_pixel
    total_rock_area = np.sum(pixel_sizes) * area_per_pixel
    total_ellipse_area = np.sum(ellipse_area) * area_per_pixel

    print(f"Mean size of rock shadow: {np.mean(ellipse_area):.2f} px")
    print(f"Median size of rock shadow: {np.median(ellipse_area):.2f} px")
    print(f"Standard deviation of rock shadow size: {np.std(ellipse_area):.2f} px")
    print(f"Max size of rock shadow: {np.max(ellipse_area):.2f} px")
    print(f"Min size of rock shadow: {np.min(ellipse_area):.2f} px")
    print(f"Number of rock shadows: {len(ellipse_area)}")
    print(f"Total area of rock shadows: {total_rock_area:.2f} m^2")
    print(f"Total area of rock shadows (ellipse): {total_ellipse_area:.2f} m^2")
    print(f"Total area of image: {total_area:2f} m^2")
    print(f"Percent of image covered by rock (pixel): {total_rock_area/total_area*100:.2f}%")
    print(f"Percent of image covered by rock (ellipse): {total_ellipse_area/total_area*100:.2f}%")
    # compute rock density per square km
    print(f"Rock density (pixel): {len(rock_data['pixel_sizes'])/total_area*1e6:.0f} rocks/km^2")
    print(f"Rock density (pixel): {len(rock_data['pixel_sizes'])/total_area*1e4:.0f} rocks/100m^2")
    # compute average distance between rocks
    print(f"Average distance between rocks: {np.sqrt(total_area/len(rock_data['pixel_sizes'])):.2f} m")
    print(f"Average distance between rocks: {np.sqrt(total_area/len(rock_data['pixel_sizes'])/0.3):.2f} px")

    # save rock data
    rock_data['area_per_pixel'] = area_per_pixel
    rock_data['total_image_area'] = total_area
    rock_data['total_rock_area'] = total_rock_area
    rock_data['total_ellipse_area'] = total_ellipse_area
    rock_data['rock_density_km2'] = len(rock_data['pixel_sizes'])/total_area*1e6
    rock_data['rock_density_100m2'] = len(rock_data['pixel_sizes'])/total_area*1e4
    rock_data['average_rock_distance_m'] = np.sqrt(total_area/len(rock_data['pixel_sizes']))

    # plot histogram of sizes
    fig,ax = plt.subplots(1, 2, figsize=(8, 5))
    total_count = len(ellipse_area)
    counts, bins, _ = ax[0].hist(ellipse_area*area_per_pixel, bins=np.linspace(0.5,2.5,9), alpha=0.5, label=f'N = {total_count}')
    ax[0].set_xlabel(r'Area of Rock (m$^2$)')
    ax[0].set_title('Rock Size Distribution')
    ax[0].grid(True, ls='--')
    ax[0].legend(loc='best')
    dbin = np.diff(bins).mean()
    cfa = counts*(bins[:-1]+dbin)*area_per_pixel/total_area*100

    # fit quadratic to density data and extrapolate to smaller sizes
    x = bins[1:-1]
    y = counts[1:]*(bins[1:-1]+dbin)/total_area*1e4
    p = np.polyfit(x, y, 1)
    xfit = np.linspace(0.25, 2.5, 100)
    yfit = np.polyval(p, xfit)

    # plot density per square 100m
    ax[1].plot(bins[:-1], counts*(bins[:-1]+dbin)/total_area*1e4,'ko')
    ax[1].plot(xfit, yfit, 'r--', label=f'Linear Fit (y = {p[0]:.2f}x + {p[1]:.2f})')
    ax[1].legend(loc='best')
    ax[1].set_xlabel(r'Area of Rock (m$^2$)')
    ax[1].grid(True, ls='--')
    ax[1].set_ylabel(r'Rock Density (# per 100x100 m$^2$)')
    ax[1].set_xlim([0.25,2.5])
    ax[1].set_ylim([0, 90])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'rock_density.png'))
    plt.close()
    print(f"Saved {os.path.join(outdir,'rock_density.png')}")
