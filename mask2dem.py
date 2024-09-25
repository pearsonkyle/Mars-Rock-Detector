import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import glymur
import argparse
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import label, gaussian_filter
from skimage.transform import resize
from scipy import stats
import pandas as pd
import xarray as xr
import joblib

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image", type=str, default="images/PSP_001410_2210_RED_A_01_ORTHO.JP2",
            help="Choose a JPEG2000 image to decode")

    parser.add_argument("-d", "--dem", type=str, default="images/DTEEC_002175_2210_001410_2210_A01.IMG",
            help="Choose a DEM image to open")
    
    parser.add_argument("-b", "--brain_coral", type=str, default="images/PSP_001410_2210_RED_A_01_ORTHO_classifier_mask.png",
            help="Choose a brain coral mask for the image")

    parser.add_argument("-o", "--outdir", type=str, default="output/", help="Directory to save outputs")

    parser.add_argument("-th", "--threads", default=4, type=int,
            help="number of threads for reading in JP2000 image")

    # run mask2dem.py --image images/ESP_016287_2205_RED_A_01_ORTHO.JP2 --dem images/DTEED_077488_2205_016287_2205_A01.IMG -b images/ESP_016287_2205_RED_classifier_mask.png

    return parser.parse_args()

def dem_to_array(dem_file):
    with rasterio.open(dem_file) as src:
        dem = src.read(1)
        dem = np.array(dem)
    return dem

if __name__ == "__main__":
    # open image
    args = parse_args()

    # image loading options
    glymur.set_option('lib.num_threads', args.threads)

    # open jp2k image
    res = 0
    image = glymur.Jp2k(args.image).read(rlevel=res).astype(np.float32)
    area_per_pixel = 0.3**2 # m^2

    # image metrics
    non_zero_count = np.count_nonzero(image)*(2**res)**2
    total_area = non_zero_count * area_per_pixel # in m^2
    total_area_background = total_area*1.0

    # set up output
    image_name = os.path.basename(args.image)
    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.image))[0])

    # read in boolean mask for rocks
    rock_mask = cv2.imread(os.path.join(outdir,"rock_mask_0.png"), cv2.IMREAD_GRAYSCALE)
    rock_mask = rock_mask.astype(bool)

    # clean mask with binary operations
    #rock_mask = binary_closing(rock_mask)
    #rock_mask = mask & binary_dilation(binary_erosion(rock_mask),iterations=3)

    # load brain coral mask if provided
    if args.brain_coral is not None:
        brain_mask = cv2.imread(args.brain_coral, cv2.IMREAD_GRAYSCALE)
        brain_mask = resize(brain_mask, rock_mask.shape, order=0, anti_aliasing=False, preserve_range=True)
        brain_mask = brain_mask > 0
        # resize brain mask to match rock mask
        total_area_brain = np.count_nonzero(brain_mask)*area_per_pixel
        total_area_background = total_area - total_area_brain

    # open dem image
    dem = dem_to_array(args.dem)

    # resize dem to match mask
    dem = resize(dem, rock_mask.shape, order=1, anti_aliasing=False, preserve_range=True)

    # apply gaussian filter to dem
    dem = gaussian_filter(dem, sigma=2)

    # mask bad values
    dem[dem < -1e8] = np.nan

    # compute gradient of dem
    #ddy, ddx = np.gradient(dem)

    # compute gradient magnitude
    #gradient = np.sqrt(ddx**2 + ddy**2) # in meters
    len_per_pixel = 0.3 # meters
    #slope = np.arctan(gradient/len_per_pixel)*180/np.pi # in degrees

    # start counting rocks
    label_image, ngroups = label(rock_mask)
    regions = regionprops(label_image)

    # lists to store size of each rock
    rock_data = {
        "diameter": [],         # diameter of rock
        "pixel_sizes": [],      # number of pixels in rock
        "ellipse_area": [],     # area of ellipse that fits rock
        "rock_locations": [],   # location of rock in pixels
        "brain_coral": [],      # whether rock is in brain coral
        "elevation": [],        # elevation of rock in meters
        "rel_elevation": [],    # relative elevation of rock in meters  after plane fit
        "median_elevation": [],   # average elevation of plane
        "rplane_slope": [],     # slope of rock after plane fit in m/px
        "rplane_slope_deg": [], # slope of rock after plane fit in deg
        "plane_slope": [],      # linear gradient of terrain in meters 
        "plane_slope_deg": [],  # linear slope of terrain in degrees
        #"total_area_brain": total_area_brain, # will be added later
        #"total_area_background": total_area_background 
    }

    bbox = 150 # ~100-150 meters for 0.3-0.5 m/pixel resolution

    # compute size of each rock
    for region in tqdm(regions):

        # if not np.any(brain_mask[region.coords[:,0], region.coords[:,1]]):
        #     continue

        # filter out small rocks
        if region.area <= 2:
            continue

        # compute area of ellipse in pixels
        area = region.axis_major_length * region.axis_minor_length # * np.pi

        # set up bounding box for elevation calculation
        xmin = max(0,region.coords[:,1].min()-bbox)
        xmax = min(region.coords[:,1].max()+bbox, rock_mask.shape[1])
        ymin = max(0,region.coords[:,0].min()-bbox)
        ymax = min(region.coords[:,0].max()+bbox, rock_mask.shape[0])

        # grid for linear regression
        xx = np.arange(xmin, xmax)
        yy = np.arange(ymin, ymax)
        xg, yg = np.meshgrid(xx, yy)

        # compute elevation of rock
        elevation = np.nanmean(dem[region.coords[:,0], region.coords[:,1]])

        # compute relative elevation of rock within bbox
        median_elevation = np.nanmedian(dem[ymin:ymax,xmin:xmax])
        #rel_elevation = elevation - mean_elevation

        # estimate plane, using linalg, elevation = mx + ny + b
        # first filter all nans
        flat_dem = dem[ymin:ymax,xmin:xmax].flatten()
        non_nan_dem = flat_dem[~np.isnan(flat_dem)]
        non_nan_x = xg.flatten()[~np.isnan(flat_dem)]
        non_nan_y = yg.flatten()[~np.isnan(flat_dem)]

        # linear regression
        A = np.vstack([non_nan_x, non_nan_y, np.ones(len(non_nan_x))]).T
        m, n, b = np.linalg.lstsq(A, non_nan_dem, rcond=None)[0]
        plane_elevation = m*xg + n*yg + b
    
        # compute magnitude of gradient
        mag_linear = np.sqrt(m**2 + n**2)

        # compute gradient of surface minus plane
        diff_dem = dem[ymin:ymax,xmin:xmax] - plane_elevation
        ddx, ddy = np.gradient(diff_dem)
        mag = np.sqrt(ddx**2 + ddy**2)

        # compute relative elevation of rock after plane fit
        rel_elevation_plane = np.nanmean(diff_dem[region.coords[:,0]-ymin, region.coords[:,1]-xmin])

        # compute gradient of rock after plane fit
        grad_m = np.nanmean(mag[region.coords[:,0]-ymin, region.coords[:,1]-xmin])
        grad_deg = np.arctan(grad_m/len_per_pixel)*180/np.pi

        # check for nans
        if np.isnan(grad_deg):
            continue
        if np.isnan(elevation):
            continue

        if args.brain_coral is not None:
            # check if coords are in brain coral mask
            if np.all(brain_mask[region.coords[:,0], region.coords[:,1]]):
                rock_data["brain_coral"].append(True)

                # # compare plane to elevation
                # fig, ax = plt.subplots(2, 2, figsize=(9, 8))
                # fig.suptitle(f"{'_'.join(image_name.split('_')[:3])}", fontsize=16)
                # ax[0,0].imshow(image[ymin:ymax,xmin:xmax], origin='lower', cmap='binary_r')
                # ax[0,0].set_title(f"Image", fontsize=16)
                # ax[0,1].imshow(dem[ymin:ymax,xmin:xmax], origin='lower', cmap='jet')
                # #ax[0,1].imshow(diff_dem, origin='lower',vmin=-2,vmax=2, cmap='jet')
                # ax[0,1].set_title(f"Relative Elevation", fontsize=16)
                # ax[1,0].imshow(image[ymin:ymax,xmin:xmax], origin='lower', cmap='binary_r')
                # ax[1,0].imshow(rock_mask[ymin:ymax,xmin:xmax], origin='lower', cmap='Greens', alpha=0.5)
                # ax[1,0].set_title(f"Rock Mask", fontsize=16)
                # sdiff_dem = gaussian_filter(diff_dem, sigma=2)
                # ddx, ddy = np.gradient(sdiff_dem)
                # mag= np.sqrt(ddx**2 + ddy**2)
                # ax[1,1].imshow(image[ymin:ymax,xmin:xmax], origin='lower', cmap='binary_r')
                # ax[1,1].imshow(mag, origin='lower', cmap='jet', alpha=0.25)
                # #ax[1,1].imshow(rock_mask[ymin:ymax,xmin:xmax], origin='lower', cmap='binary_r', alpha=0.5)
                # ax[1,1].set_title('Gradient of Surface', fontsize=16)
                # num_pix = 10/len_per_pixel     
                # # plot 10m white line
                # ax[1,1].plot([5, 5], [5, 5+10/len_per_pixel], 'w-', lw=2)
                # ax[1,1].plot([5, 5+10/len_per_pixel], [5, 5], 'w-', lw=2)
                # ax[1,1].text(5+1, 5+3, '10m', color='w', fontsize=14)

                # ax[0,0].plot([5, 5], [5, 5+10/len_per_pixel], 'w-', lw=2)
                # ax[0,0].plot([5, 5+10/len_per_pixel], [5, 5], 'w-', lw=2)
                # ax[0,0].text(5+1, 5+3, '10m', color='w', fontsize=14)        

                # ax[1,0].plot([5, 5], [5, 5+10/len_per_pixel], 'w-', lw=2)
                # ax[1,0].plot([5, 5+10/len_per_pixel], [5, 5], 'w-', lw=2)
                # ax[1,0].text(5+1, 5+3, '10m', color='w', fontsize=14)

                # ax[0,1].plot([5, 5], [5, 5+10/len_per_pixel], 'w-', lw=2)
                # ax[0,1].plot([5, 5+10/len_per_pixel], [5, 5], 'w-', lw=2)
                # ax[0,1].text(5+1, 5+3, '10m', color='w', fontsize=14)

                # plt.tight_layout()
                # plt.savefig(os.path.join(outdir, f"rock_{np.mean(region.coords[:,0]):.0f}_{np.mean(region.coords[:,1]):.0f}.png"))
                # plt.close()

            else:
                rock_data["brain_coral"].append(False)
        else:
            rock_data["brain_coral"].append(False)

        # save data to dict
        rock_data["pixel_sizes"].append(region.area)
        rock_data["ellipse_area"].append(area)
        rock_data["diameter"].append(region.equivalent_diameter_area) # in pixels
        rock_data["rock_locations"].append(region.centroid)
        rock_data["elevation"].append(elevation)
        rock_data["median_elevation"].append(median_elevation)
        rock_data["rel_elevation"].append(rel_elevation_plane)
        rock_data["rplane_slope"].append(grad_m)
        rock_data["rplane_slope_deg"].append(grad_deg) 
        rock_data["plane_slope"].append(mag_linear)
        rock_data["plane_slope_deg"].append(np.arctan(mag_linear/len_per_pixel)*180/np.pi)

    # save rock data to file - not efficient
    #with open(os.path.join(outdir, "rock_data.pkl"), 'wb') as f:
    #    joblib.dump(rock_data, f)

    # split up rock locations
    centroid_x, centroid_y = zip(*rock_data["rock_locations"])
    rock_data["centroid_x"] = centroid_x
    rock_data["centroid_y"] = centroid_y

    del rock_data["rock_locations"]

    # convert to pandas dataframe
    df = pd.DataFrame(rock_data)

    # convert to netcdf
    ds = xr.Dataset.from_dataframe(df)

    # add total area
    ds.attrs["total_area"] = total_area
    ds.attrs["total_area_background"] = total_area_background

    # save to netcdf
    ds.to_netcdf(os.path.join(outdir, "rock_data.nc"))

    # cast as numpy arrays
    grad = np.array(rock_data["rplane_slope_deg"])
    brain_coral = np.array(rock_data["brain_coral"])
    pixel_sizes = np.array(rock_data["pixel_sizes"])
    ellipse_area = np.array(rock_data["ellipse_area"])
    rel_elevation = np.array(rock_data["rel_elevation"])

    # make some plots
    fig,ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(f"Rock Sizes in {image_name}", fontsize=16)

    # estimate size density of rocks based on meter sized bins
    size_bins = np.linspace(0.5,2.5,19) # in meters

    if np.any(brain_coral):
        # rock density in brain coral
        counts_bc, bins = np.histogram( (ellipse_area*area_per_pixel)[brain_coral], bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2
        density_bc = counts_bc/total_area_brain # N per m^2
        ax.semilogy(bin_center, density_bc, 'ko-', label=f'Brain Coral, N = {len(ellipse_area[brain_coral])}')
        #ax2.semilogy(bin_center, counts_bc, 'ko', label=f'Brain Coral, N = {len(ellipse_area[brain_coral])}')

        # rock density in background
        counts_bg, bins = np.histogram( (ellipse_area*area_per_pixel)[~brain_coral], bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2
        density_bg = counts_bg/total_area_background # N per m^2
        ax.semilogy(bin_center, density_bg, 'ro-', label=f'Background, N = {len(ellipse_area[~brain_coral])}')
        #ax2.semilogy(bin_center, counts_bg, 'ro', label=f'Background, N = {len(ellipse_area[~brain_coral])}')

        #counts_max = max(counts_bc.max(), counts_bg.max())
        density_max = max(density_bc.max(), density_bg.max())

        # set up second axes
        # ax2.set_ylabel('Number of Rocks', fontsize=14)
        # ax2.tick_params(axis='both', which='major', labelsize=14)
        # ax2.set_ylim([0, counts_max*1.1])
        # ax2.set_xlim([0.85,2.5])

        # set up first axes
        ax.set_xlabel(r'Area of Rock (m$^2$)', fontsize=14)
        ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, ls='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_xlim([0.85,2.5])
    else:
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(f"Rock Sizes in {image_name}", fontsize=16)
        ax2 = ax.twinx()

        # estimate size density of rocks based on meter sized bins
        size_bins = np.linspace(0.5,2.5,19) # in meters
        counts, bins = np.histogram(ellipse_area*area_per_pixel, bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2

        # estimate size density of rocks outside brain coral
        density = counts/total_area # per m^2

        # plot density
        ax.semilogy(bin_center, density, 'ko', label=f'N = {len(ellipse_area)}')

        # set up second axes
        ax2 = ax.twinx()
        ax2.semilogy(bin_center, counts, 'ko', label=f'N = {len(ellipse_area)}')
        ax2.set_ylabel('Number of Rocks', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylim([1e-7, max(counts)*1.1])
        ax2.set_xlim([0.85,2.5])

        # set up first axes
        ax.set_xlabel(r'Area of Rock (m$^2$)', fontsize=14)
        ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, ls='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_ylim([1e-7, max(density)*1.1])
        ax.set_xlim([0.85,2.5])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'rock_density.png'))
    plt.close()

    fig,ax = plt.subplots(2, 1, figsize=(7, 9))
    ax[0].set_title(f"Rock Distribution in {image_name.replace('_ORTHO.JP2','')}", fontsize=16)

    # estimate size density of rocks based on meter sized bins
    size_bins = np.linspace(0.5,2.5,19) # in meters

    if np.any(brain_coral):
        
        # rock density in brain coral
        counts_bc, bins = np.histogram( (ellipse_area*area_per_pixel)[brain_coral], bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2

        # cumulative density from smallest to largest
        density_bc = np.cumsum(counts_bc[::-1])[::-1]/total_area_brain # per m^2
        ax[0].plot(bin_center, density_bc, 'ko-', label=f'Brain Coral, N = {len(ellipse_area[brain_coral])}')
        ax[1].semilogy(bin_center, density_bc, 'ko-', label=f'Brain Coral, N = {len(ellipse_area[brain_coral])}')

        # rock density in background
        counts_bg, bins = np.histogram( (ellipse_area*area_per_pixel)[~brain_coral], bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2
        density_bg = counts_bg/total_area_background # N per m^2

        # cumulative density from smallest to largest
        density_bg = np.cumsum(counts_bg[::-1])[::-1]/total_area_background
        ax[0].plot(bin_center, density_bg, 'ro-', label=f'Background, N = {len(ellipse_area[~brain_coral])}')
        ax[1].semilogy(bin_center, density_bg, 'ro-', label=f'Background, N = {len(ellipse_area[~brain_coral])}')

        density_max = max(density_bc.max(), density_bg.max())

        # set up first axes
        ax[0].set_xlabel(r'Area of Rock (m$^2$)', fontsize=14)
        ax[0].set_ylabel(r'Cumulative Density (# per m$^2$)', fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=14)
        ax[0].grid(True, ls='--')
        ax[0].legend(loc='best', fontsize=14)
        ax[0].set_xlim([0.85,2.4])

        ax[1].set_xlabel(r'Area of Rock (m$^2$)', fontsize=14)
        ax[1].set_ylabel(r'Cumulative Density (# per m$^2$)', fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=14)
        ax[1].grid(True, ls='--')
        ax[1].legend(loc='best', fontsize=14)
        ax[1].set_xlim([0.85,2.4])

    else:
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(f"Rock Sizes in {image_name}", fontsize=16)
        ax2 = ax.twinx()

        # estimate size density of rocks based on meter sized bins
        size_bins = np.linspace(0.5,2.5,19) # in meters
        counts, bins = np.histogram(ellipse_area*area_per_pixel, bins=size_bins)
        bin_center = (bins[:-1]+bins[1:])/2

        # cumulative density from smallest to largest
        density = np.cumsum(counts[::-1])[::-1]/total_area # per m^2

        # plot density
        ax.semilogy(bin_center, density, 'ko', label=f'N = {len(ellipse_area)}')

        # set up second axes
        ax2 = ax.twinx()
        ax2.semilogy(bin_center, counts, 'ko', label=f'N = {len(ellipse_area)}')
        ax2.set_ylabel('Number of Rocks', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_ylim([1e-7, max(counts)*1.1])
        ax2.set_xlim([0.85,2.4])

        # set up first axes
        ax.set_xlabel(r'Area of Rock (m$^2$)', fontsize=14)
        ax.set_ylabel(r'Cumulative Rock Density\\ (# per m$^2$)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, ls='--')
        ax.legend(loc='best', fontsize=14)
        ax.set_ylim([1e-7, max(density)*1.1])
        ax.set_xlim([0.85,2.4])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'cdf_rock_density.png'))
    plt.close()


    # relative elevation distribution plot in/out of brain coral
    fig,ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(f"Elevation Distribution in {image_name.replace('_ORTHO.JP2','')}", fontsize=16)
    elevation_bins = np.linspace(-2,2,41) # in meters

    if np.any(brain_coral):
        counts_bc, bins = np.histogram( (rel_elevation)[brain_coral], bins=elevation_bins)
        density_bc = counts_bc/total_area_brain # N per m^2

        counts_bg, bins = np.histogram( (rel_elevation)[~brain_coral], bins=elevation_bins)
        density_bg = counts_bg/total_area_background # N per m^2

        bin_center = (bins[:-1]+bins[1:])/2

        ax.plot(bin_center, density_bc, 'ko-', label=f'Brain Coral, median = {np.median(rel_elevation[brain_coral]):.2f} +- {np.std(rel_elevation[brain_coral]):.2f} m',alpha=1)
        ax.plot(bin_center, density_bg, 'ro-', label=f'Background, median = {np.median(rel_elevation[~brain_coral]):.2f} +- {np.std(rel_elevation[~brain_coral]):.2f} m',alpha=1)
        # make bar chart
        #ax.bar(bin_center, density_bc, width=np.diff(bins), color='k', alpha=0.75, label=f'Brain Coral, mean = {np.mean(rel_elevation[brain_coral]):.2f} +- {np.std(rel_elevation[brain_coral]):.2f}')
        #ax.bar(bin_center, density_bg, width=np.diff(bins), color='r', alpha=0.75, label=f'Background, mean = {np.mean(rel_elevation[~brain_coral]):.2f} +- {np.std(rel_elevation[~brain_coral]):.2f}')
        ax.set_ylim([1e-6, max(density_bg)*1.1])

        # create plot of difference
        #ax[1].plot(bin_center, density_bc-density_bg, 'ko-', label=f'Brain Coral - Background')
        #ax[1].set_ylim([0, max(density_bc-density_bg)*1.1])

    else:
        # histogram of relative elevation
        counts, bins = np.histogram(rel_elevation, bins=elevation_bins)
        density = counts/total_area # per m^2
        bin_center = (bins[:-1]+bins[1:])/2

        # create plot
        ax.semilogy(bin_center, density, 'k-', label=f'mean = {np.mean(rel_elevation):.2f} m, std = {np.std(rel_elevation):.2f} m, skew = {stats.skew(rel_elevation):.2f}')
        ax.set_ylim([1e-6, max(density)*1.1])

    ax.set_xlabel(r'Relative Elevation (m)', fontsize=14)
    ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, ls='--')
    ax.set_xlim([-2,2])
    ax.legend(loc='best', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'elevation_distribution.png'))
    plt.close()


    # create a plot of sizes vs elevation
    # create masks for different size ranges
    tiny_mask = ellipse_area*area_per_pixel < 1
    small_mask = (ellipse_area*area_per_pixel >= 1) & (ellipse_area*area_per_pixel < 1.5)
    medium_mask = (ellipse_area*area_per_pixel >= 1.5) & (ellipse_area*area_per_pixel < 2)
    large_mask = ellipse_area*area_per_pixel >= 2

    if np.any(brain_coral):
        fig,ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{image_name}", fontsize=16)
        ax[0].set_title(f"Brain Coral", fontsize=16)
        ax[1].set_title(f"Background", fontsize=16)

        # y-axis = Density
        # x-axis = Rel. Elevation

        # tiny rocks
        counts_bc, bins = np.histogram( (rel_elevation)[brain_coral & tiny_mask], bins=elevation_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (rel_elevation)[~brain_coral & tiny_mask], bins=elevation_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'k-', label=f'< 1 m   ({np.mean(rel_elevation[brain_coral & tiny_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'k-', label=f'< 1 m   ({np.mean(rel_elevation[~brain_coral & tiny_mask]):.3f} m)')

        # small rocks
        counts_bc, bins = np.histogram( (rel_elevation)[brain_coral & small_mask], bins=elevation_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (rel_elevation)[~brain_coral & small_mask], bins=elevation_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'r-', label=f'1-1.5 m ({np.mean(rel_elevation[brain_coral & small_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'r-', label=f'1-1.5 m ({np.mean(rel_elevation[~brain_coral & small_mask]):.3f} m)')

        # medium rocks
        counts_bc, bins = np.histogram( (rel_elevation)[brain_coral & medium_mask], bins=elevation_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (rel_elevation)[~brain_coral & medium_mask], bins=elevation_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'b-', label=f'1.5-2 m ({np.mean(rel_elevation[brain_coral & medium_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'b-', label=f'1.5-2 m ({np.mean(rel_elevation[~brain_coral & medium_mask]):.3f} m)')

        # large rocks
        counts_bc, bins = np.histogram( (rel_elevation)[brain_coral & large_mask], bins=elevation_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (rel_elevation)[~brain_coral & large_mask], bins=elevation_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'g-', label=f'> 2 m   ({np.mean(rel_elevation[brain_coral & large_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'g-', label=f'> 2 m   ({np.mean(rel_elevation[~brain_coral & large_mask]):.3f} m)')

        ax[0].set_ylim([1e-7,1e-1])
        ax[0].set_xlabel(r'Relative Elevation (m)', fontsize=14)
        ax[0].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=14)
        ax[0].grid(True, ls='--')
        ax[0].legend(loc='best', fontsize=14)

        #ax[1].set_ylim([-7,7])
        ax[1].set_ylim([1e-7,1e-1])
        ax[1].set_xlabel(r'Relative Elevation (m)', fontsize=14)
        ax[1].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=14)
        ax[1].grid(True, ls='--')
        ax[1].legend(loc='best', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir,'multi_size_vs_elevation.png'))
        plt.close()

    else:
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(f"{image_name}", fontsize=16)

        # tiny rocks
        counts, bins = np.histogram( (rel_elevation)[tiny_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'k-', label=f'< 1 m, N = {len(rel_elevation[tiny_mask])}')

        # small rocks
        counts, bins = np.histogram( (rel_elevation)[small_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'r-', label=f'1-1.5 m, N = {len(rel_elevation[small_mask])}')

        # medium rocks
        counts, bins = np.histogram( (rel_elevation)[medium_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'b-', label=f'1.5-2 m, N = {len(rel_elevation[medium_mask])}')

        # large rocks
        counts, bins = np.histogram( (rel_elevation)[large_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'g-', label=f'> 2 m, N = {len(rel_elevation[large_mask])}')

        ax.set_xlim([-4,4])
        ax.set_xlabel(r'Relative Elevation (m)', fontsize=14)
        ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, ls='--')
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,'multi_size_vs_elevation.png'))
        plt.close()


    if np.any(brain_coral):
        fig,ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{image_name}", fontsize=16)
        ax[0].set_title(f"Brain Coral", fontsize=16)
        ax[1].set_title(f"Background", fontsize=16)

        # y-axis = Density
        # x-axis = slopes
        slope_bins = np.linspace(0,31,21) # in meters

        # tiny rocks
        counts_bc, bins = np.histogram( (grad)[brain_coral & tiny_mask], bins=slope_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (grad)[~brain_coral & tiny_mask], bins=slope_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'k-', label=f'< 1 m   ({np.mean(grad[brain_coral & tiny_mask]):.3f} +- {np.std(grad[brain_coral & tiny_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'k-', label=f'< 1 m   ({np.mean(grad[~brain_coral & tiny_mask]):.3f} +- {np.std(grad[~brain_coral & tiny_mask]):.3f} m)')

        # small rocks
        counts_bc, bins = np.histogram( (grad)[brain_coral & small_mask], bins=slope_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (grad)[~brain_coral & small_mask], bins=slope_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'r-', label=f'1-1.5 m ({np.mean(grad[brain_coral & small_mask]):.3f} +- {np.std(grad[brain_coral & small_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'r-', label=f'1-1.5 m ({np.mean(grad[~brain_coral & small_mask]):.3f} +- {np.std(grad[~brain_coral & small_mask]):.3f} m)')

        # medium rocks
        counts_bc, bins = np.histogram( (grad)[brain_coral & medium_mask], bins=slope_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (grad)[~brain_coral & medium_mask], bins=slope_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'b-', label=f'1.5-2 m ({np.mean(grad[brain_coral & medium_mask]):.3f} +- {np.std(grad[brain_coral & medium_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'b-', label=f'1.5-2 m ({np.mean(grad[~brain_coral & medium_mask]):.3f} +- {np.std(grad[~brain_coral & medium_mask]):.3f} m)')

        # large rocks
        counts_bc, bins = np.histogram( (grad)[brain_coral & large_mask], bins=slope_bins)
        density_bc = counts_bc/total_area_brain # N per m^2
        counts_bg, bins = np.histogram( (grad)[~brain_coral & large_mask], bins=slope_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        bin_center = (bins[:-1]+bins[1:])/2
        ax[0].semilogy(bin_center, density_bc, 'g-', label=f'> 2 m   ({np.mean(grad[brain_coral & large_mask]):.3f} +- {np.std(grad[brain_coral & large_mask]):.3f} m)')
        ax[1].semilogy(bin_center, density_bg, 'g-', label=f'> 2 m   ({np.mean(grad[~brain_coral & large_mask]):.3f} +- {np.std(grad[~brain_coral & large_mask]):.3f} m)')

        ax[0].set_ylim([1e-7,1e-1])
        ax[0].set_xlabel(r'Slope (deg)', fontsize=14)
        ax[0].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=14)
        ax[0].grid(True, ls='--')
        ax[0].legend(loc='best', fontsize=14)

        #ax[1].set_ylim([-7,7])
        ax[1].set_ylim([1e-7,1e-1])
        ax[1].set_xlabel(r'Slope (deg)', fontsize=14)
        ax[1].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=14)
        ax[1].grid(True, ls='--')
        ax[1].legend(loc='best', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir,'multi_size_vs_slope.png'))
        plt.close()

    else:
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(f"{image_name}", fontsize=16)

        # tiny rocks
        counts, bins = np.histogram( (grad)[tiny_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'k-', label=f'< 1 m, N = {len(grad[tiny_mask])}')

        # small rocks
        counts, bins = np.histogram( (grad)[small_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'r-', label=f'1-1.5 m, N = {len(grad[small_mask])}')

        # medium rocks
        counts, bins = np.histogram( (grad)[medium_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'b-', label=f'1.5-2 m, N = {len(grad[medium_mask])}')

        # large rocks
        counts, bins = np.histogram( (grad)[large_mask], bins=elevation_bins)
        density = counts/total_area
        bin_center = (bins[:-1]+bins[1:])/2
        ax.semilogy(bin_center, density, 'g-', label=f'> 2 m, N = {len(grad[large_mask])}')

        ax.set_xlim([-4,4])
        ax.set_xlabel(r'Slope (deg)', fontsize=14)
        ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, ls='--')
        ax.legend(loc='best', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,'multi_size_vs_slope.png'))
        plt.close()


    # slope vs size
    fig,ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(f"{image_name}", fontsize=16)
    grad_bins = np.linspace(0,30,151)

    if np.any(brain_coral):
        counts_bc, bins = np.histogram( (grad)[brain_coral], bins=grad_bins)
        bin_center = (bins[:-1]+bins[1:])/2
        
        density_bc = counts_bc/total_area_brain # N per m^2
        max_bin_bc = bin_center[np.argmax(density_bc)]
        error_bc = np.sqrt(counts_bc)/total_area_brain

        counts_bg, bins = np.histogram( (grad)[~brain_coral], bins=grad_bins)
        density_bg = counts_bg/total_area_background # N per m^2
        max_bin_bg = bin_center[np.argmax(density_bg)]
        error_bg = np.sqrt(counts_bg)/total_area_background

        ax.plot(bin_center, density_bc, 'k.-', label=f'Brain Coral (max: {max_bin_bc:.2f} deg)')
        ax.plot(bin_center, density_bg, 'r.-', label=f'Background  (max: {max_bin_bg:.2f} deg)')

        ax.set_ylim([1e-6, max(density_bg)*1.2])
        ax.legend(loc='best', fontsize=14)
    else:
        # histogram of relative elevation
        counts, bins = np.histogram(grad, bins=grad_bins)
        density = counts/total_area # per m^2
        bin_center = (bins[:-1]+bins[1:])/2

        # create plot
        ax.plot(bin_center, density, 'k-')
        ax.set_ylim([1e-6, max(density)*1.1])

    ax.set_xlabel(r'Slope (deg)', fontsize=14)
    ax.set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, ls='--')
    ax.set_xlim([0,30])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'gradient_distribution.png'))
    plt.close()

    # slope vs size
    fig,ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title(f"{image_name}", fontsize=16)
    grad_bins = np.linspace(0,30,151)

    if np.any(brain_coral):
        counts_bc, bins = np.histogram( (grad)[brain_coral], bins=grad_bins)
        bin_center = (bins[:-1]+bins[1:])/2

        # compute cumulative distribution from smallest to largest
        cdf_bc = np.cumsum(counts_bc[::-1])[::-1]/total_area_brain

        density_bc = counts_bc/total_area_brain # N per m^2
        max_bin_bc = bin_center[np.argmax(density_bc)]
        error_bc = np.sqrt(counts_bc)/total_area_brain

        counts_bg, bins = np.histogram( (grad)[~brain_coral], bins=grad_bins)

        # compute cumulative distribution from smallest to largest
        cdf_bg = np.cumsum(counts_bg[::-1])[::-1]/total_area_background

        density_bg = counts_bg/total_area_background # N per m^2
        max_bin_bg = bin_center[np.argmax(density_bg)]
        error_bg = np.sqrt(counts_bg)/total_area_background

        ax.plot(bin_center, cdf_bc, 'k-', label=f'Brain Coral')
        ax.plot(bin_center, cdf_bg, 'r-', label=f'Background ')

        ax.legend(loc='best', fontsize=14)
    else:
        # histogram of relative elevation
        counts, bins = np.histogram(grad, bins=grad_bins)
        density = counts/total_area # per m^2
        bin_center = (bins[:-1]+bins[1:])/2

        # create plot
        ax.plot(bin_center, density, 'k-')

    ax.set_xlabel(r'Slope (deg)', fontsize=14)
    ax.set_ylabel(r'Cumulative Rock Density (# per m$^2$)', fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, ls='--')
    ax.set_xlim([0,30])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,'cdf_gradient_distribution.png'))
    plt.close()



    # save metrics to compare to other images

    # print(f"Mean size of rock shadow: {np.mean(ellipse_area):.2f} px")
    # print(f"Median size of rock shadow: {np.median(ellipse_area):.2f} px")
    # print(f"Standard deviation of rock shadow {np.std(ellipse_area):.2f} px")
    # print(f"Max size of rock shadow: {np.max(ellipse_area):.2f} px")
    # print(f"Min size of rock shadow: {np.min(ellipse_area):.2f} px")
    # print(f"Number of rock shadows: {len(ellipse_area)}")
    # print(f"Total area of rock shadows: {total_rock_area:.2f} m^2")
    # print(f"Total area of rock shadows (ellipse): {total_ellipse_area:.2f} m^2")
    # print(f"Total area of image: {total_area:2f} m^2")
    # print(f"Percent of image covered by rock (pixel): {total_rock_area/total_area*100:.2f}%")
    # print(f"Percent of image covered by rock (ellipse): {total_ellipse_area/total_area*100:.2f}%")
    # # compute rock density per square km
    # print(f"Rock density (pixel): {len(rock_data['pixel_sizes'])/total_area*1e6:.0f} rocks/km^2")
    # print(f"Rock density (pixel): {len(rock_data['pixel_sizes'])/total_area*1e4:.0f} rocks/100m^2")
    # # compute average distance between rocks
    # print(f"Average distance between rocks: {np.sqrt(total_area/len(rock_data['pixel_sizes'])):.2f} m")
    # print(f"Average distance between rocks: {np.sqrt(total_area/len(rock_data['pixel_sizes'])/0.3):.2f} px")

    # # save rock data
    # rock_data['area_per_pixel'] = area_per_pixel
    # rock_data['total_image_area'] = total_area
    # rock_data['total_rock_area'] = total_rock_area
    # rock_data['total_ellipse_area'] = total_ellipse_area
    # rock_data['rock_density_km2'] = len(rock_data['pixel_sizes'])/total_area*1e6
    # rock_data['rock_density_100m2'] = len(rock_data['pixel_sizes'])/total_area*1e4
    # rock_data['average_rock_distance_m'] = np.sqrt(total_area/len(rock_data['pixel_sizes']))

    # save pickle of rock data
    # with open(os.path.join(outdir,'rock_data.pkl'), 'wb') as f:
    #     pickle.dump(rock_data, f)


    # TODO calculate absolute gradient instead of from relative elevation