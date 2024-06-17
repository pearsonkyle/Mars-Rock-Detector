import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from tqdm import tqdm

# find all directories in output/
dirs = glob.glob('output/*')
dirs = [d for d in dirs if os.path.isdir(d)]

# create subplots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].set_title("Cumulative Slope Distribution Difference", fontsize=16)
ax[1].set_title("Cumulative Slope Distribution in Brain Terrain", fontsize=16)
ax[2].set_title("Cumulative Slope Distribution in Background", fontsize=16)

grad_bins = np.linspace(0,30,61)

# load rock_data.nc from each directory and plot the distributions
for d in tqdm(dirs):
    f = os.path.join(d, 'rock_data.nc')
    if os.path.exists(f):
        data = nc.Dataset(f)

        # get the data from the netCDF file
        slope = data.variables['plane_slope_deg'][:]
        rslope = data.variables['rplane_slope_deg'][:]

        brain_coral = data.variables['brain_coral'][:].astype(bool)
        image_name = os.path.basename(d)
        outdir = d

        # calculate total area
        total_area = np.sum(data.variables['pixel_sizes'][:])
        total_area_brain = np.sum(data.variables['pixel_sizes'][brain_coral])
        total_area_background = np.sum(data.variables['pixel_sizes'][~brain_coral])
        
        if np.any(brain_coral):
            # do the same time for rslope
            counts_bc, bins = np.histogram(rslope[brain_coral], bins=grad_bins)
            density_bc = counts_bc / total_area_brain  # N per m^2

            counts_bg, bins = np.histogram(rslope[~brain_coral], bins=grad_bins)
            density_bg = counts_bg / total_area_background  # N per m^2

            # compute cumulative distribution from smallest to largest
            cdf_bg = np.cumsum(counts_bg[::-1])[::-1]/total_area_background
            cdf_bc = np.cumsum(counts_bc[::-1])[::-1]/total_area_brain
            cdf_diff = cdf_bc - cdf_bg

            bin_center = (bins[:-1] + bins[1:]) / 2

            # cdf diff
            ax[0].plot(bin_center, cdf_diff, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")}', alpha=0.75)

            ax[1].plot(bin_center, cdf_bc, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")}', alpha=0.75)
            ax[2].plot(bin_center, cdf_bg, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")}', alpha=0.75)
        else:
            # histogram of relative elevation
            counts, bins = np.histogram(slope, bins=grad_bins)
            density = counts / total_area  # per m^2
            bin_center = (bins[:-1] + bins[1:]) / 2

        data.close()


for i in range(3):
    ax[i].set_xlabel('Slope (degrees)', fontsize=14)
    ax[i].set_ylabel(r'Cumulative Rock Density (# per m$^2$)', fontsize=14)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[i].grid(True, ls='--')
    ax[i].set_xlim([0, 30])
    ax[i].legend(loc='best', fontsize=12)
ax[0].set_ylabel('Cumulative Rock Density Difference\n [Brain Terrain - Background] ' + r'(# per m$^2$)', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join("output", 'gradient_distribution_comparison.png'))
plt.close()
print(os.path.join("output", 'gradient_distribution_comparison.png'))