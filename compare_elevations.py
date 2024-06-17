import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from tqdm import tqdm
from scipy.optimize import curve_fit

def gaussian(x, a, std, mean):
    return a * np.exp(-((x - mean) / std) ** 2)

# find all directories in output/
dirs = glob.glob('output/*')
dirs = [d for d in dirs if os.path.isdir(d)]

# create subplots
fig, ax = plt.subplots(1, 3, figsize=(24, 6))
ax[1].set_title("Elevation Distribution in Brain Coral", fontsize=16)
ax[2].set_title("Elevation Distribution in Background", fontsize=16)
ax[0].set_title("Difference in Elevation Distributions", fontsize=16)
elevation_bins = np.linspace(-2, 2, 61)  # in meters

# load rock_data.nc from each directory and plot the distributions
for d in tqdm(dirs):
    f = os.path.join(d, 'rock_data.nc')
    if os.path.exists(f):
        data = nc.Dataset(f)

        # get the data from the netCDF file
        rel_elevation = data.variables['rel_elevation'][:]
        brain_coral = data.variables['brain_coral'][:].astype(bool)
        image_name = os.path.basename(d)
        outdir = d

        # calculate total area
        total_area = np.sum(data.variables['pixel_sizes'][:])
        total_area_brain = np.sum(data.variables['pixel_sizes'][brain_coral])
        total_area_background = np.sum(data.variables['pixel_sizes'][~brain_coral])

        if np.any(brain_coral):
            counts_bc, bins = np.histogram(rel_elevation[brain_coral], bins=elevation_bins)
            density_bc = counts_bc / total_area_brain  # N per m^2

            counts_bg, bins = np.histogram(rel_elevation[~brain_coral], bins=elevation_bins)
            density_bg = counts_bg / total_area_background  # N per m^2

            bin_center = (bins[:-1] + bins[1:]) / 2

            # find rel. elevation at max density
            max_density_bc = np.max(density_bc)
            max_density_bg = np.max(density_bg)
            max_density_bc_idx = np.argmax(density_bc)
            max_density_bg_idx = np.argmax(density_bg)
            max_density_bc_elevation = bin_center[max_density_bc_idx]
            max_density_bg_elevation = bin_center[max_density_bg_idx]

            # fit gaussian to the data
            popt_bc, pcov_bc= curve_fit(gaussian, bin_center, density_bc, p0=[max_density_bc, 1, 0.5])
            popt_bg, pcov_bg= curve_fit(gaussian, bin_center, density_bg, p0=[max_density_bg, 1, 0.])

            ax[1].plot(bin_center, density_bc, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")} ({max_density_bc_elevation:.1f} +- {popt_bc[1]:.1f} m)', alpha=0.75)
            ax[2].plot(bin_center, density_bg, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")} ({max_density_bg_elevation:.1f} +- {popt_bg[1]:.1f} m)', alpha=0.75)
            ax[0].plot(bin_center, density_bc - density_bg, '-', label=f'{image_name.replace("_RED_A_01_ORTHO","")}', alpha=0.75)

        data.close()

for i in range(3):
    ax[i].set_xlabel(r'Relative Elevation (m)', fontsize=14)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[i].grid(True, ls='--')
    ax[i].set_xlim([-2, 2])
    ax[i].legend(loc='best', fontsize=12)

ax[1].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
ax[1].set_ylim([1e-6, 0.02])

ax[2].set_ylabel(r'Rock Density (# per m$^2$)', fontsize=14)
ax[2].set_ylim([1e-6, 0.02])

ax[0].set_ylabel(r'Density Difference(# per m$^2$)' +'\n[Brain Terrain - Background]', fontsize=14)
ax[0].set_ylim([-1e-6, 0.0035])

plt.tight_layout()
filename = os.path.join("output", 'elevation_distribution_comparison.png')
plt.savefig(filename)
plt.close()
print(f"Saved {filename}")
