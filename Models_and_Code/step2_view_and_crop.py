import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Load the downloaded FITS file for NGC 4565 (Needle Galaxy)
fits_path = "../02_Datasets/sample_galaxies/NGC_4565_Needle_r.fits"

with fits.open(fits_path) as hdul:
    image_data = hdul[0].data
    header = hdul[0].header
    wcs = WCS(header)

print(f"Full Telescope Frame Dimensions: {image_data.shape}")

# 2. Crop a 128x128 pixel cutout centered exactly on NGC 4565 coordinates
galaxy_pos = SkyCoord(ra=189.0866*u.degree, dec=25.9874*u.degree, frame='icrs')
cutout = Cutout2D(image_data, position=galaxy_pos, size=(128, 128), wcs=wcs)
cropped_galaxy = cutout.data

print(f"Cropped Galaxy Dimensions for AI: {cropped_galaxy.shape}")

# 3. Visualize both the full frame and the cropped galaxy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Full Telescope Frame (using log scaling to see faint stars)
axes[0].imshow(np.arcsinh(image_data * 10), cmap='inferno', origin='lower')
axes[0].set_title("Full SDSS Telescope Frame (1489 x 2048)")
axes[0].set_xlabel("X Pixels")
axes[0].set_ylabel("Y Pixels")

# Plot Cropped Galaxy (Ready for AI Model)
axes[1].imshow(np.arcsinh(cropped_galaxy * 10), cmap='inferno', origin='lower')
axes[1].set_title("Cropped Target: NGC 4565 Needle Galaxy (128 x 128)")
axes[1].set_xlabel("X Pixels")
axes[1].set_ylabel("Y Pixels")

plt.tight_layout()
plt.savefig("../02_Datasets/sample_galaxies/NGC_4565_visualization.png", dpi=300)
plt.show()