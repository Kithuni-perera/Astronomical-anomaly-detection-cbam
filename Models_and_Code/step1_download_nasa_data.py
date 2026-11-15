import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.sdss import SDSS  # connect with SDSS (Sloan Digital Sky Survey)

# 1. Set up the directory to save downloaded telescope datasets
output_dir = "../02_Datasets/sample_galaxies"
os.makedirs(output_dir, exist_ok=True)

# 2. Sky coordinates (RA, Dec in degrees) of real target galaxies
# (M51 Whirlpool Galaxy, NGC 4565 Needle Galaxy, NGC 4631, M101, NGC 3310)
galaxy_coords = [
    ("M51_Whirlpool", 202.4696, 47.1952),
    ("NGC_4565_Needle", 189.0866, 25.9874),
    ("NGC_4631_Whale", 190.5333, 32.5415),
    ("M101_Pinwheel", 210.8023, 54.3489),
    ("NGC_3310", 159.6911, 53.5034)
]

print("Connecting to NASA / SDSS Telescope Data Archive...")

for name, ra, dec in galaxy_coords:
    pos = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    try:
        # Query and download the r-band (Red filter) FITS image
        imgs = SDSS.get_images(coordinates=pos, band='r', radius=2*u.arcmin)
        if imgs:
            file_path = os.path.join(output_dir, f"{name}_r.fits")
            imgs[0].writeto(file_path, overwrite=True)
            print(f"✅ Successfully downloaded: {name}")
        else:
            print(f"⚠️ No data found for: {name}")
    except Exception as e:
        print(f"❌ Error downloading {name}: {e}")

# 3. Load and visualize the first downloaded FITS image
sample_file = os.path.join(output_dir, "M51_Whirlpool_r.fits")
if os.path.exists(sample_file):
    with fits.open(sample_file) as hdul:
        image_data = hdul[0].data
        print(f"\nImage Dimensions (Shape): {image_data.shape}")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image_data, cmap='inferno', origin='lower', vmin=0, vmax=1.5)
        plt.colorbar(label='Flux (Brightness)')
        plt.title("SDSS Telescope Image: M51 Whirlpool Galaxy")
        plt.savefig(os.path.join(output_dir, "M51_preview.png"))
        plt.show()