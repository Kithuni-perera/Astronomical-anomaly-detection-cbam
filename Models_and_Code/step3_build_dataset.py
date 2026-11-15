import os
import io
import time
import requests
import pandas as pd

# 1. Create directories for Training (Normal Galaxies) and Testing (Mixed/Anomalies)
train_dir = "../02_Datasets/train_normal"
test_dir = "../02_Datasets/test_mixed"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Helper function to query SDSS DR17 SkyServer SQL API directly in CSV format
def query_sdss_sql(sql_query):
    url = "http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
    params = {
        "cmd": sql_query,
        "format": "csv"
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    # Skip the first comment line (#Table1) returned by SDSS SkyServer
    lines = [line for line in response.text.splitlines() if not line.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(lines)))

# 2. Query SDSS SQL Database for Normal Galaxies (Training Set)
print("Querying SDSS DR17 database for normal galaxy coordinates...")
query_normal = """
SELECT TOP 500
    p.objid, p.ra, p.dec, p.petroR90_r
FROM PhotoObj AS p
WHERE 
    p.type = 3
    AND p.clean = 1
    AND p.r BETWEEN 14.0 AND 17.5
    AND p.petroR90_r > 5.0
"""
normal_df = query_sdss_sql(query_normal)
normal_df.to_csv("../02_Datasets/normal_galaxies_catalog.csv", index=False)
print(f"✅ Retrieved {len(normal_df)} normal galaxy coordinates.")

# 3. Query SDSS SQL Database for Unclassified / Unusual / High-Redshift Objects (Test Set)
print("Querying SDSS DR17 database for potential anomalies and unusual objects...")
query_test = """
SELECT TOP 200
    p.objid, p.ra, p.dec, s.class, s.z
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
    (s.class = 'QSO' OR s.z > 0.5)
    AND p.r BETWEEN 15.0 AND 18.5
"""
test_df = query_sdss_sql(query_test)
test_df.to_csv("../02_Datasets/test_anomalies_catalog.csv", index=False)
print(f"✅ Retrieved {len(test_df)} test object coordinates.")

# 4. Function to download 64x64 cutout images directly from SDSS SkyServer
def download_cutouts(df, save_folder, prefix):
    base_url = "http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
    success_count = 0
    for idx, row in df.iterrows():
        params = {
            "ra": row["ra"],
            "dec": row["dec"],
            "scale": 0.396,  # Native SDSS pixel scale (arcsec/pixel)
            "width": 64,
            "height": 64
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                file_path = os.path.join(save_folder, f"{prefix}_{int(row['objid'])}.jpg")
                with open(file_path, "wb") as f:
                    f.write(response.content)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Downloaded {success_count}/{len(df)} images into {save_folder}...")
        except Exception:
            pass
        time.sleep(0.05)  # Polite delay to avoid server rate-limiting
    print(f"🎉 Completed! Total {success_count} images saved in {save_folder}.\n")

# 5. Execute downloads
print("Starting Training Set Download (Normal Galaxies)...")
download_cutouts(normal_df, train_dir, "norm")

print("Starting Test Set Download (Mixed / Anomalous Candidates)...")
download_cutouts(test_df, test_dir, "test")