


import re
import constants
import os
import requests
import pandas as pd
import multiprocessing
import time
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib
from PIL import Image

# Define common unit mistake handling
def common_mistake(unit):
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

# Parse string to extract number and unit
def parse_string(s):
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError(f"Invalid format in {s}")
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError(f"Invalid unit [{unit}] found in {s}. Allowed units: {constants.allowed_units}")
    return number, unit

# Create placeholder image for failed downloads
def create_placeholder_image(image_save_path):
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
    except Exception as e:
        return

# Function to download an individual image
def download_image(image_link, save_folder, retries=3, delay=3):
    if not isinstance(image_link, str):
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        return

    for _ in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            return
        except Exception as e:
            print(f"Failed to download {image_link}: {e}")
            time.sleep(delay)
    
    create_placeholder_image(image_save_path)  # Create a placeholder if download fails

# Function to handle the downloading of multiple images
def download_images(image_links, download_folder, allow_multiprocessing=True):
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    if allow_multiprocessing:
        # Use partial to pass additional arguments to the download_image function
        download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)

        # Limit the number of processes to avoid exceeding the Windows handle limit
        with multiprocessing.Pool(processes=58) as pool:
            # Use tqdm for progress tracking
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
            pool.close()
            pool.join()
    else:
        # Sequential download without multiprocessing
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)
