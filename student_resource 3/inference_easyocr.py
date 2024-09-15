# Import necessary libraries
import pandas as pd
import easyocr
import re
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import csv

# Initialize EasyOCR reader with English language only
reader = easyocr.Reader(['en','fr','de','es'], gpu=True)

# Define entity-unit mapping with additional units
entity_unit_map = {
    'width': {'centimetre', 'centimeter', 'cm', 'millimetre', 'millimeter', 'mm',
              'metre', 'meter', 'm', 'inch', 'in', '"', 'foot', 'ft', "'", 'yard'},
    'depth': {'centimetre', 'centimeter', 'cm', 'millimetre', 'millimeter', 'mm',
              'metre', 'meter', 'm', 'inch', 'in', '"', 'foot', 'ft', "'", 'yard'},
    'height': {'centimetre', 'centimeter', 'cm', 'millimetre', 'millimeter', 'mm',
               'metre', 'meter', 'm', 'inch', 'in', '"', 'foot', 'ft', "'", 'yard'},
    'item_weight': {'gram', 'g', 'kilogram', 'kg', 'microgram', 'milligram', 'mg',
                    'ounce', 'oz', 'pound', 'lb', 'ton'},
    'maximum_weight_recommendation': {'gram', 'g', 'kilogram', 'kg', 'microgram',
                                      'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'ton'},
    'voltage': {'kilovolt', 'kv', 'millivolt', 'mv', 'volt', 'v'},
    'wattage': {'kilowatt', 'kw', 'watt', 'w'},
    'item_volume': {'centilitre', 'centiliter', 'cl', 'cubicfoot', 'cubicinch', 'cup',
                    'decilitre', 'deciliter', 'dl', 'fluidounce', 'gallon', 'imperialgallon',
                    'litre', 'liter', 'l', 'microlitre', 'microliter', 'millilitre',
                    'milliliter', 'ml', 'pint', 'quart'}
}

# Function to handle common unit mistakes and variations
def common_mistake(unit, allowed_units):
    unit = unit.lower()
    unit = unit.replace(' ', '')  # Remove spaces
    unit = unit.replace('″', '"').replace('′', "'")  # Replace unicode quotation marks
    unit = unit.replace("''", '"').replace("'''", '"').replace('""', '"')
    unit = unit.replace('“', '"').replace('”', '"')  # Replace other quotation marks
    corrections = {
        'feet': 'foot',
        'foot': 'foot',
        'ft': 'foot',
        "'": 'foot',
        'inches': 'inch',
        'inch': 'inch',
        'in': 'inch',
        '"': 'inch',
        'centimeter': 'centimetre',
        'cm': 'centimetre',
        'meter': 'metre',
        'm': 'metre',
        'millimeter': 'millimetre',
        'mm': 'millimetre',
        'liter': 'litre',
        'l': 'litre',
        'milliliter': 'millilitre',
        'ml': 'millilitre',
        'centiliter': 'centilitre',
        'cl': 'centilitre',
        'deciliter': 'decilitre',
        'dl': 'decilitre',
        'microliter': 'microlitre',
        'g': 'gram',
        'kg': 'kilogram',
        'mg': 'milligram',
        'oz': 'ounce',
        'lb': 'pound',
        'kv': 'kilovolt',
        'mv': 'millivolt',
        'v': 'volt',
        'kw': 'kilowatt',
        'w': 'watt',
        # Add more corrections as needed
    }
    unit = corrections.get(unit, unit)
    if unit in allowed_units:
        return unit
    return None

# Function to parse string to extract number and unit
def parse_string(s, allowed_units):
    s_stripped = s.strip()
    if s_stripped == "":
        return None, None
    # Split by first non-digit character
    match = re.match(r'^(-?\d+(?:[\.,]\d+)?)(.*)$', s_stripped)
    if not match:
        return None, None
    number_str = match.group(1)
    unit_str = match.group(2).strip()
    if not unit_str:
        return None, None
    try:
        number = float(number_str.replace(',', '.'))
    except ValueError:
        return None, None
    unit = unit_str.lower()
    unit = unit.replace(' ', '')  # Remove spaces
    unit = unit.replace('″', '"').replace('′', "'")  # Replace unicode quotation marks
    unit = unit.replace("''", '"').replace("'''", '"').replace('""', '"')
    unit = unit.strip()
    unit = common_mistake(unit, allowed_units)
    if unit is None:
        return None, None
    return number, unit

# Function to extract measurements from OCR result
def extract_measurements(ocr_result, allowed_units):
    measurements = []
    for text in ocr_result:
        # Split text by '/' or ',' to handle multiple measurements
        parts = re.split(r'/|,', text)
        for part in parts:
            part = part.strip()
            # Use regex to find patterns like 'number unit' with optional spaces or hyphens
            matches = re.findall(r'(-?\d+(?:[\.,]\d+)?)[\s\-]*([a-zA-Z%°\'"″′]+)', part)
            for match in matches:
                number_str = match[0]
                unit_str = match[1]
                number, unit = parse_string(f"{number_str}{unit_str}", allowed_units)
                if number is not None and unit is not None:
                    measurements.append(f"{number} {unit}")
    return measurements

# Deskewing the image
def deskew(image):
    # Convert to grayscale and use edge detection to find skew angle
    import cv2
    import numpy as np
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = [np.rad2deg(np.arctan2(line[0][1], line[0][0])) for line in lines]
        median_angle = np.median(angles)
        image = image.rotate(-median_angle, expand=True)
    return image

# Read test CSV file
input_csv = r'student_resource 3\dataset\test.csv'  # Update the path as needed
df = pd.read_csv(input_csv)

# Process only the first 200 images
#df = df.head()

# Output CSV file
output_csv = 'output_predictions.csv'  # Update the path as needed

# Open the output CSV file and write the header
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['index', 'prediction']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process each image with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing images', unit='image'):
        index = row['index']
        image_link = row['image_link']  # Adjust if the column name is different
        entity_name = row['entity_name'].lower()
        allowed_units = entity_unit_map.get(entity_name, set())

        prediction = ''  # Default empty prediction

        # Download image
        try:
            response = requests.get(image_link, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            tqdm.write(f"Error downloading image at index {index}: {e}")
            writer.writerow({'index': index, 'prediction': prediction})
            continue  # Skip to the next image

        # Preprocess image
        image = deskew(image)
        gray_image = image.convert('L')  # Grayscale
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)  # Increase contrast
        enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)  # Sharpen

        # Perform OCR on the enhanced image
        try:
            ocr_result = reader.readtext(np.array(enhanced_image), detail=0)
            # Since detail=0, ocr_result is a list of strings
            recognized_texts = ocr_result
            tqdm.write(f"OCR result for index {index}: {recognized_texts}")
        except Exception as e:
            tqdm.write(f"OCR failed for image at index {index}: {e}")
            writer.writerow({'index': index, 'prediction': prediction})
            continue

        # Extract measurements relevant to the entity
        measurements = extract_measurements(recognized_texts, allowed_units)
        tqdm.write(f"Measurements extracted for index {index}: {measurements}")

        if measurements:
            # Take the first extracted measurement
            prediction = measurements[0]
        else:
            prediction = ''

        # Write the prediction to the CSV file immediately
        writer.writerow({'index': index, 'prediction': prediction})

        # Update progress bar with more detailed info
        tqdm.write(f"Processed index {index}, prediction: {prediction}")

print(f"Predictions saved to {output_csv}")
