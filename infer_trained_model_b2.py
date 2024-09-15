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
import os
import joblib

model = joblib.load(r'C:\Users\Ritabrata\Downloads\66e31d6ee96cd_student_resource_3\trained_model.pkl')  # Assuming model.pkl is your trained model

# Initialize EasyOCR reader with multiple languages
reader = easyocr.Reader(['en', 'fr', 'de', 'es'], gpu=True)
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

# Function to process a batch of data
def process_batch(batch_df, batch_number):
    output_csv = f'output_predictions_batch_{batch_number}.csv'

    # Check if the output CSV file already exists
    if os.path.exists(output_csv):
        processed_df = pd.read_csv(output_csv)
        processed_indices = set(processed_df['index'].tolist())
    else:
        processed_indices = set()

    # Exclude already processed indices
    batch_df = batch_df[~batch_df['index'].isin(processed_indices)].reset_index(drop=True)

    # Open the output CSV file in append mode if it exists, write header if not
    if os.path.exists(output_csv):
        csvfile = open(output_csv, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'prediction'])
    else:
        csvfile = open(output_csv, 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'prediction'])
        writer.writeheader()

    try:
        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f'Processing batch {batch_number}', unit='image'):
            index = row['index']
            image_link = row['image_link']
            entity_name = row['entity_name'].lower()
            allowed_units = entity_unit_map.get(entity_name, set())

            prediction = ''

            # Download image
            try:
                response = requests.get(image_link, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                tqdm.write(f"Error downloading image at index {index}: {e}")
                writer.writerow({'index': index, 'prediction': prediction})
                continue

            # Preprocess image
            image = deskew(image)
            gray_image = image.convert('L')
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2.0)
            enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)

            # Perform OCR on the enhanced image
            try:
                ocr_result = reader.readtext(np.array(enhanced_image), detail=0)
                recognized_texts = ocr_result
                tqdm.write(f"OCR result for index {index}: {recognized_texts}")
            except Exception as e:
                tqdm.write(f"OCR failed for image at index {index}: {e}")
                writer.writerow({'index': index, 'prediction': prediction})
                continue

            # Extract measurements relevant to the entity
            measurements = extract_measurements(recognized_texts, allowed_units)
            tqdm.write(f"Measurements extracted for index {index}: {measurements}")

            # Format input for the model
            input_text = ' '.join(ocr_result) + ' ' + entity_name

            # Make predictions using the model
            try:
                predicted_value = model.predict([input_text])[0]
                # Format the prediction
                prediction = f"{predicted_value:.2f}"
            except Exception as e:
                tqdm.write(f"Model prediction failed for index {index}: {e}")
                prediction = ''

            # Write the prediction to the CSV file immediately
            writer.writerow({'index': index, 'prediction': prediction})

            # Update progress bar with more detailed info
            tqdm.write(f"Processed index {index}, prediction: {prediction}")

    except KeyboardInterrupt:
        print(f"Processing of batch {batch_number} interrupted by user. Saving progress and exiting...")
    finally:
        csvfile.close()

    print(f"Predictions for batch {batch_number} saved to {output_csv}")

# Main script
if __name__ == '__main__':
    input_csv = r'C:\Users\Ritabrata\Downloads\66e31d6ee96cd_student_resource_3\student_resource 3\dataset\test.csv'
    df = pd.read_csv(input_csv)

    total_entries = len(df)
    batch_size = total_entries // 4

    # Divide the data into 4 batches
    batch1_df = df.iloc[0:batch_size].reset_index(drop=True)
    batch2_df = df.iloc[batch_size:2*batch_size].reset_index(drop=True)
    batch3_df = df.iloc[2*batch_size:3*batch_size].reset_index(drop=True)
    batch4_df = df.iloc[3*batch_size:].reset_index(drop=True)

    # Process each batch separately
    #process_batch(batch1_df, batch_number=1)
    process_batch(batch2_df, batch_number=2)
    #process_batch(batch3_df, batch_number=3)
    #process_batch(batch4_df, batch_number=4)

    print("All batches processed.")