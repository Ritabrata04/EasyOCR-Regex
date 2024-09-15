# Import necessary libraries
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import easyocr
from tqdm import tqdm
import re
import os
import joblib

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Define entity-unit mapping for standardization
entity_unit_map = {
    'width': 'metre',
    'depth': 'metre',
    'height': 'metre',
    'item_weight': 'kilogram',
    'maximum_weight_recommendation': 'kilogram',
    'voltage': 'volt',
    'wattage': 'watt',
    'item_volume': 'litre'
}

# Conversion factors to standard units
unit_conversion_factors = {
    # Length units to metres
    'millimetre': 0.001,
    'centimetre': 0.01,
    'metre': 1.0,
    'inch': 0.0254,
    'foot': 0.3048,
    'yard': 0.9144,
    # Weight units to kilograms
    'gram': 0.001,
    'kilogram': 1.0,
    'ounce': 0.0283495,
    'pound': 0.453592,
    # Volume units to litres
    'millilitre': 0.001,
    'centilitre': 0.01,
    'decilitre': 0.1,
    'litre': 1.0,
    'gallon': 3.78541,
    'pint': 0.473176,
    'quart': 0.946353,
    'cup': 0.236588,
    # Voltage
    'volt': 1.0,
    'millivolt': 0.001,
    'kilovolt': 1000.0,
    # Wattage
    'watt': 1.0,
    'kilowatt': 1000.0
}

# Function to parse and normalize entity_value
def parse_entity_value(value_str, target_unit):
    # Regex to extract number and unit
    match = re.match(r'^(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:\s[a-zA-Z]+)?)$', str(value_str).strip())
    if not match:
        return None  # Unable to parse
    number = float(match.group(1))
    unit = match.group(2).lower().replace(' ', '')
    # Handle unit conversions
    factor = unit_conversion_factors.get(unit)
    if factor is None:
        return None  # Unknown unit
    # Convert to standard unit
    value_in_standard_unit = number * factor
    # Convert to target unit if necessary
    target_factor = unit_conversion_factors.get(target_unit)
    if target_factor is None:
        return None  # Unknown target unit
    normalized_value = value_in_standard_unit / target_factor
    return normalized_value

# Load training data
train_df = pd.read_csv(r'student_resource 3\dataset\train.csv')

# Limit the number of images per entity_name
# First, get counts per entity_name
entity_counts = train_df['entity_name'].value_counts()

# Define the desired number of images per entity_name
desired_counts = {
    entity: min(500, count) if count >= 500 else count
    for entity, count in entity_counts.items()
}

# Adjust counts for less frequent entity_names to use more images
total_images_desired = 500 * len(entity_counts)
total_images_current = sum(desired_counts.values())

# If total_images_current < desired total images, adjust counts
if total_images_current < 4000:
    # Increase counts for less frequent entities proportionally
    scaling_factor = 4000 / total_images_current
    for entity in desired_counts:
        desired_counts[entity] = int(desired_counts[entity] * scaling_factor)
    # Recalculate total images after scaling
    total_images_current = sum(desired_counts.values())
    # Ensure we do not exceed total images desired
    if total_images_current > 5000:
        # Scale down if necessary
        scaling_factor = 5000 / total_images_current
        for entity in desired_counts:
            desired_counts[entity] = int(desired_counts[entity] * scaling_factor)

# Now, sample the data accordingly
sampled_dfs = []
for entity, desired_count in desired_counts.items():
    entity_df = train_df[train_df['entity_name'] == entity]
    sampled_df = entity_df.sample(n=desired_count, random_state=42)
    sampled_dfs.append(sampled_df)

# Combine sampled data
train_df = pd.concat(sampled_dfs).reset_index(drop=True)

# Prepare lists to store data
ocr_texts = []
normalized_values = []

# Process each training sample
for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc='Processing training data'):
    image_link = row['image_link']
    entity_name = row['entity_name']
    entity_value = row['entity_value']
    target_unit = entity_unit_map.get(entity_name)
    # Parse and normalize entity_value
    normalized_value = parse_entity_value(entity_value, target_unit)
    if normalized_value is None:
        normalized_values.append(None)
    else:
        normalized_values.append(normalized_value)
    # Download image and extract OCR text
    try:
        response = requests.get(image_link, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        ocr_result = reader.readtext(np.array(image), detail=0)
        ocr_text = ' '.join(ocr_result)
        ocr_texts.append(ocr_text)
    except Exception as e:
        print(f"Error processing image at index {idx}: {e}")
        ocr_texts.append('')  # Empty text in case of error

# Add OCR texts and normalized values to the DataFrame
train_df['ocr_text'] = ocr_texts
train_df['normalized_value'] = normalized_values

# Drop samples with missing normalized values
train_df = train_df.dropna(subset=['normalized_value']).reset_index(drop=True)

# Combine OCR text and entity_name
train_df['text_input'] = train_df['ocr_text'] + ' ' + train_df['entity_name']

# Prepare features and target
X = train_df['text_input']
y = train_df['normalized_value']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('regressor', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {mse:.4f}")

# Save the trained model
joblib.dump(pipeline, 'trained_model.pkl')

print("Training completed and model saved.")
