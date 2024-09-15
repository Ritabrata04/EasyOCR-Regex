import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import easyocr
import numpy as np
import joblib
from tqdm import tqdm
import re

# Load the trained model
pipeline = joblib.load(r'C:\Users\Ritabrata\Downloads\66e31d6ee96cd_student_resource_3\trained_model.pkl')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en','fr','de','es'], gpu=True)

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

# Function to format prediction
def format_prediction(predicted_value, entity_name):
    target_unit = entity_unit_map.get(entity_name)
    if target_unit is None:
        # If the entity_name is not recognized, return an empty string
        return ''
    # Format predicted value to two decimal places
    prediction = f"{predicted_value:.2f} {target_unit}"
    return prediction

# Read test CSV file
test_df = pd.read_csv(r'student_resource 3\dataset\test.csv')

# Prepare output list
output_data = []

# Process each image
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing test images'):
    index = row['index']
    image_link = row['image_link']
    entity_name = row['entity_name']
    try:
        response = requests.get(image_link, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        ocr_result = reader.readtext(np.array(image), detail=0)
        ocr_text = ' '.join(ocr_result)
    except Exception as e:
        print(f"Error processing image at index {index}: {e}")
        ocr_text = ''

    # Prepare input text
    text_input = ocr_text + ' ' + entity_name

    # Predict numerical value
    try:
        predicted_value = pipeline.predict([text_input])[0]
        # Format the prediction
        prediction = format_prediction(predicted_value, entity_name)
    except Exception as e:
        print(f"Error predicting value at index {index}: {e}")
        prediction = ''

    # Append to output data
    output_data.append({'index': index, 'prediction': prediction})

# Create output DataFrame
output_df = pd.DataFrame(output_data)

# Save output CSV
output_csv = 'output_predictions.csv'
output_df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")
