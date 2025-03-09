import os
from flask import Flask, render_template, request, jsonify
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the new CNN model
MODEL_PATH = 'mango_leaf_disease_model.h5'  # Path to the new model
model = load_model(MODEL_PATH)

# Weather API key (replace with your key)
WEATHER_API_KEY = '5ac40f50de444f039bd161516250703'

# Disease labels sorted alphabetically to match training class indices
DISEASES = sorted([
    'Healthy', 'Anthracnose', 'Bacterial Canker', 'Cutting Weevil',
    'Die Back', 'Gall Midge', 'Powdery Mildew', 'Sooty Mould'
])

# Image preprocessing function updated for 256x256 input
def preprocess_image(img):
    img = img.resize((256, 256))  # New model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/weather')
def get_weather():
    city = request.args.get('city', 'Mumbai')
    url = f'http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days=3'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({'error': 'Unable to fetch weather data'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        # Open and preprocess the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        
        disease = DISEASES[predicted_class]
        return jsonify({'disease': disease, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=3000)