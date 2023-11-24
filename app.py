from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('cancer_webapp2.h5')

# Load the trained StandardScaler
scaler = joblib.load('scaler.joblib')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from the JSON data and convert to numeric types
        features = [
            float(data['GENDER']),
            float(data['AGE']),
            float(data['SMOKING']),
            float(data['YELLOW_FINGERS']),
            float(data['ANXIETY']),
            float(data['PEER_PRESSURE']),
            float(data['CHRONIC_DISEASE']),
            float(data['FATIGUE']),
            float(data['ALLERGY']),
            float(data['WHEEZING']),
            float(data['ALCOHOL_CONSUMING']),
            float(data['COUGHING']),
            float(data['SHORTNESS_OF_BREATH']),
            float(data['SWALLOWING_DIFFICULTY']),
            float(data['CHEST_PAIN'])
        ]

        # Convert features to a NumPy array and reshape
        features = np.array(features).reshape(1, -1)

        # Preprocess features using the fitted scaler
        features_scaled = scaler.transform(features)

        # Make predictions
        predictions = model.predict(features_scaled)
        probability_positive_class = predictions.flatten()[0]  # Extract probability for the positive class
        response = {
            'predictions': np.round(predictions).flatten().tolist(),
            'probability_positive_class': probability_positive_class.tolist() * 100
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
