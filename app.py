from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Optionally load any preprocessing pipeline (e.g., scaling, encoding)
# If you don't have a preprocessing pipeline, you can skip this step
# scaler = joblib.load('scaler.pkl')

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract features from the data (adapt this to your feature names)
    # For example, if your features are 'area', 'bedrooms', 'bathrooms', you can do:
    area = data['area']
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']

    # Prepare the features as a numpy array
    features = np.array([[area, bedrooms, bathrooms]])

    # Optionally, if you used scaling/encoding, apply those transformations here
    # For example, if you used a scaler:
    # features = scaler.transform(features)

    # Make a prediction using the model
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': prediction[0]})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
