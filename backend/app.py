import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"Received input: {data}")  # Print received input
        
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        
        return jsonify({"crop": prediction[0]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
