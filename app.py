from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

model = pickle.load(open('model.pkl', 'rb'))

FEATURES = ['sqft_living', 'bedrooms', 'bathrooms', 'floors',
            'waterfront', 'view', 'condition', 'sqft_lot', 'yr_built']

USD_TO_INR = 83  # Approx conversion

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/analytics')
def get_analytics():
    import json
    try:
        with open('analytics.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        input_df = pd.DataFrame([{
            'sqft_living': float(data['sqft_living']),
            'bedrooms':    float(data['bedrooms']),
            'bathrooms':   float(data['bathrooms']),
            'floors':      float(data['floors']),
            'waterfront':  int(data['waterfront']),
            'view':        int(data['view']),
            'condition':   int(data['condition']),
            'sqft_lot':    float(data['sqft_lot']),
            'yr_built':    int(data['yr_built']),
        }])[FEATURES]

        # Predict
        price = float(model.predict(input_df)[0])
        price = max(50000, min(price, 5000000))

        # USD values
        price_usd = round(price)
        low_usd   = round(price_usd * 0.85)
        high_usd  = round(price_usd * 1.15)

        # INR values
        price_inr = round(price_usd * USD_TO_INR)
        low_inr   = round(low_usd   * USD_TO_INR)
        high_inr  = round(high_usd  * USD_TO_INR)

        return jsonify({
            'price_usd':      price_usd,
            'price_inr':      price_inr,
            'price_low_usd':  low_usd,
            'price_high_usd': high_usd,
            'price_low_inr':  low_inr,
            'price_high_inr': high_inr,
            'currency_note':  'Approx conversion (USD -> INR)',
            'confidence':     'HIGH'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)