# app.py
from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load pickled objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Extract encoder classes for dropdowns
fat_classes = list(encoders['fat'].classes_)
product_classes = list(encoders['product'].classes_)
outlet_classes = list(encoders['outlet'].classes_)


@app.route('/')
def welcome():
    # Page 1: Welcome page
    return render_template('welcome.html')


@app.route('/input', methods=['GET'])
def input_page():
    # Page 2: Input page with dropdowns
    return render_template(
        'input.html',
        fat_options=fat_classes,
        product_options=product_classes,
        outlet_options=outlet_classes
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve inputs
        Weight = float(request.form['Weight'])
        FatContent_raw = request.form['FatContent']
        ProductVisibility = float(request.form['ProductVisibility'])
        ProductType_raw = request.form['ProductType']
        EstablishmentYear = int(request.form['EstablishmentYear'])
        OutletSize_raw = request.form['OutletSize']

        # Encoders
        fat_le = encoders['fat']
        type_le = encoders['product']
        out_le = encoders['outlet']

        def safe_transform(le, value):
            if value in le.classes_:
                return int(le.transform([value])[0])
            else:
                # Fallback to first class if unseen
                return int(0)

        FatContent = safe_transform(fat_le, FatContent_raw)
        ProductType = safe_transform(type_le, ProductType_raw)
        OutletSize = safe_transform(out_le, OutletSize_raw)

        # Build feature vector
        X = np.array([[Weight, FatContent, ProductVisibility,
                       ProductType, EstablishmentYear, OutletSize]], dtype=float)

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        pred = model.predict(X_scaled)
        mrp_pred = round(float(pred[0]), 2)

        # Page 3: Output page
        return render_template('output.html', mrp=mrp_pred)

    except Exception as e:
        return f"Error during prediction: {e}"


if __name__ == '__main__':
    app.run(debug=True)
