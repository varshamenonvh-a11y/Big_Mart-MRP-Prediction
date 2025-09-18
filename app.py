from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


# Load pickled objects
with open('model.pkl', 'rb') as f:
 model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
 scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
 encoders = pickle.load(f)


# Extract encoder classes for populating selects
fat_classes = list(encoders['fat'].classes_)
product_classes = list(encoders['product'].classes_)
outlet_classes = list(encoders['outlet'].classes_)


@app.route('/')
def welcome():
# Page 1: Welcome page
 return render_template('welcome.html')


@app.route('/input', methods=['GET', 'POST'])
def input_page():
 if request.method == 'POST':
# Collect form data and redirect to prediction route
  form_data = request.form.to_dict()
# Pass form via query params or session; we'll POST to /predict
 return redirect(url_for('predict'))


# Render Page 2: Input page, pass the encoder categories to populate dropdowns
 return render_template('input.html', fat_options=fat_classes, product_options=product_classes, outlet_options=outlet_classes)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
# This route accepts POST from input.html
 if request.method == 'POST':
# Retrieve and convert inputs
  try:
      Weight = float(request.form['Weight'])
      FatContent_raw = request.form['FatContent']
      ProductVisibility = float(request.form['ProductVisibility'])
      ProductType_raw = request.form['ProductType']
      EstablishmentYear = int(request.form['EstablishmentYear'])
      OutletSize_raw = request.form['OutletSize']
 except Exception as e:
    return f'Invalid input: {e}'


# Encode categorical using saved label encoders
fat_le = encoders['fat']
type_le = encoders['product']
out_le = encoders['outlet']

# If user provided a category not seen during training, handle it
def safe_transform(le, value):
 if value in le.classes_:
  return int(le.transform([value])[0])
 else:
# If unseen, append temporarily to classes_ and map to a new label
# Simple fallback: map to most frequent class (0)
 return int(0)


FatContent = safe_transform(fat_le, FatContent_raw)
ProductType = safe_transform(type_le, ProductType_raw)
OutletSize = safe_transform(out_le, OutletSize_raw)


# Build feature array in the same column order as training data
# Columns were: ['Weight','FatContent','ProductVisibility','ProductType','EstablishmentYear','OutletSize']
X = np.array([[Weight, FatContent, ProductVisibility, ProductType, EstablishmentYear, OutletSize]], dtype=float)


# Scale
X_scaled = scaler.transform(X)


# Predict
pred = model.predict(X_scaled)
mrp_pred = float(pred[0])


# Round or format as needed
mrp_pred_rounded = round(mrp_pred, 2)


# Render output.html with prediction
return render_template('output.html', mrp=mrp_pred_rounded)


# If GET, simply redirect to input
return redirect(url_for('input_page'))

# If user provided a category not seen during training, handle it
app.run(debug=True)