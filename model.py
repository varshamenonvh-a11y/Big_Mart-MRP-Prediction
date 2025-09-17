import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# --- Load data (adjust path if needed) ---
mart = pd.read_csv("C:/Users/user/OneDrive/Documents/VaStUfFs/DATA SCIENCE AND AI/AI/PROJECT/Big_mart.csv")


# --- Impute missing values (same as your model.py) ---
mart['Weight'] = mart['Weight'].fillna(mart['Weight'].mean())
mart['OutletSize'] = mart['OutletSize'].fillna(mart['OutletSize'].mode()[0])


# --- Label encoding (we'll save encoders to reuse at inference) ---
fat_le = LabelEncoder()
type_le = LabelEncoder()
outsize_le = LabelEncoder()


mart['FatContent'] = fat_le.fit_transform(mart['FatContent'])
mart['ProductType'] = type_le.fit_transform(mart['ProductType'])
mart['OutletSize'] = outsize_le.fit_transform(mart['OutletSize'])


# --- Features and target ---
x = mart.drop(['ProductID', 'OutletID', 'LocationType', 'OutletType', 'MRP'], axis=1)
y = mart['MRP']


# --- Train/test split ---
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.30, random_state=15)


# --- Scaling ---
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# --- Train model ---
model = DecisionTreeRegressor()
model.fit(x_train_scaled, y_train)


# --- Save model and preprocessing objects ---
with open('model.pkl', 'wb') as f:
 pickle.dump(model, f)


with open('scaler.pkl', 'wb') as f:
 pickle.dump(scaler, f)


# Save label encoders as a dict
encoders = {'fat': fat_le, 'product': type_le, 'outlet': outsize_le}
with open('encoders.pkl', 'wb') as f:
 pickle.dump(encoders, f)


print('Saved: model.pkl, scaler.pkl, encoders.pkl')