import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Load data
car = pd.read_csv('quikr_car.csv')

# Data Cleaning
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car = car[car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '').str.isnumeric()]
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '').astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(' ').str.slice(0, 3).str.join(' ')
car.columns = car.columns.str.strip().str.lower()
car = car.reset_index(drop=True)

# Save cleaned data
car.to_csv('quikr_car_cleaned.csv', index=False)

# Features & target
x = car.drop(['price'], axis=1)
y = car['price']

# Preprocessor (with handle_unknown='ignore')
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['name', 'fuel_type', 'company']),
    remainder='passthrough'
)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Pipeline
pipe = make_pipeline(column_trans, LinearRegression())
pipe.fit(x_train, y_train)

# Model evaluation
y_pred = pipe.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
pkl.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
print("Model saved as LinearRegressionModel.pkl")

# Example prediction (apply same cleaning to new data)
def preprocess_input(name, fuel, company, year, kms):
    name = ' '.join(name.split()[:3])  # match training cleaning
    return pd.DataFrame([[name, fuel, company, year, kms]],
                        columns=['name', 'fuel_type', 'company', 'year', 'kms_driven'])

new_data = preprocess_input("Maruti Suzuki Swift VDI", "Diesel", "Maruti Suzuki", 2015, 50000)
print("Predicted Price:", pipe.predict(new_data))
