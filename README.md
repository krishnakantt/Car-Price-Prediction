# Car Price Prediction using Machine Learning
This project predicts the selling price of used cars based on user inputs like brand, model, year, mileage, and fuel type. It uses Linear Regression as the machine learning model and is deployed as a Flask web application with a clean HTML/CSS interface.

The goal is to help buyers and sellers make informed decisions in the used car market.

---

## 🚀 Features
- Predicts used car prices instantly based on details entered by the user.
- Simple and interactive Flask-based web app.
- Data preprocessing and cleaning for accurate predictions.
- Machine Learning model trained on real-world Quikr Cars dataset.
- Lightweight and easy to run locally.

---

## 📂 Project Structure
```bash

├── Static/                    # CSS styles for the frontend  
├── templates/                 # HTML templates (index.html)  
├── application.py             # Main Flask application  
├── Car Price Predictor.py     # Data preprocessing & model training  
├── LinearRegressionModel.pkl  # Saved trained ML model  
├── quikr_car.csv              # Raw dataset  
├── quikr_car_cleaned.csv      # Cleaned dataset  
├── requirements.txt           # Required Python libraries
```

## 📊 Dataset
- Source: Scraped from Quikr Cars listings.
- Raw File: quikr_car.csv
- Cleaned File: quikr_car_cleaned.csv
- Features:
    - Name – Car brand and model
    - Year – Manufacturing year
    - Selling Price – Target variable (in INR)
    - KM Driven – Mileage in kilometers
    - Fuel Type – Petrol/Diesel/CNG/Electric
    - Seller Type – Dealer/Individual
    - Transmission – Manual/Automatic
    - Owner – Number of previous owners

---

## 🛠️ Installation & Setup
Follow these steps to run the project locally:
```bash
# Clone the repository
git clone https://github.com/krishnakantt/Car-Price-Prediction.git
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python application.py
```
Once running, open your browser and visit:
http://127.0.0.1:5000/



---

## 📌 How to Use
1. Open the web application.
2. Enter details about the car (name, year, km driven, fuel type, etc.).
3. Click Predict.
4. The app will display the estimated selling price.

---

## 💻 Technologies Used
- Python
- Flask – Backend framework
- HTML, CSS – Frontend interface
- Pandas, NumPy – Data handling
- scikit-learn – Machine learning model
- Pickle – Model serialization

