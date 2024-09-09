# Import required libraries
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the air quality data
def load_data():
    data = pd.read_csv('air_quality_data.csv')
    return data

# Train a linear regression model
def train_model(data):
    X = data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']]
    y = data['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    return model

# Create a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create a route for predicting the AQI
@app.route('/predict', methods=['POST'])
def predict():
    pm25 = float(request.form['pm25'])
    pm10 = float(request.form['pm10'])
    no = float(request.form['no'])
    no2 = float(request.form['no2'])
    nox = float(request.form['nox'])
    nh3 = float(request.form['nh3'])
    co = float(request.form['co'])
    so2 = float(request.form['so2'])
    o3 = float(request.form['o3'])
    data = load_data()
    model = train_model(data)
    input_values = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])
    predicted_aqi = model.predict(input_values)
    return render_template('index.html', predicted_aqi=predicted_aqi[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)