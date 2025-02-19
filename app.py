from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import io
import base64

# Initialize the Flask app
app = Flask(__name__)

# Helper functions for RSI, Bollinger Bands, etc.
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(data, window=20, num_sd=2):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * num_sd)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * num_sd)
    return data

def collect_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        data.ffill(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def preprocess_data(data):
    data.dropna(inplace=True)
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['RSI'] = compute_rsi(data['Close'])
    data = compute_bollinger_bands(data)
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Stochastic_Oscillator'] = (data['Close'] - data['Low'].rolling(window=14).min()) / (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())

    for lag in range(1, 6):
        data[f'Lag{lag}'] = data['Close'].shift(lag)
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data.dropna(inplace=True)

    features = ['MA10', 'MA50', 'Volatility', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'Day_of_Week', 'Month', 'MACD', 'MACD_Signal', 'Stochastic_Oscillator'] + [f'Lag{lag}' for lag in range(1, 6)]
    X = data[features]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test, data, scaler

def train_model(X_train, y_train):
    base_estimator = DecisionTreeRegressor(max_depth=5)
    model = AdaBoostRegressor(estimator=base_estimator, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_estimator_, 'adaboost_model.pkl')
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2

def plot_future_projections(processed_data, future_prices, days_to_predict):
    plt.figure(figsize=(14, 7))
    plt.plot(processed_data['Close'].values, label='Actual Prices', color='blue')
    plt.plot(range(len(processed_data), len(processed_data) + days_to_predict), future_prices, label='Predicted Future Prices', color='red')
    plt.title("Actual and Predicted Future Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    # Save the plot to a bytes buffer and encode it in base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def predict_future_prices(model, processed_data, scaler, days_to_predict):
    future_features = []
    last_row = processed_data[['MA10', 'MA50', 'Volatility', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'Day_of_Week', 'Month', 'MACD', 'MACD_Signal', 'Stochastic_Oscillator'] + [f'Lag{lag}' for lag in range(1, 6)]].iloc[-1].values

    for _ in range(days_to_predict):
        future_features.append(last_row)
        last_row = np.roll(last_row, -1)
        last_row[-1] = model.predict(scaler.transform([last_row]))[0]

    future_features_scaled = scaler.transform(future_features)
    future_prices = model.predict(future_features_scaled)
    return future_prices

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        days_to_predict = int(request.form['days_to_predict'])
        
        data = collect_data(ticker, start_date, end_date)
        
        if data is not None:
            X_train, X_test, y_train, y_test, processed_data, scaler = preprocess_data(data)
            model = train_model(X_train, y_train)
            mse, mae, r2 = evaluate_model(model, X_test, y_test)
            future_prices = predict_future_prices(model, processed_data, scaler, days_to_predict)
            plot_url = plot_future_projections(processed_data, future_prices, days_to_predict)
            
            future_dates = pd.date_range(start=processed_data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
            future_predictions = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_prices})

            return render_template('result.html', ticker=ticker, start_date=start_date, end_date=end_date, mse=mse, mae=mae, r2=r2, plot_url=plot_url, future_predictions=future_predictions.to_html(index=False))
        else:
            return render_template('index.html', error="Failed to collect data. Please check the ticker and dates.")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
