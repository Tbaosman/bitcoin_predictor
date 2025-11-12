from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

class BitcoinPredictor:
    def __init__(self):
        self.model = None
        self.btc_data = None
        self.last_update = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('models/bitcoin_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully")
        except:
            print("No pre-trained model found. Please run the update first.")
            self.model = None
    
    def get_current_data(self):
        """Get current Bitcoin data and prepare features"""
        try:
            # Load sentiment data
            sentiment_data = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            
            # Get recent Bitcoin data
            btc_ticker = yf.Ticker("BTC-USD")
            btc = btc_ticker.history(period="60d")
            btc = btc.reset_index()
            btc['Date'] = btc['Date'].dt.tz_localize(None)
            btc.columns = [c.lower() for c in btc.columns]
            
            # Merge with sentiment data
            btc['date'] = btc['date'].dt.normalize()
            btc = btc.merge(sentiment_data, left_on='date', right_index=True, how='left')
            btc[['sentiment', 'neg_sentiment', 'edit_count']] = btc[['sentiment', 'neg_sentiment', 'edit_count']].fillna(0)
            btc = btc.set_index('date')
            
            # Create features
            btc = self.create_features(btc)
            
            self.btc_data = btc
            self.last_update = datetime.now()
            return True
            
        except Exception as e:
            print(f"Error getting current data: {e}")
            return False
    
    def create_features(self, data):
        """Create technical features for prediction"""
        horizons = [2, 7, 60, 365]
        
        for horizon in horizons:
            rolling_averages = data.rolling(horizon, min_periods=1).mean()
            
            ratio_column = f"close_ratio_{horizon}"
            data[ratio_column] = data["close"] / rolling_averages["close"]
            
            edit_column = f"edit_{horizon}"
            data[edit_column] = rolling_averages["edit_count"]
            
            # For trend, we need target which we don't have for latest data
            # We'll use price movement instead
            trend_column = f"trend_{horizon}"
            data[trend_column] = (data["close"] > data["close"].shift(1)).rolling(horizon, min_periods=1).mean()
        
        return data
    
    def predict_tomorrow(self):
        """Make prediction for tomorrow's price movement"""
        if self.model is None or self.btc_data is None:
            return {"error": "Model or data not loaded. Please run update first."}
        
        try:
            # Get the latest data point
            latest_data = self.btc_data.iloc[-1:].copy()
            
            # Define predictors (should match training)
            predictors = [
                'close', 'sentiment', 'neg_sentiment', 'close_ratio_2', 
                'trend_2', 'edit_2', 'close_ratio_7', 'trend_7', 'edit_7', 
                'close_ratio_60', 'trend_60', 'edit_60', 'close_ratio_365', 
                'trend_365', 'edit_365'
            ]
            
            # Ensure all predictors are present
            for pred in predictors:
                if pred not in latest_data.columns:
                    latest_data[pred] = 0
            
            prediction = self.model.predict(latest_data[predictors])
            prediction_proba = self.model.predict_proba(latest_data[predictors])
            
            confidence = max(prediction_proba[0])
            
            result = {
                "prediction": "UP" if prediction[0] == 1 else "DOWN",
                "confidence": round(confidence * 100, 2),
                "current_price": round(latest_data['close'].iloc[0], 2),
                "last_updated": self.last_update.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize predictor
predictor = BitcoinPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """API endpoint to get prediction"""
    # Update data if it's older than 1 hour
    if predictor.last_update is None or (datetime.now() - predictor.last_update).seconds > 3600:
        predictor.get_current_data()
    
    result = predictor.predict_tomorrow()
    return jsonify(result)

@app.route('/update', methods=['POST'])
def update_model():
    """Force update of model and data"""
    try:
        # This would run your complete training pipeline
        # For now, we'll just reload the current data
        success = predictor.get_current_data()
        if success:
            return jsonify({"status": "success", "message": "Data updated successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to update data"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/status')
def status():
    """Get current system status"""
    status_info = {
        "model_loaded": predictor.model is not None,
        "data_loaded": predictor.btc_data is not None,
        "last_update": predictor.last_update.strftime("%Y-%m-%d %H:%M:%S") if predictor.last_update else "Never",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(status_info)

if __name__ == '__main__':
    # Load data on startup
    predictor.get_current_data()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))