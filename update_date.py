import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime
import mwclient
import time
from transformers import pipeline
import numpy as np
from statistics import mean

def update_wikipedia_data():
    """Update Wikipedia edits data"""
    print("Updating Wikipedia data...")
    
    site = mwclient.Site("en.wikipedia.org")
    site.rate_limit_wait = True
    site.rate_limit_grace = 60
    page = site.pages["Bitcoin"]

    revs = []
    continue_param = None
    start_date = '2010-01-01T00:00:00Z'

    while True:
        params = {
            'action': 'query', 
            'prop': 'revisions', 
            'titles': page.name, 
            'rvdir': 'newer', 
            'rvprop': 'ids|timestamp|flags|comment|user', 
            'rvlimit': 500, 
            'rvstart': start_date
        }
        if continue_param:
            params.update(continue_param)

        response = site.api(**params)

        for page_id in response['query']['pages']:
            if 'revisions' in response['query']['pages'][page_id]:
                revs.extend(response['query']['pages'][page_id]['revisions'])

        if 'continue' in response:
            continue_param = response['continue']
            time.sleep(2)
        else:
            break

    # Process revisions
    revs_df = pd.DataFrame(revs)
    
    # Sentiment analysis
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    def find_sentiment(text):
        if not text or str(text) == 'nan':
            return 0
        try:
            sent = sentiment_pipeline([str(text)[:250]])[0]
            score = sent["score"]
            if sent["label"] == "NEGATIVE":
                score *= -1
            return score
        except:
            return 0

    edits = {}
    for index, row in revs_df.iterrows():
        date = time.strftime("%Y-%m-%d", time.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ"))
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)

        edits[date]["edit_count"] += 1
        comment = row.get("comment", "")
        if isinstance(comment, float) and np.isnan(comment):
            comment = ""
        edits[date]["sentiments"].append(find_sentiment(comment))

    # Aggregate by date
    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0
        del edits[key]["sentiments"]

    edits_df = pd.DataFrame.from_dict(edits, orient="index")
    edits_df.index = pd.to_datetime(edits_df.index)
    
    # Fill missing dates
    dates = pd.date_range(start="2010-03-08", end=datetime.today())
    edits_df = edits_df.reindex(dates, fill_value=0)
    
    # Apply rolling average
    rolling_edits = edits_df.rolling(30, min_periods=30).mean()
    rolling_edits = rolling_edits.dropna()
    
    rolling_edits.to_csv("wikipedia_edits.csv")
    print("Wikipedia data updated successfully")
    return rolling_edits

def update_bitcoin_model():
    """Update Bitcoin price prediction model"""
    print("Updating Bitcoin prediction model...")
    
    # Load Bitcoin price data
    btc_ticker = yf.Ticker("BTC-USD")
    btc = btc_ticker.history(period='max')
    btc = btc.reset_index()
    btc['Date'] = btc['Date'].dt.tz_localize(None)
    del btc["Dividends"]
    del btc["Stock Splits"]
    btc.columns = [c.lower() for c in btc.columns]
    
    # Load sentiment data
    bit_sent = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
    
    # Merge datasets
    btc['date'] = btc['date'].dt.normalize()
    btc = btc.merge(bit_sent, left_on='date', right_index=True, how='left')
    btc[['sentiment', 'neg_sentiment', 'edit_count']] = btc[['sentiment', 'neg_sentiment', 'edit_count']].fillna(0)
    btc = btc.set_index('date')
    
    # Create target
    btc["tomorrow"] = btc["close"].shift(-1)
    btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)
    btc = btc.dropna(subset=['target'])
    
    # Create features
    def compute_rolling(btc_data):
        horizons = [2, 7, 60, 365]
        new_predictors = ["close", "sentiment", "neg_sentiment"]

        for horizon in horizons:
            rolling_averages = btc_data.rolling(horizon, min_periods=1).mean()
            ratio_column = f"close_ratio_{horizon}"
            btc_data[ratio_column] = btc_data["close"] / rolling_averages["close"]
            edit_column = f"edit_{horizon}"
            btc_data[edit_column] = rolling_averages["edit_count"]
            rolling = btc_data.rolling(horizon, closed='left', min_periods=1).mean()
            trend_column = f"trend_{horizon}"
            btc_data[trend_column] = rolling["target"]
            new_predictors.extend([ratio_column, trend_column, edit_column])

        return btc_data, new_predictors

    btc_enhanced, predictors = compute_rolling(btc.copy())
    
    # Train model
    from xgboost import XGBClassifier
    model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=100)
    model.fit(btc_enhanced[predictors], btc_enhanced["target"])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/bitcoin_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model updated successfully")
    return model

if __name__ == "__main__":
    print("Starting data update...")
    update_wikipedia_data()
    update_bitcoin_model()
    print("Update completed!")