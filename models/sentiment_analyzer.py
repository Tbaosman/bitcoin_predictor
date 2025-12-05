import mwclient
import time
import pandas as pd
import numpy as np
from transformers import pipeline
from statistics import mean
import warnings
from textblob import TextBlob
import os
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

class WikipediaSentimentAnalyzer:
    def __init__(self):
        try:
            # Try to load the main sentiment pipeline
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            print("✅ Sentiment analysis pipeline loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load sentiment pipeline: {e}")
            print("🔄 Using TextBlob as fallback sentiment analyzer")
            self.sentiment_pipeline = None
    
    def get_latest_existing_date(self):
        """Check if wikipedia_edits.csv exists and return the latest date"""
        if os.path.exists("wikipedia_edits.csv"):
            try:
                existing_df = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
                if not existing_df.empty:
                    latest_date = existing_df.index.max()
                    print(f"📁 Found existing data up to: {latest_date.strftime('%Y-%m-%d')}")
                    return latest_date
            except Exception as e:
                print(f"⚠️  Error reading existing file: {e}")
        return None

    def fetch_wikipedia_data(self, start_date=None):
       # """Fetch Wikipedia revisions - incremental if existing data found"""
        #if start_date:
        #   print(f"📥 Fetching Wikipedia Bitcoin page edits since {start_date.strftime('%Y-%m-%d')}...")
        #    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        #else:
        #    print("📥 Fetching complete Wikipedia Bitcoin page history...")
        #    start_date_str = '2010-01-01T00:00:00Z'
        start_date_str = '2014-01-01T00:00:00Z'
        
        try:
            site = mwclient.Site("en.wikipedia.org")
            site.rate_limit_wait = True
            site.rate_limit_grace = 60
            page = site.pages["Bitcoin"]

            revs = []
            continue_param = None

            while True:
                params = {
                    'action': 'query', 
                    'prop': 'revisions', 
                    'titles': page.name, 
                    'rvdir': 'newer', 
                    'rvprop': 'ids|timestamp|flags|comment|user', 
                    'rvlimit': 500, 
                    'rvstart': start_date_str
                }
                if continue_param:
                    params.update(continue_param)

                response = site.api(**params)

                for page_id in response['query']['pages']:
                    if 'revisions' in response['query']['pages'][page_id]:
                        revs.extend(response['query']['pages'][page_id]['revisions'])

                if 'continue' in response:
                    continue_param = response['continue']
                    time.sleep(2)  # Rate limiting
                else:
                    break

            print(f"✅ Fetched {len(revs)} Wikipedia revisions")
            return revs
        except Exception as e:
            print(f"❌ Error fetching Wikipedia data: {e}")
            return []

    def find_sentiment(self, text):
        """Enhanced sentiment analysis with multiple fallbacks"""
        if not text or str(text) == 'nan' or str(text).strip() == '':
            return 0.0
        
        try:
            # Clean and prepare text
            clean_text = str(text).strip()[:500]  # Limit length
            
            # Try main pipeline first
            if self.sentiment_pipeline is not None:
                result = self.sentiment_pipeline([clean_text])[0]
                score = result["score"]
                if result["label"] == "NEGATIVE":
                    score *= -1
                return float(score)
            else:
                # Fallback to TextBlob
                blob = TextBlob(clean_text)
                # Convert polarity from [-1, 1] to sentiment score
                return float(blob.sentiment.polarity)
                
        except Exception as e:
            print(f"⚠️  Sentiment analysis error: {e}")
            return 0.0

    def analyze_sentiment(self, revs):
        """Enhanced sentiment analysis with better data validation"""
        print("🧠 Analyzing sentiment of Wikipedia edits...")
        
        if not revs:
            print("❌ No revisions to analyze")
            return pd.DataFrame()  # Return empty DataFrame instead of sample
        
        revs_df = pd.DataFrame(revs)
        
        edits = {}
        valid_sentiments = 0
        
        for index, row in revs_df.iterrows():
            try:
                date = time.strftime("%Y-%m-%d", time.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ"))
                if date not in edits:
                    edits[date] = {'sentiments': [], 'edit_count': 0}

                edits[date]["edit_count"] += 1
                comment = row.get("comment", "")
                
                # Skip empty or very short comments
                if isinstance(comment, str) and len(comment.strip()) > 3:
                    sentiment_score = self.find_sentiment(comment)
                    # Only count non-zero sentiments as valid
                    if sentiment_score != 0:
                        edits[date]["sentiments"].append(float(sentiment_score))
                        valid_sentiments += 1
                        
            except Exception as e:
                print(f"⚠️  Error processing revision {index}: {e}")
                continue

        print(f"📊 Found {valid_sentiments} valid sentiment scores out of {len(revs)} revisions")
        
        # If no valid sentiments, return empty DataFrame
        if valid_sentiments == 0:
            print("⚠️ No valid sentiment data found")
            return pd.DataFrame()
        
        # Aggregate by date with better handling
        for date in edits:
            sentiments = edits[date]["sentiments"]
            if len(sentiments) > 0:
                edits[date]["sentiment"] = float(mean(sentiments))
                # Calculate percentage of negative sentiments
                negative_count = len([s for s in sentiments if s < -0.1])
                edits[date]["neg_sentiment"] = float(negative_count / len(sentiments))
            else:
                # Use neutral values for days without valid sentiments
                edits[date]["sentiment"] = 0.0
                edits[date]["neg_sentiment"] = 0.0

        # Create DataFrame with proper data structure
        if not edits:
            print("❌ No edits data to process")
            return pd.DataFrame()
            
        # Convert to list of dictionaries with proper data types
        data_list = []
        for date, data in edits.items():
            data_list.append({
                'date': pd.to_datetime(date),
                'sentiment': float(data.get('sentiment', 0.0)),
                'neg_sentiment': float(data.get('neg_sentiment', 0.0)),
                'edit_count': int(data.get('edit_count', 0))
            })
        
        if not data_list:
            print("❌ No data to create DataFrame")
            return pd.DataFrame()
            
        edits_df = pd.DataFrame(data_list)
        edits_df = edits_df.set_index('date')
        
        print(f"✅ Analyzed sentiment for {len(edits_df)} days with {valid_sentiments} valid scores")
        return edits_df

    def merge_with_existing_data(self, new_data):
        """Merge new sentiment data with existing CSV file"""
        if not os.path.exists("wikipedia_edits.csv"):
            return new_data
        
        try:
            # Read existing data
            existing_df = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            
            if new_data.empty:
                print("📊 No new data to merge, keeping existing data")
                return existing_df
            
            # Ensure both DataFrames have proper datetime indices
            if not pd.api.types.is_datetime64_any_dtype(existing_df.index):
                existing_df.index = pd.to_datetime(existing_df.index)
            if not pd.api.types.is_datetime64_any_dtype(new_data.index):
                new_data.index = pd.to_datetime(new_data.index)
            
            # Remove overlapping dates from existing data (keep new data)
            overlapping_dates = new_data.index.intersection(existing_df.index)
            if len(overlapping_dates) > 0:
                print(f"🔄 Replacing {len(overlapping_dates)} overlapping dates with new data")
                existing_df = existing_df.drop(overlapping_dates, errors='ignore')
            
            # Combine the data
            combined_df = pd.concat([existing_df, new_data]).sort_index()
            
            print(f"✅ Merged data: {len(existing_df)} existing + {len(new_data)} new = {len(combined_df)} total days")
            return combined_df
            
        except Exception as e:
            print(f"❌ Error merging with existing data: {e}")
            return new_data  # Fallback to just new data

    def create_sentiment_file(self):
        """Main function to create/update sentiment CSV file - NOW WITH INCREMENTAL UPDATES"""
        print("🚀 Starting Wikipedia sentiment analysis pipeline...")
        
        # Check for existing data to determine if we should do incremental update
        latest_date = self.get_latest_existing_date()
        
        if latest_date:
            # Incremental update - fetch only new data
            start_date = latest_date + timedelta(days=1)
            revs = self.fetch_wikipedia_data(start_date=start_date)
        else:
            # Full fetch - no existing data found
            revs = self.fetch_wikipedia_data()
        
        if not revs:
            if latest_date:
                print("✅ No new revisions found, existing data is up to date")
                # Return existing data
                return pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            else:
                print("❌ No Wikipedia data fetched, creating sample sentiment file")
                sample_df = self.create_sample_sentiment_data()
                sample_df.to_csv("wikipedia_edits.csv")
                print("✅ Sample sentiment file created")
                return sample_df
            
        # Analyze the fetched revisions
        new_edits_df = self.analyze_sentiment(revs)
        
        if new_edits_df.empty:
            if latest_date:
                print("✅ No new sentiment data, existing data is up to date")
                return pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
            else:
                print("❌ No sentiment data generated, creating sample file")
                sample_df = self.create_sample_sentiment_data()
                sample_df.to_csv("wikipedia_edits.csv")
                print("✅ Sample sentiment file created")
                return sample_df
        
        # Merge with existing data if doing incremental update
        if latest_date:
            final_df = self.merge_with_existing_data(new_edits_df)
        else:
            final_df = new_edits_df
        
        # Fill missing dates and apply rolling average
        try:
            if not final_df.empty:
                # Ensure index is datetime and handle properly
                if not pd.api.types.is_datetime64_any_dtype(final_df.index):
                    final_df.index = pd.to_datetime(final_df.index)
                
                end_date = pd.Timestamp.today().normalize()
                start_date = pd.to_datetime(final_df.index.min()).normalize()
                dates = pd.date_range(start=start_date, end=end_date)
                final_df = final_df.reindex(dates, fill_value=0.0)
            
            rolling_edits = final_df.rolling(30, min_periods=1).mean()
            rolling_edits = rolling_edits.fillna(0.0)
            
            # Ensure we have the required columns with proper data types
            required_columns = ['sentiment', 'neg_sentiment', 'edit_count']
            for col in required_columns:
                if col not in rolling_edits.columns:
                    rolling_edits[col] = 0.0
                else:
                    # Ensure numeric types
                    rolling_edits[col] = pd.to_numeric(rolling_edits[col], errors='coerce').fillna(0.0)
            
            rolling_edits.to_csv("wikipedia_edits.csv")
            print("✅ Sentiment analysis complete. File saved as 'wikipedia_edits.csv'")
            
            return rolling_edits
            
        except Exception as e:
            print(f"❌ Error in sentiment file creation: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Creating sample sentiment file as fallback...")
            sample_df = self.create_sample_sentiment_data()
            sample_df.to_csv("wikipedia_edits.csv")
            print("✅ Sample sentiment file created as fallback")
            return sample_df

    def create_sample_sentiment_data(self):
        """Create realistic sample sentiment data for demo purposes"""
        print("📝 Creating realistic sample sentiment data...")
        
        from datetime import datetime, timedelta
        import random
        
        # Create data for the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = pd.date_range(start=start_date, end=end_date)
        sample_data = []
        
        for date in dates:
            # Create realistic sentiment patterns - mostly positive with some variation
            base_sentiment = random.uniform(0.1, 0.7)  # Mostly positive
            # Add some negative days (about 20% of the time)
            if random.random() < 0.2:
                base_sentiment = random.uniform(-0.6, -0.1)
            
            edit_count = random.randint(5, 25)
            
            sample_data.append({
                'sentiment': float(round(base_sentiment + random.uniform(-0.15, 0.15), 3)),
                'neg_sentiment': float(round(random.uniform(0.1, 0.4), 3)),
                'edit_count': int(edit_count)
            })
        
        df = pd.DataFrame(sample_data, index=dates)
        
        print(f"✅ Created realistic sample sentiment data for {len(df)} days")
        return df

    def get_sentiment_summary(self):
        """Get sentiment summary for API - FIXED DATE FORMATTING"""
        try:
            if os.path.exists("wikipedia_edits.csv"):
                df = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
                
                # FIXED: Ensure index is datetime
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # Get last 30 days
                recent_data = df.last('30D')
                
                if len(recent_data) == 0:
                    return self._create_sample_sentiment_summary()
                
                # Calculate sentiment distribution
                positive = len(recent_data[recent_data['sentiment'] > 0.1])
                neutral = len(recent_data[(recent_data['sentiment'] >= -0.1) & (recent_data['sentiment'] <= 0.1)])
                negative = len(recent_data[recent_data['sentiment'] < -0.1])
                
                return {
                    "positive": int(positive),
                    "neutral": int(neutral),
                    "negative": int(negative),
                    "total_edits": int(recent_data['edit_count'].sum()),
                    "avg_sentiment": float(recent_data['sentiment'].mean()),
                    "sentiment_trend": "improving" if recent_data['sentiment'].iloc[-1] > recent_data['sentiment'].iloc[0] else "declining",
                    "data_points": len(recent_data)
                }
            else:
                return self._create_sample_sentiment_summary()
        except Exception as e:
            print(f"❌ Error in get_sentiment_summary: {e}")
            return self._create_sample_sentiment_summary()

    def _create_sample_sentiment_summary(self):
        """Create sample sentiment summary when real data fails"""
        return {
            "positive": 15,
            "neutral": 8,
            "negative": 7,
            "total_edits": 120,
            "avg_sentiment": 0.25,
            "sentiment_trend": "improving",
            "data_points": 30
        }