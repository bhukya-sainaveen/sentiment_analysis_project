# sentiment_analysis.py

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def calculate_sentiment(text):
    # Function to calculate sentiment of given text
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        return 1
    elif sentiment_dict['compound'] <= -0.05:
        return 0
    else:
        return 1
