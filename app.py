import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request

# Download necessary NLTK resources
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('C:\\Users\\dell\\OneDrive\\Desktop\\Dataset\\ECO.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Replace NaN values in the 'Reviews' column with empty strings
df['Reviews'] = df['Reviews'].fillna('')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis and assign sentiment labels
def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"]
    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis to each review and create a new 'Sentiment' column
df['Sentiment'] = df['Reviews'].apply(analyze_sentiment)

# Feature extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(df['Reviews']).toarray()
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Model evaluation
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment_score = sia.polarity_scores(text)["compound"]
        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return render_template('result.html', text=text, sentiment_score=sentiment_score, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
