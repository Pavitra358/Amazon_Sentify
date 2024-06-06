Review_Bot:
This project is a sentiment analysis web application designed to categorize product reviews as positive, negative, or neutral. Leveraging Natural Language Processing (NLP) techniques, the app utilizes three main components:

-VADER (Valence Aware Dictionary and sEntiment Reasoner): VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It provides a way to quantify the sentiment of text data in terms of polarity (positive, negative, neutral) and intensity.

-TF-IDF (Term Frequency-Inverse Document Frequency): TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. In this project, TF-IDF is used for feature extraction to represent the reviews in numerical form for input into the machine learning model.

-Random Forest Classifier: The Random Forest Classifier is a popular ensemble learning algorithm used for classification tasks. It leverages multiple decision trees to classify input data. In this project, the classifier is trained on the TF-IDF transformed data to predict the sentiment of product reviews.

-Key Features:

1.Web-based interface for easy input of product reviews.
2.Sentiment categorization into positive, negative, or neutral.
3.Actionable insights for businesses to understand consumer sentiment.
4.User-friendly interface for seamless interaction.

-Technologies Used:

Python
Flask
VADER
TF-IDF
Random Forest Classifier
HTML/CSS

-Folder Structure:

1. Folder Structure:
   - Create "SentimentAnalysisWebApp" folder.
   - Inside, add "app.py".
   - Create "static" folder, add "style.css".
   - Create "templates" folder, add "index.html" and "result.html".

2. Flask App:
   - Set up Flask routes in `app.py`.
   - Implement sentiment analysis functionality.

3. Run:
   - Launch Flask app with `python app.py`.




