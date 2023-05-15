from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from sklearn.metrics import accuracy_score, f1_score
import os

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

mlb_new = None
pipeline = None

def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

def load_model():
    global mlb_new, pipeline
    # Load the saved MultiLabelBinarizer
    mlb_new = pickle.load(open("mlb.pkl", 'rb'))
    print(mlb_new)
    # Load the saved model
    pipeline = pickle.load(open("model_pipeline.pkl", 'rb'))


# # Load the dataset
# data = pd.read_csv('movies_metadata.csv')

# # Extract required columns
# data = data[['overview', 'genres']]

# # Clean the data
# def parse(x):
#     names = []
#     x = eval(x)
#     for dictionary in x:
#         names.append(dictionary['name'])
#     return names

# data['target'] = data['genres'].apply(parse)
# data = data[data['target'].apply(lambda x: len(x)) > 0]
# data = data.dropna(subset=['overview'])

# Text cleaning and stopword removal
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.apply(self.clean_text)
        return X_cleaned
    
    def clean_text(self, text):
        text = re.sub("\'", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = ' '.join(text.split())
        text = text.lower()
        text = self.remove_stopwords(text)
        return text
    
    def remove_stopwords(self, text):
        tokens = word_tokenize(text)
        tokens_cleaned = [token for token in tokens if token.lower() not in self.stopwords]
        return ' '.join(tokens_cleaned)

# # Split the dataset
# overview = data['overview']
# genres = data['target']
# X_train, X_test, y_train, y_test = train_test_split(overview, genres, test_size=0.5, random_state=42)

# # Apply MultiLabelBinarizer to target labels
# mlb = MultiLabelBinarizer()
# y_train_binarized = mlb.fit_transform(y_train)

# # Create the pipeline
# pipeline = Pipeline([
#     ('cleaner', TextPreprocessor()),
#     ('tfidf', TfidfVectorizer(max_df=0.8, max_features=10000)),
#     ('model', OneVsRestClassifier(LogisticRegression()))
# ])

# # Train the model
# pipeline.fit(X_train, y_train_binarized)

# # Save the pipeline and MultiLabelBinarizer
# pickle.dump(pipeline, open('model_pipeline.pkl', 'wb'))
# pickle.dump(mlb, open('mlb.pkl', 'wb'))

# # Load the saved MultiLabelBinarizer
# mlb_new = pickle.load(open("mlb.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # dat = request.get_json()
    # print(dat)
    overview =  request.form["overview"]
    print(overview)
    # Transform the input overview
    overview_cleaned = pd.Series(overview).apply(pipeline.named_steps['cleaner'].clean_text)
    # Predict the genres
    y_pred_prob = pipeline.predict_proba(overview_cleaned)
    t = 0.3  # threshold value
    y_pred_new = (y_pred_prob >= t).astype(int)
    # Convert the binary predictions back to genre labels
    predicted_genres = mlb_new.inverse_transform(y_pred_new)
    # Format the predicted genres as a list
    predicted_genres = [list(genres) for genres in predicted_genres]
    return render_template('index.html', overview=overview, predicted_genres=predicted_genres)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # dat = request.get_json()
    # print(dat)
    overview =  request.form["overview"]
    # Transform the input overview
    overview_cleaned = pd.Series(overview).apply(pipeline.named_steps['cleaner'].clean_text)
    # Predict the genres
    y_pred_prob = pipeline.predict_proba(overview_cleaned)
    t = 0.3  # threshold value
    y_pred_new = (y_pred_prob >= t).astype(int)
    # Convert the binary predictions back to genre labels
    predicted_genres = mlb_new.inverse_transform(y_pred_new)
    # Format the predicted genres as a list
    predicted_genres = [list(genres) for genres in predicted_genres]
    res = {}
    res['genre'] = predicted_genres
    return res

# @app.route('/train', methods=['POST'])
# def train():
#     # Split the dataset
#     overview = data['overview']
#     genres = data['target']
#     X_train, X_test, y_train, y_test = train_test_split(overview, genres, test_size=0.5, random_state=42)

#     # Apply MultiLabelBinarizer to target labels
#     y_train_binarized = mlb.fit_transform(y_train)

#     # Train the model
#     pipeline.fit(X_train, y_train_binarized)

#     # Save the updated pipeline and MultiLabelBinarizer
#     pickle.dump(pipeline, open('model_pipeline.pkl', 'wb'))
#     pickle.dump(mlb, open('mlb.pkl', 'wb'))

#     return jsonify({'message': 'Training completed successfully!'})

if __name__ == '__main__':
    download_nltk_data()
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
