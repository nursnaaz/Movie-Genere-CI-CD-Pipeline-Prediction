import json  # Required for handling JSON data.
import logging  # Used for logging messages.
import os  # Provides functions for interacting with the operating system.
import re  # Regular expression operations for text pattern matching.

import nltk  # Natural Language Toolkit library for text processing.
import numpy as np  # Numerical computing library for array operations.
import pandas as pd  # Data manipulation library for working with structured data.
import pickle  # Used for object serialization and deserialization.

from flask import Flask, render_template, request  # Flask web framework for building web applications.
from nltk.corpus import stopwords  # Corpus of stopwords for text preprocessing.
from nltk.tokenize import word_tokenize  # Tokenization library for splitting text into words.

from sklearn.base import BaseEstimator, TransformerMixin  # Base classes for creating custom transformers.
from sklearn.feature_extraction.text import TfidfVectorizer  # Text feature extraction using TF-IDF.
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier.
from sklearn.metrics import accuracy_score, f1_score  # Metrics for model evaluation.
from sklearn.model_selection import train_test_split  # Splitting dataset into train and test sets.
from sklearn.multiclass import OneVsRestClassifier  # One-vs-Rest classifier strategy.
from sklearn.pipeline import Pipeline  # Pipeline for chaining data processing steps.
from sklearn.preprocessing import MultiLabelBinarizer  # Binarize multilabel data.

# Set up the logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger instance
logger = logging.getLogger(__name__)

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
    try:
        # Load the saved MultiLabelBinarizer
        mlb_new = pickle.load(open("mlb.pkl", 'rb'))
        # Load the saved model
        pipeline = pickle.load(open("model_pipeline.pkl", 'rb'))
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {str(e)}")
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling model: {str(e)}")

# Text cleaning and stopword removal
class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for text preprocessing.

    This class provides methods for cleaning text data and removing stopwords.

    Attributes:
        stopwords (set): A set of stopwords for English language.

    Methods:
        fit(self, X, y=None):
            Fit method required by the scikit-learn transformer interface.

        transform(self, X):
            Transform method required by the scikit-learn transformer interface.

        clean_text(self, text):
            Cleans the input text by removing special characters and converting to lowercase.

        remove_stopwords(self, text):
            Removes stopwords from the input text.
    """
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        """
        Fit method required by the scikit-learn transformer interface.

        Args:
            X (array-like or sparse matrix): Input data.
            y (array-like, optional): Target data. Defaults to None.

        Returns:
            self (object): Returns the instance itself.
        """
        return self
    
    def transform(self, X):
        """
        Transform method required by the scikit-learn transformer interface.

        Args:
            X (array-like or sparse matrix): Input data.

        Returns:
            X_cleaned (array-like or sparse matrix): Cleaned input data.
        """
        X_cleaned = X.apply(self.clean_text)
        return X_cleaned
    
    def clean_text(self, text):
        """
        Cleans the input text by removing special characters and converting to lowercase.

        Args:
            text (str): Input text.

        Returns:
            cleaned_text (str): Cleaned text.
        """
        text = re.sub("\'", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = ' '.join(text.split())
        text = text.lower()
        text = self.remove_stopwords(text)
        return text
    
    def remove_stopwords(self, text):
        """
        Removes stopwords from the input text.

        Args:
            text (str): Input text.

        Returns:
            cleaned_text (str): Text with stopwords removed.
        """
        tokens = word_tokenize(text)
        tokens_cleaned = [token for token in tokens if token.lower() not in self.stopwords]
        return ' '.join(tokens_cleaned)

@app.route('/')
def home():
    """
    Home route for the Flask application.

    Returns:
        rendered HTML template: Renders the index.html template.
    """
    logger.info('Home page accessed')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for predicting movie genres based on the input overview.

    Returns:
        rendered HTML template: Renders the result.html template with the predicted genres.
    """
    try:
        overview =  request.form["overview"]
        if not overview or not overview.strip():
            logger.error("Error: Empty input overview.")
            return render_template('index.html', error_message="Error: Empty input overview.")
        # Log the input overview
        logger.info(f'Input overview: {overview}')
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
        # Log the predicted genres
        logger.info(f'Predicted genres: {predicted_genres}')
        return render_template('index.html', overview=overview, predicted_genres=predicted_genres)
    except:
        logger.error(f"Error predicting genres: {str(e)}")
        return render_template('index.html', error_message="An error occurred while predicting the genres.")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    Endpoint for predicting movie genres based on the input overview (API).

    Returns:
        JSON response: Returns a JSON response containing the predicted genres.
    """
    try:

        overview =  request.form["overview"]
        if not overview or not overview.strip():
            logger.error("Error: Empty input overview (API).")
            return {"error": "Error: Empty input overview."}
        # Log the input overview
        logger.info(f'Input overview (API): {overview}')
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
        # Log the predicted genres
        logger.info(f'Predicted genres (API): {predicted_genres}')
        # Return the JSON response
        return res
    except Exception as e:
        logger.error(f"Error predicting genres (API): {str(e)}")
        return {"error": "An error occurred while predicting the genres."}

if __name__ == '__main__':
    download_nltk_data()
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5555)), debug=True)
