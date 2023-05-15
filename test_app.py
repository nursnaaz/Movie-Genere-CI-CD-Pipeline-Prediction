import unittest
import requests
from app import app
from app import TextPreprocessor

API_URL = "https://movie-genere.herokuapp.com/predict_api"

class FlaskAppTestCase(unittest.TestCase):
    """
    Test case class for the Flask app.
    """

    def setUp(self):
        """
        Set up the test case by creating a test client and enabling testing mode.
        """
        self.app = app.test_client()
        self.app.testing = True

    def test_text_preprocessor_clean_text(self):
        """
        Test the clean_text method of TextPreprocessor.
        """
        preprocessor = TextPreprocessor()
        text = "It's a sunny day! Let's go for a walk."
        cleaned_text = preprocessor.clean_text(text)
        self.assertEqual(cleaned_text, "sunny day lets go walk")

    def test_text_preprocessor_remove_stopwords(self):
        """
        Test the remove_stopwords method of TextPreprocessor.
        """
        preprocessor = TextPreprocessor()
        text = "It's a sunny day! Let's go for a walk."
        cleaned_text = preprocessor.clean_text(text)
        cleaned_text_without_stopwords = preprocessor.remove_stopwords(cleaned_text)
        self.assertEqual(cleaned_text_without_stopwords, "sunny day lets go walk")

    def test_predict_endpoint_with_preprocessing(self):
        """
        Test the prediction endpoint of the Flask app with text preprocessing.
        """
        # Create a dictionary of data to be sent.
        data = {'overview': "It's a sunny day! Let's go for a walk."}

        # Send the request.
        response = requests.post(API_URL, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('genre', list(response.json().keys())[0])

    def test_home_page(self):
        """
        Test the home page of the Flask app.
        """
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Movie Genre Prediction', response.data)

    def test_predict_endpoint(self):
        """
        Test the prediction endpoint of the Flask app.
        """
        # Create a dictionary of data to be sent.
        data = {'overview': "A comedy movie"}

        # Send the request.
        response = requests.post(API_URL, data=data)
        result = list(response.json().values())[0][0][0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual('Comedy', result)


if __name__ == '__main__':
    unittest.main()
