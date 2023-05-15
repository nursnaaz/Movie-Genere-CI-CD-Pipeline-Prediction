# Movie Genre Classification With CI/CD Implementation in Heroku

This repository contains a movie genre classification project that predicts the genres of movies based on their overviews. It includes a trained model, a Flask web application, and instructions for local setup and usage.

## Prerequisites
Python 3.8 or higher
Docker (optional)

## Table of Contents

1. Introduction
2. Setup
   * Local Setup
   * Docker Setup
3. Usage
    * Web Application
    * API Endpoint
4. Model Training
5. Model Evaluation
6. CI/CD
7. License


# Introduction
The movie genre classification project uses natural language processing techniques to predict the genres of movies based on their overviews. It employs a machine learning model trained on a labeled dataset of movie overviews and corresponding genres.

## Setup
To use the movie genre classification project, you can follow the instructions below to set it up in your local environment or as a Docker container.

## Local Setup
1. Clone the repository to your local machine:
    
    ``` git clone https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction.git ```

2. Navigate to the project directory:

    ```cd Movie-Genere-CI-CD-Pipeline-Prediction```

3. Install the required dependencies:

    ```pip install -r requirements.txt```

4. Run the Flask web application:

    ```python app.py```
    
    
 ## Docker Setup
 
 1. Install Docker on your machine. Refer to the [official Docker documentation](https://docs.docker.com/get-docker/) for instructions specific to your operating system.
 2. Clone the repository to your local machine:

    ``` git clone https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction.git ```
    
 3. Navigate to the project directory:

    ```cd Movie-Genere-CI-CD-Pipeline-Prediction```
    
 4. Build the Docker image:

    ``` docker build -t movie-genre-classification . ```

5. Run the Docker container:

    ```  docker run -p 5555:5555 movie-genre-classification ```


## Usage

Once you have set up the movie genre classification project, you can use it in two ways: through the web application or via the API endpoint.

## Web Application
* Access the local web application by opening your web browser and visiting http://localhost:5555.

![Screenshot 2023-05-15 at 7 15 18 PM](https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction/assets/18391640/aa7b718c-4ac9-47f8-a186-277f46d05216)

* Access the Heroku web application by opening your web browser and visiting https://movie-genere.herokuapp.com/

![Screenshot 2023-05-15 at 7 15 05 PM](https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction/assets/18391640/f51bfb31-2c8b-4bfc-85d2-5b3a8e655b1d)
Enter a movie overview in the provided input field.

Click the "Predict" button to get the predicted genres for the movie.

## Local API Endpoint
You can also make predictions using the API endpoint.

Endpoint: http://localhost:5555/predict_api
Method: POST
Request Payload:
Parameter name: overview
Parameter value: A vengeful New York transit cop decides to steal a trainload of subway fares; his foster brother, a fellow cop, tries to protect him.

## Heroku Endpoint
Endpoint: https://movie-genere.herokuapp.com/predict_api
Method: POST
Request Payload:
Parameter name: overview
Parameter value: A vengeful New York transit cop decides to steal a trainload of subway fares; his foster brother, a fellow cop, tries to protect him.

### Example using cURL:

```curl -d "overview=A vengeful New York transit cop decides to steal a trainload of subway fares; his foster brother, a fellow cop, tries to protect him." -X POST https://movie-genere.herokuapp.com/predict_api```

<img width="1495" alt="Screenshot 2023-05-15 at 7 00 38 PM" src="https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction/assets/18391640/e7798cf5-c8a7-46fc-932e-577606377ac4">


``` curl -d "overview=A vengeful New York transit cop decides to steal a trainload of subway fares; his foster brother, a fellow cop, tries to protect him." -X POST http:///localhost:5555/predict_api ```

 <img width="1499" alt="Screenshot 2023-05-15 at 7 01 47 PM" src="https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction/assets/18391640/12fabc15-9622-49a6-b4c7-c1b4459e42f3">
 
## Model Training

The model for movie genre classification was trained using the provided code. Here's a summary of the training process:

* The dataset used for training is the movies_metadata.csv file, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv) (34.45MB file within the 239MB zip file).
* The data was preprocessed and cleaned by extracting the 'overview' and 'genres' columns from the dataset.
* A TextPreprocessor class was implemented to clean the text data by removing special characters, converting to lowercase, and removing stopwords.
* The dataset was split into training and testing sets using a 50% ratio.
* The target labels were binarized using MultiLabelBinarizer.
* The movie genre classification model was built using a pipeline that includes text preprocessing, TF-IDF vectorization, and a logistic regression classifier.
* The model was trained using the training dataset.
* The trained model pipeline and the MultiLabelBinarizer were saved as model_pipeline.pkl and mlb.pkl, respectively.


## Model Evaluation

After training the model, it was evaluated using the testing dataset. Here are the evaluation results:

* Accuracy: 0.16206766206766207
* F1-score: 0.5600685836094441 (micro-average F1-score for multi-label classification)

These metrics provide an indication of the model's performance in predicting the genres of movies based on their overviews.

## CI/CD
This repository implements CI/CD (Continuous Integration/Continuous Deployment) using GitHub Actions and Heroku. The CI/CD pipeline ensures that whenever a code commit is made, the code is automatically built, tested, and deployed to Heroku.

The workflow in the .github/workflows/main.yml file defines the CI/CD pipeline. It includes steps for installing dependencies, running tests, and deploying the application to Heroku.

To set up CI/CD for your own repository, you can follow these steps:

* Create a Heroku account (if you don't have one already) and create a new app.

* Set up the Heroku CLI on your local machine and log in to your Heroku account.

* Add the necessary Heroku environment variables in your GitHub repository's secrets. These variables may include the Heroku API key, Heroku app name, etc.

* Push the code to the GitHub repository, and the CI/CD pipeline will automatically trigger. The workflow will build, test, and deploy the code to your Heroku app.

![Screenshot 2023-05-15 at 7 20 04 PM](https://github.com/nursnaaz/Movie-Genere-CI-CD-Pipeline-Prediction/assets/18391640/a6ba4881-3d98-4394-a8cb-2685fa940075)


## License
The project is released under the MIT License. You are free to use, modify, and distribute the code for personal and commercial purposes.



