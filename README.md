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
7. Contributing
8. License


## Introduction
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

    ```  docker run -p 5000:5000 movie-genre-classification ```





 




