# Use the Python 3.8 base image
FROM python:3.8

# Copy the current directory into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install the required dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Set the command to run when the container starts: python app.py
CMD [ "python", "app.py" ]