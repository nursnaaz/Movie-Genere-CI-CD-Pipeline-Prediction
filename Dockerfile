FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords
CMD [ "python", "app.py" ]