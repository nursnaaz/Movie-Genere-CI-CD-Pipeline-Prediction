FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords
ENV PORT=4080
EXPOSE $PORT
#CMD gunicorn --workers=1 --bind 0.0.0.0:$PORT app:app
CMD [ "python", "app.py" ]