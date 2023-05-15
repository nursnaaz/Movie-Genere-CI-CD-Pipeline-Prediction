FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 6000
CMD gunicorn --workers=4 --bind 0.0.0.0:6000 app:app