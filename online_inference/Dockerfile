FROM python:3.9-slim

EXPOSE 5000
WORKDIR /online_inference
COPY . /online_inference

RUN pip install -r requirements.txt
CMD ["python", "app.py", "sources/transformer.pkl", "sources/model_logreg.pkl"]
