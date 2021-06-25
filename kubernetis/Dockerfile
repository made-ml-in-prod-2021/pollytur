FROM python:3.6-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY sources/model_logreg.pkl /sources/model_logreg.pkl
COPY sources/transformer.pkl /sources/transformer.pkl
COPY app.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="sources/model_logreg.pkl"
ENV PATH_TO_TRANSFORMER="sources/transformer.pkl"
ENV DELAY=20
ENV WORKING_TIME=60

CMD ["bash", "-c", "sleep ${DELAY}; (uvicorn app:app  --host 0.0.0.0 --port 8000) & sleep ${WORKING_TIME}; kill $!"]