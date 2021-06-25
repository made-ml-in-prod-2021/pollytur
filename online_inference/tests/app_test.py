import pytest
import json

from app import app
from heart.predict_pipeline import setup_models


@pytest.fixture(autouse=True)
def load_models():
    transformer_path = 'sources/transformer.pkl'
    model_path = 'sources/model_logreg.pkl'
    setup_models(transformer_path, model_path)


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@pytest.fixture
def req():
    with open('request.json', 'r') as f:
        return json.load(f)


def test_400_dropped_column(load_models, req, client):
    del req['columns'][0]
    del req['data'][0]
    response = client.post('/predict', json=req)

    assert response.status_code == 400, 'Validation error was expected'


def test_400_missing_column(load_models, req, client):
    req['columns'][0] = 'redhat'
    response = client.post('/predict', json=req)

    assert response.status_code == 400, 'Validation error was expected'


def test_200(load_models, req, client):
    response = client.post('/predict', json=req)
    data = response.json

    assert response.status_code == 200, 'Successful inference was expected'
    assert 'answer' in data and isinstance(data['answer'], list), 'List of labels was expeced'
