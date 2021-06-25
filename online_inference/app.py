from flask import Flask, request, jsonify
from heart.predict_pipeline import predict_pipeline, setup_models
import pandas as pd
import click
import time
from contextlib import contextmanager
import signal


AVAILABLE = None
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    # copypasted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.json

    try:
        values, columns = data_json['data'], data_json['columns']
        assert all(len(vals) == len(columns) for vals in values)
        data: pd.DataFrame = pd.DataFrame(values, columns=columns)
        out = predict_pipeline(data)
        response = {'answer': out.tolist()} # should be simple data type
        return jsonify(response), 200
    except AssertionError:
        return 'Validation error', 400


@app.get("/healt_check")
def liveness_check() -> bool:
    return not (AVAILABLE is None)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('transformer_path', nargs=1, type=str)
@click.argument('model_path', nargs=1, type=str)
def main(transformer_path='sources/transformer.pkl', model_path='sources/model_logreg.pkl'):
    time.sleep(30)
    with time_limit(60):
        global AVAILABLE
        AVAILABLE = setup_models(transformer_path, model_path)
        app.run(host='0.0.0.0')


if __name__ == '__main__':
    main()
