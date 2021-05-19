from flask import Flask, request, jsonify
from heart.predict_pipeline import predict_pipeline, setup_models
import pandas as pd
import click

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



@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('transformer_path', nargs=1, type=str)
@click.argument('model_path', nargs=1, type=str)
def main(transformer_path='sources/transformer.pkl', model_path='sources/model_logreg.pkl'):
    setup_models(transformer_path, model_path)
    app.run(host='0.0.0.0')


if __name__ == '__main__':
    main()
