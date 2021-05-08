from heart.models import *
from typing import Optional
from sklearn.utils.validation import check_is_fitted
from heart.data import make_dataset
from heart.parameters import TrainingParameters, FeatureTypes, TrainTestSplit

DATASET_PATH = 'data/raw/heart.csv'
TRAIN_PARAMS = TrainingParameters()
FEATURE_PARAMS = FeatureTypes(target='target',
                              categorical=[],
                              numerical=['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'],
                              one_hot_encoding=['sex', 'thal', 'slope', 'cp', 'fbs', 'restecg', 'exang'])
TRAIN_SPLIT_PARAMS = TrainTestSplit()


def test_make_model_one(module: str, method: str, model_parameters: Optional[dict]):
    model = make_model(module, method, model_parameters)
    assert model.__class__.__name__ == method
    model_params = model.get_params()
    if model_parameters:
        for key, val in model_parameters.items():
            assert val == model_params[key], 'Incorrect model parameter in make model'


def test_many_make_models():
    test_make_model_one('sklearn.linear_model', 'LogisticRegression', None)
    test_make_model_one('sklearn.linear_model', 'LogisticRegression', {'C': 5.0, 'fit_intercept': False})
    test_make_model_one('sklearn.ensemble', 'RandomForestClassifier', None)
    test_make_model_one('sklearn.ensemble', 'RandomForestClassifier', {'n_estimators': 50, 'criterion': 'gini'})
    test_make_model_one('sklearn.svm', 'SVC', None)
    test_make_model_one('sklearn.svm', 'SVC', {'degree': 5})


def test_train():
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(DATASET_PATH, TRAIN_SPLIT_PARAMS, FEATURE_PARAMS)
    model = train(train_x, train_y, TRAIN_PARAMS)
    check_is_fitted(model, 'estimators_')


def test_predict():
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(DATASET_PATH, TRAIN_SPLIT_PARAMS, FEATURE_PARAMS)
    model = train(train_x, train_y, TRAIN_PARAMS)
    predictions = predict(model, test_x)
    assert predictions.shape[0] == test_y.shape[0], 'Incorrect prediction return shape'


def test_evaluate():
    train_x, train_y, test_x, test_y, val_x, val_y = make_dataset(DATASET_PATH, TRAIN_SPLIT_PARAMS, FEATURE_PARAMS)
    model = train(train_x, train_y, TRAIN_PARAMS)
    predictions = predict(model, test_x)
    eval_json = evaluate(predictions, test_y)
    assert 'accuracy' in eval_json.keys(), 'Incorrect eval return'
    assert 'report' in eval_json.keys(), 'Incorrect eval return'
    assert len(eval_json.keys()) == 2, 'Incorrect eval return'
    assert isinstance(eval_json, dict), 'Incorrect eval return'
