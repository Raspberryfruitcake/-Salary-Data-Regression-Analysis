import csv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logistic_model import Logistic_Regression, predict

EXPECTED_W = -1.2538518496912419
EXPECTED_B = 7.44253588187274


def load_data():
    with open('Salary_Data.csv') as f:
        next(f)
        data = [float(line.split(',')[1]) for line in f]
    return data

def prepare_points(data):
    points = [(salary, 1 if salary < 60000 else 0) for salary in data]
    from random import Random
    rng = Random(42)
    indices = list(range(len(points)))
    rng.shuffle(indices)
    split = int(len(points) * 0.7)
    train = [points[i] for i in indices[:split]]
    test = [points[i] for i in indices[split:]]
    return train, test

def accuracy(pred, true):
    return sum(p == t for p, t in zip(pred, true)) / len(pred)

def test_logistic_regression_training():
    data = load_data()
    train, test = prepare_points(data)
    w, b = Logistic_Regression(train, epochs=5000, lr=0.05)
    assert abs(w - EXPECTED_W) < 1e-6
    assert abs(b - EXPECTED_B) < 1e-6
    # ensure predictions are correct
    train_probs = predict(train, (w, b))
    train_pred = [0 if p <= 0.5 else 1 for p in train_probs]
    test_probs = predict(test, (w, b))
    test_pred = [0 if p <= 0.5 else 1 for p in test_probs]
    assert accuracy(train_pred, [y for _, y in train]) == 1.0
    assert accuracy(test_pred, [y for _, y in test]) == 1.0
