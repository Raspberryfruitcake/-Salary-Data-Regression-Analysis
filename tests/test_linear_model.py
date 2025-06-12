import csv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linear_model import Linear_Regression, r2_score

# expected values computed from the algorithm with seed 42
EXPECTED_M = 9337.119110572356
EXPECTED_B = 25569.309614786842
EXPECTED_R2 = 0.9692472823850864


def load_data():
    with open('Salary_Data.csv') as f:
        next(f)
        data = [tuple(map(float, line.split(','))) for line in f]
    return data

def split_data(data):
    from random import Random
    rng = Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    split = int(len(data) * 0.7)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [data[i] for i in train_idx]
    test = [data[i] for i in test_idx]
    return train, test

def test_linear_regression_training():
    data = load_data()
    train, test = split_data(data)
    m, b = Linear_Regression(train, epochs=1000, lr=0.01)
    assert abs(m - EXPECTED_M) < 1e-6
    assert abs(b - EXPECTED_B) < 1e-6
    r2 = r2_score(m, b, test)
    assert abs(r2 - EXPECTED_R2) < 1e-6
