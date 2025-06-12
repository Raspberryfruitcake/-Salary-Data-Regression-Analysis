from typing import List, Tuple
import math

Point = Tuple[float, int]

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def Logistic_Regression(points: List[Point], epochs: int = 5000, lr: float = 0.05) -> Tuple[float, float]:
    w = 0.0
    b = 0.0
    n = len(points)
    for _ in range(epochs):
        w_grad = 0.0
        b_grad = 0.0
        for x, y in points:
            x_scaled = x / 10000
            y_cap = sigmoid(w * x_scaled + b)
            w_grad += (1 / n) * x_scaled * (y_cap - y)
            b_grad += (1 / n) * (y_cap - y)
        w -= lr * w_grad
        b -= lr * b_grad
    return w, b

def accuracy(y_pred: List[int], y_true: List[int]) -> float:
    correct_predictions = sum(p == t for p, t in zip(y_pred, y_true))
    return correct_predictions / len(y_pred)

def predict(points: List[Point], model: Tuple[float, float]) -> List[float]:
    w_pred, b_pred = model
    preds: List[float] = []
    for x, _ in points:
        x_scaled = x / 10000
        exponent = w_pred * x_scaled + b_pred
        y_pred = sigmoid(exponent)
        preds.append(y_pred)
    bin_preds = [0 if p <= 0.5 else 1 for p in preds]
    print("Accuracy of the model: ", accuracy(bin_preds, [y for _, y in points]))
    print()
    return preds

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

def model_plot(model: Tuple[float, float], X_train: List[float], X_test: List[float], y_train: List[int], y_test: List[int]) -> None:  # pragma: no cover
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    w, b = model
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='lightgreen', label='Test Data')
    plt.xlabel('Salary')
    plt.ylabel('Salary<60000')
    plt.title('Scatter Plot')
    import numpy as np  # optional, may raise ImportError if numpy not installed
    x = np.linspace(10000, 150000, 100)
    y = sigmoid(w * (x / 10000) + b)
    plt.plot(x, y, color='red', label='Log Reg Decision Boundary')
    plt.grid()
    plt.legend()
    plt.show()
