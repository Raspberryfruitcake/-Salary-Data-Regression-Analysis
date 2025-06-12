from typing import List, Tuple
import math

Point = Tuple[float, float]

def mse_score(m: float, b: float, points: List[Point]) -> float:
    ss = 0.0
    n = len(points)
    for x, y in points:
        ss += (y - (m * x + b)) ** 2
    return ss / float(n)

def r2_score(m: float, b: float, points: List[Point]) -> float:
    y_mean = sum(y for _, y in points) / len(points)
    rss = 0.0
    tss = 0.0
    for x, y in points:
        rss += (y - (m * x + b)) ** 2
        tss += (y - y_mean) ** 2
    return 1 - (rss / tss)

def mae_score(m: float, b: float, points: List[Point]) -> float:
    abs_diff = 0.0
    for x, y in points:
        abs_diff += abs(y - (m * x + b))
    return abs_diff / float(len(points))

def rmse_score(m: float, b: float, points: List[Point]) -> float:
    return math.sqrt(mse_score(m, b, points))

def Linear_Regression(points: List[Point], epochs: int = 1000, lr: float = 0.01) -> Tuple[float, float]:
    m = 0.0
    b = 0.0
    n = len(points)
    for _ in range(epochs):
        grad_m = 0.0
        grad_b = 0.0
        for x, y in points:
            grad_m += -(2 / n) * x * (y - (m * x + b))
            grad_b += -(2 / n) * (y - (m * x + b))
        m -= lr * grad_m
        b -= lr * grad_b
    return m, b

def model_evaluate(points: List[Point], parameters: Tuple[float, float]) -> None:
    m, b = parameters
    print("For the Test Dataset:\n")
    print("MSE:", mse_score(m, b, points))
    print("R2:", r2_score(m, b, points))
    print("MAE:", mae_score(m, b, points))
    print("RMSE:", rmse_score(m, b, points))

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

def model_plot(model: Tuple[float, float], X_train: List[float], X_test: List[float], y_train: List[float], y_test: List[float]) -> None:  # pragma: no cover
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    m, b = model
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='lightgreen', label='Test Data')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Scatter Plot')
    X_all = X_train + X_test
    X_line = sorted(X_all)
    y_line = [m * x + b for x in X_line]
    plt.plot(X_line, y_line, color='red', label='Regression Line')
    plt.legend()
    plt.show()
