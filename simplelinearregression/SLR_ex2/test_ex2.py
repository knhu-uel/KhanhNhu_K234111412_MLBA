import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Data setup
# Area (x) and Price (y)
x = np.array([[73.5, 75, 76.5, 79, 81.5, 82.5, 84, 85, 86.5, 87.5, 89, 90, 91.5]]).T
y = np.array([[1.49, 1.50, 1.51, 1.54, 1.58, 1.59, 1.60, 1.62, 1.63, 1.64, 1.66, 1.67, 1.68]]).T

# Manual calculation of b1 and b0
def calculate_b1_b0(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    x2bar = np.mean(x ** 2)
    xybar = np.mean(x * y)
    b1 = (xbar * ybar - xybar) / (xbar ** 2 - x2bar)
    b0 = ybar - b1 * xbar
    return b1, b0

b1, b0 = calculate_b1_b0(x, y)
print("b1 =", b1)
print("b0 =", b0)

# Predicted values (manual)
y_pred_manual = b0 + b1 * x
print("Manual predicted values:\n", y_pred_manual)

# Train a linear regression model using sklearn
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x, y)
y_pred_sklearn = regr.predict(x)

print("\nSklearn model results:")
print("Coefficient (b1):", regr.coef_)
print("Intercept (b0):", regr.intercept_)
print("Predicted values:\n", y_pred_sklearn)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'r-o', label='price actual')
plt.plot(x, y_pred_sklearn, 'b--', label='price predict')

# Mean line
ybar = np.mean(y)
plt.axhline(ybar, linestyle='--', linewidth=2, color='gray', label='mean actual')

# Labels and title
plt.xlabel('Area (m²)', fontsize=12)
plt.ylabel('Price (Billion VND)', fontsize=12)
plt.title('House price by Area', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
