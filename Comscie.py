import numpy as np
import matplotlib.pyplot as plt

# Generate 10 random numbers for x and y
np.random.seed(42)
x = np.random.rand(10) * 10  # Independent variable (10 random numbers between 0 and 10)
y = 3 * x + 7 + np.random.randn(10) * 2  # Dependent variable with noise

# Perform manual linear regression
n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the slope (m) and intercept (b)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator
b = y_mean - m * x_mean

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Predicted y values based on the regression line
y_pred = m * x + b

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x, y_pred, color="red", label="Regression line")
plt.title("Linear Regression (Manual Calculation)")
plt.xlabel("Independent Variable (x)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.grid()
plt.show()
