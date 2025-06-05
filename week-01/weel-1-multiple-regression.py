import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Sample data: [time (min), caffeine (mg), age (years), stretches]
data = np.array([
    [30, 100, 25, 2],   # Student 1: 30 min, 100mg caffeine, 25 years, 2 stretches
    [45, 150, 30, 3],   # Student 2: 45 min, 150mg, 30 years, 3 stretches
    [60, 200, 35, 5],   # Student 3: 60 min, 200mg, 35 years, 5 stretches
    [75, 100, 40, 6],   # Student 4: 75 min, 100mg, 40 years, 6 stretches
    [90, 250, 28, 8],   # Student 5: 90 min, 250mg, 28 years, 8 stretches
    [120, 300, 45, 10]  # Student 6: 120 min, 300mg, 45 years, 10 stretches
])

# Split data into inputs (X) and output (y)
X = data[:, :3]  # Time, caffeine, age
y = data[:, 3]   # Stretches

# Create and train the multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict stretches for a sample input
sample_input = np.array([[60, 150, 30]])  # Example: 60 min, 150mg caffeine, 30 years
predicted_stretches = model.predict(sample_input)

# Print model details
print(f"Coefficients: Time = {model.coef_[0]:.4f}, Caffeine = {model.coef_[1]:.4f}, Age = {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Prediction for 60 min, 150mg caffeine, 30 years: {predicted_stretches[0]:.2f} stretches")

# 3D Visualization (Time and Caffeine vs. Stretches, holding Age constant at mean)
mean_age = np.mean(X[:, 2])  # Average age
time_range = np.linspace(30, 120, 20)
caffeine_range = np.linspace(100, 300, 20)
time_grid, caffeine_grid = np.meshgrid(time_range, caffeine_range)
X_grid = np.c_[time_grid.ravel(), caffeine_grid.ravel(), np.full_like(time_grid.ravel(), mean_age)]
y_grid = model.predict(X_grid).reshape(time_grid.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual stretches')
ax.plot_surface(time_grid, caffeine_grid, y_grid, color='red', alpha=0.5, label='Regression plane')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Caffeine (mg)')
ax.set_zlabel('Stretches')
ax.set_title('Multiple Linear Regression: Stretches vs. Time, Caffeine, Age')
plt.legend()
plt.show()