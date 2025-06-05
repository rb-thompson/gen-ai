import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: time spent in meetup (minutes) and number of stretches per student
# Each row is a student: [time_in_minutes, number_of_stretches]
data = np.array([
    [30, 2],   # Student 1: 30 min, 2 stretches
    [45, 3],   # Student 2: 45 min, 3 stretches
    [60, 5],   # Student 3: 60 min, 5 stretches
    [75, 6],   # Student 4: 75 min, 6 stretches
    [90, 8],   # Student 5: 90 min, 8 stretches
    [120, 10]  # Student 6: 120 min, 10 stretches
])

# Split data into inputs (X) and outputs (y)
X = data[:, 0].reshape(-1, 1)  # Time (independent variable)
y = data[:, 1]                # Stretches (dependent variable)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict stretches for a range of times
time_range = np.linspace(30, 120, 100).reshape(-1, 1)
predicted_stretches = model.predict(time_range)

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual stretches')
plt.plot(time_range, predicted_stretches, color='red', label='Linear regression')
plt.xlabel('Time in Meetup (minutes)')
plt.ylabel('Number of Stretches')
plt.title('Linear Regression: Stretches vs. Time in Coding Meetup')
plt.legend()
plt.grid(True)
plt.show()

# Print model details
print(f"Slope (stretches per minute): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Example prediction: 60 minutes -> {model.predict([[60]])[0]:.2f} stretches")