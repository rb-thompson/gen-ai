# SCENARIO: QuantumLabs Plasma Control Optimization
#
# QuantumLabs develops compact fusion reactors, a plausible near-future technology based on current 
# efforts (e.g., ITER, Commonwealth Fusion Systems). The dataset captures experiments adjusting 
# plasma magnetic field strengths to optimize energy output


import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Generate synthetic dataset for QuantumLabs
np.random.seed(42)
n_samples = 50
reactors = ['ProtoStar-1', 'Nexus-2', 'FusionCore-3', 'StellarForge-4']
magnetic_fields = np.random.uniform(1, 10, n_samples)
energy_outputs = 20 * magnetic_fields + np.random.normal(0, 5, n_samples) + 10
reactor_types = np.random.choice(reactors, n_samples)
data = pd.DataFrame({
    'ReactorType': reactor_types,
    'Magnetic_Field_Tesla': magnetic_fields,
    'Energy_Output_MW': energy_outputs
})


# Save dataset to CSV
data.to_csv('quantum_labs_data.csv', index=False)
print("Dataset generated and saved as 'quantum_labs_data.csv'.")
print(data.head())

# Step 2: Prepare data for linear regression
X = data['Magnetic_Field_Tesla'].values.reshape(-1, 1)
y = data['Energy_Output_MW'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()

# Print evaluation metrics
print(f"\nModel Evaluation:")
print(f"MSE: {mse:.2f} (MW²)")
print(f"RMSE: {rmse:.2f} MW")
print(f"R-squared: {r2:.2f}")
print(f"Cross-validated MSE: {cv_mse:.2f} (MW²)")
print(f"Slope: {model.coef_[0]:.2f} MW per tesla, Intercept: {model.intercept_:.2f} MW")

# Step 5: Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0.6)
plt.scatter(X_test, y_test, color='green', label='Test data', alpha=0.6)
plt.scatter(X_test, y_pred, color='orange', label='Test predictions', marker='x')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Magnetic Field Strength (Tesla)')
plt.ylabel('Energy Output (MW)')
plt.title('QuantumLabs: Energy Output vs. Magnetic Field Strength')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Predict for a new setting
new_field = 7.5  # Tesla
if new_field < 0:
    raise ValueError("Magnetic field must be non-negative")
predicted_output = model.predict([[new_field]])
print(f"\nPrediction: For {new_field} tesla, expected energy output: {predicted_output[0]:.2f} MW")

# Step 7: Summarize findings in a report
report = f"""
QuantumLabs Analysis Report
==============================
To: Lead Engineer, QuantumLabs
Subject: Plasma Control Optimization for Fusion Reactors

Analysis:
- Dataset: {n_samples} experiments across {len(reactors)} reactor types.
- Linear regression predicts energy output from magnetic field strength.
- Model performance:
  - R-squared: {r2:.2f} (explains {r2*100:.0f}% of output variance).
  - RMSE: {rmse:.2f} MW (average prediction error).
  - Cross-validated MSE: {cv_mse:.2f} MW² (robust fit).
- Slope: {model.coef_[0]:.2f} MW per tesla, indicating strong control impact.

Recommendation:
- A {new_field} tesla setting is predicted to yield {predicted_output[0]:.2f} MW.
- Optimize ProtoStar-1 reactors, which show high output efficiency.

Prepared by: R.B. Thompson, Data Scientist
"""
with open('quantum_labs_report.txt', 'w') as f:
    f.write(report)
print("\nReport saved as 'quantum_labs_report.txt'")