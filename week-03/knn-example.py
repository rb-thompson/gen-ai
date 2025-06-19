import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

# Import CSV data and create a DataFrame
df = pd.read_csv('team_performance.csv')

# Preprocessing and SMOTE for imbalance
X = df[['commits', 'issues_resolved', 'total_hours']]
y = df['target_achieved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Original train shape: {X_train.shape}, Resampled train shape: {X_train_res.shape}")

# kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_res, y_train_res)

# Predict and evaluate
y_pred = knn.predict(X_test)
print("Classification Report (After SMOTE):")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Visualize
plt.figure(figsize=(10, 6))
# Train data (resampled): Circles
plt.scatter(X_train_res[y_train_res == 1, 0], X_train_res[y_train_res == 1, 1], c='blue', marker='o', label='Train Yes')
plt.scatter(X_train_res[y_train_res == 0, 0], X_train_res[y_train_res == 0, 1], c='red', marker='o', label='Train No')
# Test data: Squares
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='darkblue', marker='s', label='Test Yes')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='green', marker='s', label='Test No')
plt.xlabel('Commits (Scaled)')
plt.ylabel('Issues Resolved (Scaled)')
plt.title('KNN Classification of Target Achieved (After SMOTE)')
plt.legend()
plt.grid(True)
plt.show()

# Analyze culture impact
culture_impact = df.groupby('team_culture')['target_achieved'].mean().reset_index()
print("\nCulture Impact on Target Achieved (Mean):")
print(culture_impact)