import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 150
reasons = ['Lack of time', 'Lack of interest', 'Technical difficulties', 'Poor planning', 'Budget issues', 'Team conflicts']
team_sizes = [2, 3, 4, 5, 6, 7, 8]
complexities = ['Low', 'Medium', 'High']

data = {
    'reason': np.random.choice(reasons, n_samples),
    'average_days': np.random.randint(5, 50, n_samples),  # Random days between 5 and 50
    'team_size': np.random.choice(team_sizes, n_samples),
    'complexity': np.random.choice(complexities, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
le_reason = LabelEncoder()
le_complexity = LabelEncoder()
df['reason_num'] = le_reason.fit_transform(df['reason'])
df['complexity_num'] = le_complexity.fit_transform(df['complexity'])

# Save to CSV for later use
df.to_csv('synthetic_abandonment_data.csv', index=False)
print("Dataset generated and saved as 'synthetic_abandonment_data.csv'")