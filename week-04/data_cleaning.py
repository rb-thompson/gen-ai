import pandas as pd
import numpy as np

data = {
    'Name': ['Sammy', 'Fareej', 'Sarah', 'John', 'Sarah'],
    'Age': [31, 'twenty', 28, 25, 29],
    'Score': [85, 90, np.nan, 95, 105],
    'Date': ['2023-01-01', '2023-02-30', '2023-03-15', '2023-04-01', '2023-05-05']        
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Data cleaning is important for AI projects to ensure data quality and reliability.

neighbors = {
    'Name': ['Sammy', None, 'Sarah'],
    'City': ['Nashville', 'Cleveland', 'Richmond']
}

neighbors_df = pd.DataFrame(neighbors)
print("\nNeighbors DataFrame:")
print(neighbors_df)

print(df.isnull().sum())

# Fill missing values with the mean of other scores
mean_score = df['Score'].mean() # Calculate mean of non-missing scores
print(f"\nMean Score: {mean_score}")
df['Score'] = df['Score'].fillna(mean_score) # Fill missing Score values with the mean
print("\nDataFrame after filling missing Score values:")
print(df)

