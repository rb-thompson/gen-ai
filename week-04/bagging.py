import pandas as pd
import numpy as np

# Load Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Select a small subset for demonstration
df_small = df[['Pclass', 'Sex', 'Age', 'Survived']].head(10)

# Create a bootstrap sample (without replacement)
bootstrap_sample = df_small.sample(n=len(df_small), replace=True, random_state=42)
bootstrap_sample_no_replace = df_small.sample(n=len(df_small), replace=False, random_state=42)
print("Original Subset:")
print(df_small)
print("\nBootstrap Sample:")
print(bootstrap_sample)
print("\nBootstrap Sample (No Replacement):")
print(bootstrap_sample_no_replace)