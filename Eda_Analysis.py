import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Remove irrelevant columns
df = df.drop(['Country', 'Year', 'Status'], axis=1)

# Handle missing values
df = df.dropna()

# Separate the target variable from the rest of the data
target = df['Life expectancy']
df = df.drop(['Life expectancy'], axis=1)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the explained variance ratio of each principal component
plt.plot(explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

# Plot a heat map of the correlations between the variables
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.show()

# Plot histograms of each variable
df.hist(bins=20, figsize=(20, 15))
plt.show()

# Plot line graphs of each variable
for col in df.columns:
    plt.plot(df[col], label=col)
plt.legend()
plt.show()

# Plot scatter plots of the first two principal components
pca_scores = pca.transform(scaled_data)
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=target, cmap='coolwarm')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

# Select variables based on the results of PCA
selected_variables = ['Adult Mortality', 'Alcohol', 'percentage expenditure', 'Hepatitis B', ' BMI ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

# Filter unused variables
filtered_data = df[selected_variables]

# Add back the target variable
filtered_data.loc[:, 'Life expectancy'] = target

# Write the cleaned data to a new CSV file
filtered_data.to_csv('data_clean.csv', index=False)
