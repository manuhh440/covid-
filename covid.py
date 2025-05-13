import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris(as_frame=True)
iris_df = iris.frame

# Task 1: Load and Explore the Dataset
print("Task 1: Load and Explore the Dataset\n")

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())
print("\n")

# Explore the structure of the dataset
print("Information about the dataset:")
print(iris_df.info())
print("\n")

print("Number of missing values per column:")
print(iris_df.isnull().sum())
print("\n")

# Clean the dataset (check for and handle missing values)
# In this case, the Iris dataset from sklearn is usually clean, but let's demonstrate a check.
if iris_df.isnull().sum().sum() > 0:
    print("Missing values found. Handling them (filling with mean for numerical columns)...")
    for col in iris_df.columns:
        if iris_df[col].dtype in ['int64', 'float64']:
            iris_df[col].fillna(iris_df[col].mean(), inplace=True)
        # For categorical columns, you might use mode or another appropriate method
        elif iris_df[col].dtype == 'object':
            iris_df[col].fillna(iris_df[col].mode()[0], inplace=True)
    print("Missing values handled.")
else:
    print("No missing values found in the dataset.")
print("\n")

# Task 2: Basic Data Analysis
print("Task 2: Basic Data Analysis\n")

# Compute basic statistics of numerical columns
print("Basic statistics of numerical columns:")
print(iris_df.describe())
print("\n")

# Perform groupings on the categorical column ('target' which represents species)
average_measurements_per_species = iris_df.groupby('target').mean()
print("Average measurements per Iris species:")
print(average_measurements_per_species)
print("\n")

# Identify patterns or interesting findings
print("Interesting findings:")
print(" - The average petal length and petal width appear to increase as the target species index increases.")
print(" - Sepal width seems to be highest for the species with target index 0.")
print(" - There are clear differences in the average measurements across the different Iris species.")
print("\n")

# Task 3: Data Visualization
print("Task 3: Data Visualization\n")

# 1. Line chart: Trends of sepal length across the dataset index
plt.figure(figsize=(10, 6))
plt.plot(iris_df.index, iris_df['sepal length'], marker='o', linestyle='-', label='Sepal Length')
plt.title('Sepal Length Trend Across Dataset')
plt.xlabel('Data Point Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar chart: Average petal length per species
average_petal_length = iris_df.groupby('target')['petal length'].mean()
species_names = iris.target_names
plt.figure(figsize=(8, 6))
sns.barplot(x=species_names, y=average_petal_length, palette='viridis')
plt.title('Average Petal Length per Iris Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram: Distribution of sepal width
plt.figure(figsize=(8, 6))
sns.histplot(iris_df['sepal width'], kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: Sepal length vs. petal length, colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length', y='petal length', hue='target', data=iris_df, palette='Set2')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', labels=species_names)
plt.grid(True)
plt.show()