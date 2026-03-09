from cProfile import label

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from numpy.f2py.cfuncs import includes
from scipy.stats import alpha

# Load the data and clean it
df = pd.read_csv('data/train.csv')
df_clean = df.copy()
df_clean = df_clean.drop(columns=['Cabin'])
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

print(f'Data Ready! Shape: {df_clean.shape}')
print(f'Missing values: {df_clean.isnull().sum().sum()}')

# plot 1 - People survived vs not survived
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='Survived')
plt.title('Survival Count')
plt.xticks([0, 1], ['Not Survived (0)', 'Survived (1)'])
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# plot 2 - Survival by gender
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='Sex', hue='Survived')
plt.title('Survival by gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.show()

# plot 3 - Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df_clean[df_clean['Survived'] == 0]['Age'], bins=30, color='red', alpha=0.5, label='Not survived')
sns.histplot(data=df_clean[df_clean['Survived'] == 1]['Age'], bins=30, color='green', alpha=0.5, label='Survived')
plt.title('Age Distribution - Survived vs Not Survived')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.show()

# plot 4 - Fare Box plot by Class
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean, x='Pclass', y='Fare')
plt.title('Fare Distribution by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# plot 5 - Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_cols = df_clean.select_dtypes(include='number')
correlation = numeric_cols.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# plot 6 - Subplots - all plots altogether
# Plot 6 — Subplots — sab ek saath
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Titanic — Complete EDA Dashboard', fontsize=16, fontweight='bold')
# Plot 1 — Survival by Gender
sns.countplot(data=df_clean, x='Sex', hue='Survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival by Gender')
axes[0, 0].legend(['Not Survived', 'Survived'])
# Plot 2 — Survival by Class
sns.countplot(data=df_clean, x='Pclass', hue='Survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Class')
axes[0, 1].legend(['Not Survived', 'Survived'])
# Plot 3 — Age Distribution
sns.histplot(data=df_clean[df_clean['Survived'] == 0]['Age'],
             bins=20, color='red', alpha=0.5, label='Not Survived', ax=axes[1, 0])
sns.histplot(data=df_clean[df_clean['Survived'] == 1]['Age'],
             bins=20, color='green', alpha=0.5, label='Survived', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution')
axes[1, 0].legend()
# Plot 4 — Fare by Class
sns.boxplot(data=df_clean, x='Pclass', y='Fare', ax=axes[1, 1])
axes[1, 1].set_title('Fare by Class')
plt.tight_layout()
plt.show()