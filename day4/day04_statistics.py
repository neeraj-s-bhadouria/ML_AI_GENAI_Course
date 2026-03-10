import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles
from scipy.ndimage import label
from scipy.stats import alpha

# Loading the data and cleaning it
df = pd.read_csv('../day3/data/train.csv')
df_clean = df.copy()
df_clean = df_clean.drop(columns=['Cabin'])
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
print('Data Ready!')

age = df_clean['Age']
print('Age Statistics-----------------------------------------')
print(f'Mean: {round(age.mean(), 2)}')
print(f'Median; {age.median()}')
print(f'Mode: {age.mode()[0]}')
print(f'Std deviation: {round(age.std(), 2)}')
print(f'Variance: {round(age.var(), 2)}')

# Mean vs Median visualization
plt.figure(figsize=(10, 5))
sns.histplot(age, bins=30, color='steelblue', alpha=0.7)
plt.axvline(age.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {age.mean():.1f}')
plt.axvline(age.median(), color='green', linestyle='-', linewidth=2, label=f'Median: {age.median():.1f}')
plt.title('Age Distribution - Mean vs Median')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.show()

# Normal Distribution - real vs artificial
plt.figure(figsize=(12, 5))
# original age - without data cleaning
original_age = df['Age'].dropna()

plt.subplot(1, 2, 1)
sns.histplot(original_age, bins=30, kde=True, color='steelblue')
plt.axvline(original_age.mean(), color='red', linestyle='--', label=f'Mean: {age.mean():.1f}')
plt.title('Original Age - Without data cleaning')
plt.xlabel('Age')
plt.legend()

# Filled age
plt.subplot(1, 2, 2)
sns.histplot(df_clean['Age'], bins=30, kde=True, color='orange')
plt.axvline(df_clean['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean['Age'].mean():.1f}')
plt.title('Age after median filling')
plt.xlabel('Age')
plt.legend()

plt.tight_layout()
plt.show()

# Simulating Normal Distribution
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)

plt.figure(figsize=(10, 5))
sns.histplot(normal_data, bins=40, kde=True, color='steelblue')

# showing 68-95-99.7 rule
mean = normal_data.mean()
std  = normal_data.std()

plt.axvline(mean,       color='red',    linestyle='-',  linewidth=2, label='Mean')
plt.axvline(mean + std, color='orange', linestyle='--', linewidth=1.5, label='+1 Std')
plt.axvline(mean - std, color='orange', linestyle='--', linewidth=1.5, label='-1 Std')
plt.axvline(mean + 2*std, color='green', linestyle=':', linewidth=1.5, label='+2 Std')
plt.axvline(mean - 2*std, color='green', linestyle=':', linewidth=1.5, label='-2 Std')

plt.title('Normal Distribution — 68-95-99.7 Rule')
plt.legend()
plt.show()

print(f"Mean:    {mean:.2f}")
print(f"Std Dev: {std:.2f}")
print(f"±1 Std data: {((normal_data > mean-std) & (normal_data < mean+std)).mean()*100:.1f}%")
print(f"±2 Std data: {((normal_data > mean-2*std) & (normal_data < mean+2*std)).mean()*100:.1f}%")

# Correlation — showing 3 types
np.random.seed(42)
x = np.random.randn(100)

# 3 relationships
positive = x + np.random.randn(100) * 0.3      # strong positive
negative = -x + np.random.randn(100) * 0.3     # strong negative
no_relation = np.random.randn(100)              # no correlation

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(x, positive, alpha=0.6, color='green')
axes[0].set_title(f'Positive Correlation\nr = {np.corrcoef(x, positive)[0,1]:.2f}')

axes[1].scatter(x, negative, alpha=0.6, color='red')
axes[1].set_title(f'Negative Correlation\nr = {np.corrcoef(x, negative)[0,1]:.2f}')

axes[2].scatter(x, no_relation, alpha=0.6, color='gray')
axes[2].set_title(f'No Correlation\nr = {np.corrcoef(x, no_relation)[0,1]:.2f}')

plt.suptitle('Correlation Types', fontweight='bold')
plt.tight_layout()
plt.show()

# Outlier Detection — IQR method
fare = df_clean['Fare']

Q1  = fare.quantile(0.25)
Q3  = fare.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = fare[(fare < lower) | (fare > upper)]

print(f"Q1         : {Q1:.2f}")
print(f"Q3         : {Q3:.2f}")
print(f"IQR        : {IQR:.2f}")
print(f"Lower Bound: {lower:.2f}")
print(f"Upper Bound: {upper:.2f}")
print(f"Outliers   : {len(outliers)} passengers")
print(f"Outlier %  : {len(outliers)/len(fare)*100:.1f}%")

# Visualize
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=fare, color='steelblue')
plt.title('Fare — Outliers visible')

plt.subplot(1, 2, 2)
sns.histplot(fare, bins=50, color='steelblue')
plt.axvline(upper, color='red', linestyle='--',
            linewidth=2, label=f'Upper bound: {upper:.0f}')
plt.title('Fare Distribution')
plt.legend()

plt.tight_layout()
plt.show()