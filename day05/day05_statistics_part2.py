import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data and clean it
df = pd.read_csv('../day03/data/train.csv')
df_clean = df.copy()
df_clean = df_clean.drop(columns=['Cabin'])
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
print('Data Ready!')

# Probability basics
total = len(df_clean)
survived = df_clean['Survived'].sum()
not_survived = total - survived
p_survived = survived/total
p_not_survived = not_survived/total

print(f'Total passengers: {total}')
print(f'Survived: {survived}')
print(f'Not_Survived: {not_survived}')
print(f'P_Survived: {p_survived}')
print(f'P_Not_Survived: {p_not_survived}')
print(f'P(S)+P(NS): {p_survived+p_not_survived: .1f}')