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

# Conditional Probability
# P(Survived | Female) Probability of female survival
females = df_clean[df_clean['Sex'] == 'female']
males = df_clean[df_clean['Sex'] == 'male']
p_survived_females = females['Survived'].mean()
p_survived_males = males['Survived'].mean()

print(f'P(Survived | Female): {p_survived_females:.4f} -> {p_survived_females*100:.1f}%')
print(f'P(Survived | Male): {p_survived_males:.4f} -> {p_survived_males*100:.1f}%')

# Probability of Survival by class
for cls in [1,2,3]:
    subset = df_clean[df_clean['Pclass'] == cls]
    p = subset['Survived'].mean()
    print(f'P(Survived | Class {cls}): {p:.4f} -> {p*100:.1f}%')
    females_by_class = females[females['Pclass'] == cls]
    p_survived_females_by_class = females_by_class['Survived'].mean()
    print(f'P(Survived | Female | Class {cls}): {p_survived_females_by_class:.4f} -> {p_survived_females_by_class*100:.1f}%')


# Bayes Theorem
# P(A|B) = P(B|A) * P(A) / P(B)
# Real question - If someone survived, what are the chances that they were female?
# P(Female | Survived) = P(Survived | Female) * P(Female) / P(Survived)
p_female = len(females)/ total
p_male = len(males) / total
print(f'P(Female): {p_female:.4f} -> {p_female*100:.1f}%')
print(f'P(Male): {p_male:.4f} -> {p_male*100:.1f}%')
print(f'P(Survived | Female): {p_survived_females:4f}%')
print(f'P(Survived): {p_survived:.4f}')

# Applying bayes formula
p_female_given_survived = (p_survived_females * p_female) / p_survived
# Verifying by direct calculation
direct_result = len(df_clean[(df_clean['Survived']==1) & (df_clean['Sex']=='female')]) / survived

print(f'\nBayes Result: {p_female_given_survived:.4f} -> {p_female_given_survived*100:.1f}')
print(f'Direct Result: {direct_result:.4f} -> {direct_result*100:.1f}%')
print(f'Match? {abs(p_female_given_survived - direct_result) < 0.0001}')