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


# A/B Testing
# Question - Is the survival rate of class 1 and class 3 significantly different or is it just random chances?
class1 = df_clean[df_clean['Pclass']==1]['Survived']
class3 = df_clean[df_clean['Pclass']==3]['Survived']
print(f'-------- A/B Testing: Class 1 vs class 3 Survival ---------')
print(f'\nClass 1 - n={len(class1)}, survival rate: {class1.mean()*100:.1f}%')
print(f'Class 3 - n={len(class3)}, survival rate: {class3.mean()*100:.1f}%')
print(f'Difference: {(class1.mean() - class3.mean())*100:.1f}%')

# t-test  - It checks if the difference is real or just random chances
t_stat, p_value = stats.ttest_ind(class1, class3)

print(f'\nt-statistics: {t_stat*100:.4f}')
print(f'p-value: {p_value:.4f}')
print(f'Conclusion: {'Significant Difference' if p_value < 0.05 else 'Probably random chance'}')


# p-value explanation - 3 examples
print('\n------ p-value explanation ------')
# Example 1 - Highly Significant (Class 1 vs Class 3)
t1, p1 = stats.ttest_ind(class1, class3)
print('Class 1 vs Class 3')
print(f'p-value = {p1:.6f} -> {'Significant' if p1 < 0.05 else 'Not Significant'}')

# Example 2 - Class 1 vs Class 2
class2 = df_clean[df_clean['Pclass']==2]['Survived']
t2, p2 = stats.ttest_ind(class1, class2)
print('Class 1 vs Class 2')
print(f'Survival rate: {class1.mean()*100:.1f}% vs {class2.mean()*100:.1f}%')
print(f'p-value = {p2:.6f} -> {'Significant' if p2 < 0.05 else 'Not Significant'}')

# Example 3 - Random Comparison (Should not be significant)
np.random.seed(42)
random_group_a = np.random.randint(0, 2, 100)
random_group_b = np.random.randint(0, 2, 100)
t3,p3 = stats.ttest_ind(random_group_a, random_group_b)
print('Random group A vs B')
print(f'p-value: {p3:.6f} -> {'Significant' if p3 < 0.05 else 'Not Significant(Expected!)'}')


# Confidence interval
class1_mean = class1.mean()
class1_se = stats.sem(class1)                   # standard error
ci = stats.t.interval(
    confidence=0.95,
    df=len(class1)-1,
    loc=class1_mean,
    scale=class1_se
)
print('----- Confidence Interval-----')
print(f'Class 1 Survival Rate: {class1_mean*100:.1f}%')
print(f'95% CI : ({ci[0]*100:.1f}%, {ci[1]*100:.1f}%)')
print(f'It means we are 95% sure that true survival rate is between {ci[0]*100:.1f}% & {ci[1]*100:.1f}')

# class 3 Confidence Interval
class3_mean = class3.mean()
class3_se = stats.sem(class3)
ci3 = stats.t.interval(
    0.95, df=len(class3)-1, loc=class3_mean, scale=class3_se)
print(f'\nClass 3 Survival rate : {class3_mean*100:.1f}%')
print(f'95% CI : ({ci3[0]*100:.1f}%, {ci3[1]*100:.1f}%)')