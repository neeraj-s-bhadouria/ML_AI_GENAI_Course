import pandas as pd

# load csv
df = pd.read_csv('data/train.csv')

# analyse the file
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'First five rows:\n {df.head()}\n')

# Detailed info about data
print('Dataset Info---------------')
print(df.info())
print('\nStatistical summary----------')
print(df.describe())

# Counting missing values
print('\nMissing values')
print(df.isnull().sum())
print('Missing values percentage')
print(round(df.isnull().sum()/len(df) * 100, 2))

# selecting 1 column only
print('\nSurvived column')
print(df['Survived'].head())

# selecting multiple columns
print('\nName & Age')
print(df[['Name', 'Age']].head())

# Filtering - only those who survived
survived = df[df['Survived'] == 1]
print('\nThose who survived - ', len(survived))

# Filtering - Those women who survived and belongs to class 1
class1_survived_women = df[(df['Pclass'] == 1) & (df['Sex'] == 'female') & (df['Survived'] == 1)]
print('Class 1 women who survived - ', len(class1_survived_women))

# Survival rate by gender
print('\nSurvival rate by gender')
print(df.groupby('Sex')['Survived'].mean())

# Survival rate by class
print('\nSurvival rate by class')
print(df.groupby('Pclass')['Survived'].mean())

# Survival rate by class and gender combined
print('\nSurvival rate by class and gender combined')
print(df.groupby(['Sex', 'Pclass'])['Survived'].mean())


# data cleaning - Fixing missing values
# create a copy of data, keep the original safe
df_clean = df.copy()

# Drop cabin - 77% missing values
df_clean = df_clean.drop(columns=['Cabin'])

# Age - Fill with median cause this column contains outlier
median_age = df_clean['Age'].median()
print('\nMedian age = ', median_age)
df_clean['Age'] = df_clean['Age'].fillna(median_age)

# Embarked - fill with mode
embarked_mode = df_clean['Embarked'].mode()[0]
print('Embarked mode = ', embarked_mode)
df_clean['Embarked'] = df_clean['Embarked'].fillna(embarked_mode)

# verify that no null value is left
print('Missing values after cleaning - ', df_clean.isnull().sum())


# Average age of survivors vs non-survivors
print('\nAverage age - Survived vs not')
print(df_clean.groupby('Survived')['Age'].mean())

# No of people in every class
print('\nPassengers per class')
print(df_clean['Pclass'].value_counts())

# Fare average by class
print('\nAverage fare by class')
print(df_clean.groupby('Pclass')['Fare'].mean())