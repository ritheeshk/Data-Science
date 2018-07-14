#Import Packages
import pandas as pd

#Read the Data
train=pd.read_csv('./Titanic/Data/train.csv')
test=pd.read_csv('./Titanic/Data/test.csv')

#Dataset Basic Details
train.info()
train.shape
train.columns

# Univariate analysis
value_count=train.Sex.value_counts()
value_count_target=train.groupby('Sex').Survived.sum()
value_count_target.div(value_count)
