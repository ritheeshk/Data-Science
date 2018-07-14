#Import Packages
import pandas as pd

#Read the Data
train=pd.read_csv('./Titanic/Data/train.csv')
test=pd.read_csv('./Titanic/Data/test.csv')

#Dataset Basic Details
train.info()
train.shape
train.columns

# Convert to Categorical dtype
def to_categorical(df,var_list):
    for var in var_list:
        df[var]=df[var].astype('category')
    return df

catg_cols=['Pclass','Sex','Cabin','Embarked']
target_col='Survived'
train=to_categorical(train,catg_cols)

# Univariate analysis on categorical variables
def uni_var(df,catg,target):
    uni_var_dict={}
    for var in catg:
        value_count=df[var].value_counts()
        value_count_target=df.groupby(var)[target].sum()
        uni_var_dict[var]=value_count_target.div(value_count/100).map('{:,.2f}%'.format)
    return uni_var_dict

univar_dict=uni_var(train,catg_cols,target_col)

univar_dict['Pclass']
