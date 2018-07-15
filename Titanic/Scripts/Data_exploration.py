#Import Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Read the Data
train=pd.read_csv('./Titanic/Data/train.csv')
test=pd.read_csv('./Titanic/Data/test.csv')

#Dataset Basic Details
train.info()
train.shape
train.columns

catg_cols=['Pclass','Sex','Cabin','Embarked']
target_col='Survived'
cont_cols=list(set(train.columns)-set(catg_cols)-set(target_col))
cont_cols=['Parch', 'Age', 'Fare', 'SibSp']

#Univariate Analysis for categorical variables
def uni_var_cat(df,catg_cols):
    uni_var_cat_dict={}
    for col in catg_cols:
        uni_var_cat_dict[col]=df[col].value_counts(normalize=True,dropna=False).map('{:.2%}'.format)
        uni_var_df=pd.DataFrame(uni_var_cat_dict[col]).reset_index()
        uni_var_df.rename(columns={'index':col,col:'freq_pct'},inplace=True)
        #Plots
        df[col]=df[col].fillna('missing')
        sns.set(style="ticks")
        plt.style.use("dark_background")
        if df[col].nunique()<=10:
            ax=sns.countplot(x=col,data=df,palette="pastel")
            total=float(len(df))
        elif df[col].nunique()>10:
            uni_10_df=df[df[col].isin(df[col].value_counts().nlargest(10).index)]
            plt.figure(figsize=(15,8))
            ax=sns.countplot(x=col,data=uni_10_df,palette="pastel")
            total=float(len(uni_10_df))
        for p in ax.patches:
            height = p.get_height()
            annot_text=str('{:.1%}'.format(height/total))+', '+str(int(height))
            ax.text(p.get_x()+p.get_width()/2.,height + 3,annot_text,ha="center")
        plt.savefig("C:/Users/charanv/github/Data-Science/Titanic/Output/data exploration/univariate/"+str(col)+".png")
        plt.gcf().clear()
    return uni_var_cat_dict

univar_cat_dat=uni_var_cat(train,catg_cols)

#Univariate Analysis for continuous variables
def uni_var_cont(df,cont_cols):
    uni_var_cont_df=pd.DataFrame(train[cont_cols].describe())
    sns.set(style="darkgrid")
    sns.boxplot(data=df[cont_cols])
    plt.savefig("C:/Users/charanv/github/Data-Science/Titanic/Output/data exploration/univariate/cont_boxplot.png")
    sns.set(style="ticks")
    plt.style.use("dark_background")
    for col in cont_cols:
        sns.kdeplot(data=df[cont_cols])
        plt.savefig("C:/Users/charanv/github/Data-Science/Titanic/Output/data exploration/univariate/cont_dist.png")
    return uni_var_cont_df

univar_cont_dat=uni_var_cont(train,cont_cols)        


# Convert to Categorical dtype
def to_categorical(df,var_list):
    for var in var_list:
        df[var]=df[var].astype('category')
    return df


train=to_categorical(train,catg_cols)

# Bivariate analysis on categorical variables
def bi_var(df,catg,target):
    bi_var_dict={}
    for var in catg:
        value_count=df[var].value_counts()
        value_count_target=df.groupby(var)[target].sum()
        bi_var_dict[var]=value_count_target.div(value_count/100).map('{:.2%}'.format)
    return bi_var_dict

bivar_dict=bi_var(train,catg_cols,target_col)


plt_df_test=pd.DataFrame(univar_dict['Pclass']).reset_index()
plt_df_test.columns
plt_df_test.rename(columns={'index':'Pclass',0:'Survived_pct'},inplace=True)
new_index = (plt_df_test['Survived_pct'].sort_values(ascending=False)).index.values
sorted_data = plt_df_test.reindex(new_index)
sorted_data

plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['Pclass'].index,y=sorted_data['Survived_pct'])
#plt.xticks(rotation= 45)
plt.xlabel('Pclass')
plt.ylabel('% Survived')
plt.title('% Survived by Pclass')
