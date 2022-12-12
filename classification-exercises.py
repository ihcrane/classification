#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from env import get_connection
import os
from pydataset import data



def get_titanic(get_con_func):
    
    if os.path.isfile('titanic.csv'):
        
        return pd.read_csv('titanic.csv')
    
    else:
        url = get_con_func('titanic_db')
        query = '''SELECT * FROM passengers'''
        df = pd.read_sql(query, url)
        df.to_csv('titanic.csv')
        return df


# In[5]:


pd.read_excel('titanic.xlsx')


# In[7]:


pd.read_clipboard()


# ## Exercises

# In[12]:


db_iris = data('iris')
db_iris = pd.DataFrame(db_iris)
db_iris.head(3)


# In[14]:


db_iris.shape


# In[15]:


db_iris.columns


# In[17]:


db_iris.info()


# In[18]:


db_iris.describe()


# In[24]:


sheet_url = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'
csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
df_google = pd.read_csv(csv_export_url)
df_google.head(3)


# In[25]:


df_google.shape


# In[26]:


df_google.columns


# In[27]:


df_google.info()


# In[28]:


df_google.describe()


# In[32]:


df_google['Sex'].unique()


# In[35]:


df_google['Embarked'].unique()


# In[6]:


df_excel = pd.read_excel('titanic.xlsx')
df_excel.head(3)


# In[7]:


df_excel_sample = df_excel.head(100)
df_excel.shape


# In[45]:


df_excel_sample.columns[:5]


# In[60]:


df_excel_sample.info()


# In[53]:


df_excel_sample.select_dtypes(include=object).columns


# In[8]:


df_excel_min_max = pd.DataFrame(columns=['max','min'])
df_excel_min_max['max'] = df_excel.select_dtypes(exclude=object).max()
df_excel_min_max['min'] = df_excel.select_dtypes(exclude=object).min()
df_excel_min_max


# In[373]:


from acquire import get_titanic_data

get_titanic_data(get_connection).head()


# In[374]:


from acquire import get_iris_data

get_iris_data(get_connection).head()


# In[375]:


from acquire import get_telco_data

get_telco_data(get_connection).head()


# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from acquire import get_titanic_data


# Overfitting:
# 
# Train/Validate/Test
# 
# Model Training
# 
# When a model is overfit, it makes impeccable guesses on the training dataset.
# 
# An overfit model makes poor (relatively) predictions on out-of-sample data.
# 
# Out-of-sample data can be the validation and test sets. 

# In[4]:


df = get_titanic_data(get_connection)
df.head()


# In[9]:


df.head()


# In[14]:


numerical_cols = df.select_dtypes(exclude='object').columns.to_list()

for col in numerical_cols:
    plt.hist(df[col])
    plt.title(col)
    plt.show()


# In[19]:


categorical_cols = df.select_dtypes(include='object').columns.to_list()

for col in categorical_cols:
    print(df[col].value_counts())
    print('-----------------')
    


# There is duplicate information
# 
# Get rid of class and embarked (duplicates).
# Drop deck column for excess null values.
# Drop passenger_id because it is not helpful.
# Drop age column because it may be difficult to impute.
# 

# In[24]:


df1 = df.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age'])


# In[25]:


df1


# In[26]:


df1['embark_town'].value_counts()


# In[28]:


df1['embark_town'].fillna('Southampton', inplace=True)


# In[34]:


dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True])


# In[35]:


df = pd.concat([df, dummies], axis=1)


# In[36]:


df


# In[44]:


def clean_titanic(df):
    df.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    df['embark_town'].fillna('Southampton', inplace=True)
    
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True])
    
    df = pd.concat([df, dummies], axis=1)
    
    return df


# In[45]:


test_df = get_titanic_data(get_connection)
test_df.head()


# In[46]:


clean_df = clean_titanic(test_df)
clean_df.head()


# In[47]:


seed = 42

train, test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])


# In[48]:


train.shape, test.shape


# In[49]:


train.head()


# In[52]:


seed = 42

train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])


# In[53]:


validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test['survived'])


# In[54]:


train.shape, validate.shape, test.shape


# In[55]:


def train_val_test(df):
    seed = 42 
    train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])
    
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test['survived'])
    
    return train, validate, test


# In[72]:


imputer = SimpleImputer(strategy='most_frequent')


# In[ ]:


impute_df = get_titanic_data(get_connection)
impute_df


# In[70]:


train, validate, test = train_val_test(impute_df)


# In[71]:


train.shape


# In[76]:


imputer.fit(train[['embark_town']])


# In[77]:


train['embark_town'].isna().sum()


# In[82]:


train['embark_town'] = imputer.transform(train[['embark_town']])


# In[83]:


train['embark_town'].isna().sum()


# ## Exercises

# Use the function defined in acquire.py to load the iris data.
# 
# Drop the species_id and measurement_id columns.
# 
# Rename the species_name column to just species.
# 
# Create dummy variables of the species name and concatenate onto the iris dataframe. (This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
# 
# Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

# In[161]:


from acquire import get_iris_data
iris = get_iris_data(get_connection)
iris.head()


# In[155]:


iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)


# In[156]:


iris.rename(columns={'species_name':'species'}, inplace=True)
iris.head()


# In[157]:


iris_dummies = pd.get_dummies(iris[['species']], drop_first=True)
iris = pd.concat([iris, iris_dummies], axis=1)


# In[158]:


iris.head()


# In[162]:


def prep_iris(iris):
    iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)
    
    iris.rename(columns={'species_name':'species'}, inplace=True)
    
    iris_dummies = pd.get_dummies(iris[['species']], drop_first=True)
    iris = pd.concat([iris, iris_dummies], axis=1)
    
    return iris


# Use the function defined in acquire.py to load the Titanic data.
# 
# Drop any unnecessary, unhelpful, or duplicated columns.
# 
# Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
# 
# Create a function named prep_titanic that accepts the raw titanic data, and returns the data with the transformations above applied.

# In[165]:


titanic = get_titanic_data(get_connection)
titanic.head()


# In[164]:


titanic.drop(columns=['Unnamed: 0', 'passenger_id', 'age', 'embarked', 'class', 'deck'], inplace=True)
titanic.head()


# In[113]:


titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
titanic = pd.concat([titanic, titanic_dummies], axis=1)
titanic


# In[383]:


def prep_titanic(titanic):
    titanic.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    
    return titanic


# Use the function defined in acquire.py to load the Telco data.
# 
# Drop any unnecessary, unhelpful, or duplicated columns. This could mean dropping foreign key columns but keeping the corresponding string values, for example.
# 
# Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
# 
# Create a function named prep_telco that accepts the raw telco data, and returns the data with the transformations above applied.

# In[169]:


from acquire import get_telco_data

telco = get_telco_data(get_connection)
telco.head()


# In[143]:


telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 'internet_service_type_id', 'customer_id'], inplace=True)
telco.head()


# In[144]:


telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)

telco_dummies


# In[145]:


telco = pd.concat([telco, telco_dummies], axis=1)
telco.head()


# In[172]:


def prep_telco(telco):
    telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 
                        'internet_service_type_id', 'customer_id'], inplace=True)
    
    telco['total_charges'] = (telco['total_charges'] + '0').astype('float')

    
    telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)
    
    telco = pd.concat([telco, telco_dummies], axis=1)
    
    return telco

prep_telco(telco).head()


# Write a function to split your data into train, test and validate datasets. Add this function to prepare.py.
# 
# Run the function in your notebook on the Iris dataset, returning 3 datasets, train_iris, validate_iris and test_iris.
# 
# Run the function on the Titanic dataset, returning 3 datasets, train_titanic, validate_titanic and test_titanic.
# 
# Run the function on the Telco dataset, returning 3 datasets, train_telco, validate_telco and test_telco.

# In[147]:


def train_val_test(df, col):
    seed = 42 
    train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df[col])
    
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test[col])
    
    return train, validate, test


# In[148]:


train_iris, val_iris, test_iris = train_val_test(iris, 'species')
train_iris.shape, val_iris.shape, test_iris.shape


# In[149]:


train_titanic, val_titanic, test_titanic = train_val_test(titanic, 'survived')
train_titanic.shape, val_titanic.shape, test_titanic.shape


# In[150]:


train_telco, val_telco, test_telco = train_val_test(telco, 'gender')
train_telco.shape, val_telco.shape, test_telco.shape


# In[ ]:





# ## Data Exploration

# In[199]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.model_selection import train_test_split

import acquire
from prepare import train_val_test

import warnings
warnings.filterwarnings('ignore')


# In[212]:


taxis = sns.load_dataset('taxis')
taxis.head()


# In[213]:


taxis.info()


# In[214]:


taxis['payment'].value_counts()


# In[215]:


taxis['color'].value_counts()


# In[216]:


taxis.describe().T


# In[217]:


taxis = taxis.drop(columns=['pickup', 'dropoff', 'pickup_zone', 'dropoff_zone'], axis=1)


# In[218]:


taxis.info()


# In[219]:


taxis.isnull().sum()


# In[229]:


taxis = taxis.dropna(axis=0)


# In[230]:


taxis.isnull().sum()
taxis.shape


# In[233]:


dummy_df = pd.get_dummies(taxi[['color', 'payment', 'pickup_borough', 'dropoff_borough']], drop_first=True)
dummy_df.dropna()


# In[234]:


dummy_df.shape


# In[236]:


taxis = pd.concat([taxis, dummy_df], axis=1)
taxis = taxis.dropna()


# In[240]:


taxis = taxis.rename(columns={'payment_credit card': 'payment_credit_card', 'dropoff_borough_Staten Island':'dropoff_borough_Staten_Island'})
taxis.head()


# In[241]:


num_vars = taxis.select_dtypes(include=['int64', 'float'])
cat_vars = taxis.select_dtypes(include=['object', 'uint8'])


# In[242]:


train, validate, test = train_val_test(taxis, 'color')
train.shape, validate.shape, test.shape


# # Exercises

# In[261]:


iris = prep_iris(get_iris_data(get_connection))
iris.head()


# In[485]:


train, val, test = train_val_test(iris, 'species')
train.shape, val.shape, test.shape


# ## Univariate Stats

# In[486]:


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import env
from scipy import stats
from sklearn.model_selection import train_test_split
import prepare


# In[265]:


iris.info()


# In[466]:


num_vars = iris.select_dtypes(exclude=['object', 'uint8'])
cat_vars = iris.select_dtypes(include=['object', 'uint8'])


# In[275]:


for col in num_vars:
    print(col)
    sns.boxplot(x=col, data=train)
    plt.show()


# In[276]:


for col in num_vars:
    print(col)
    sns.histplot(x=col, data=train)
    plt.show()
    


# - Takeaways: sepal length and width are normally distributed 
# - there are a lot of petals with lengths of 1 and then the large majority is 4 or more
# - there are a lot of petals that have widths of 0.5 or less with none have a width between .5 and 1.

# ## Bivariate Stats

# In[298]:


print(train['species'].value_counts())
print(train['species'].value_counts(normalize=True) * 100)
sns.countplot(x='species', data=train)
plt.show()
#Takeaways: I thought the barplot would produce more information than just 3 identical bars.


# In[487]:


for col in num_vars:
    print(col)
    sns.barplot(x='species', y=col, data=train)
    plt.axhline(num_vars[col].mean(), label="Overall Mean")
    plt.legend()
    plt.show()


# - The setosa species dimensions fluctuates a lot but the other two seem to stay pretty close to one another
# - On average the virginica species is larger.
# 

# ## Multivariate Stats

# In[312]:


sns.pairplot(train, hue='species', corner=True)


# In[314]:


sns.relplot(x='sepal_length', y='sepal_width', data=train, hue='species')


# In[317]:


for col in num_vars:
    sns.swarmplot(x=col, y='species', data=train)
    plt.show()


# In[469]:


train_melt = train.melt(id_vars='species')
train_melt.head()


# In[472]:


sns.swarmplot(x='variable', y='value', data=train_melt, hue='species')


# In[332]:


setosa = train[train['species'] == 'setosa']
setosa.drop(columns=['petal_length', 'petal_width', 'species', 
                     'species_versicolor', 'species_virginica'], inplace=True)


# In[335]:


versicolor = train[train['species'] == 'versicolor']
versicolor.drop(columns=['petal_length', 'petal_width', 'species', 
                         'species_versicolor', 'species_virginica'], inplace=True)


# In[464]:


setosa['area'] = setosa['sepal_length'] * setosa['sepal_width']
setosa.head()


# In[465]:


versicolor['area'] = versicolor['sepal_length'] * versicolor['sepal_width']
versicolor.head()


# H_0: Mean of setosa area <= mean of versicolor area  
# H_A: Mean of setosa area > mean of versicolor area

# In[353]:


t, p = stats.ttest_ind(setosa['area'], versicolor['area'])
t, p


# In[354]:


alpha = .05

if p <= alpha:
    print('Reject null')
else:
    print('Fail to reject null')


# In[357]:


train['sepal_area'] = train['sepal_length'] * train['sepal_width']
train.head()


# In[372]:


sns.histplot(x='sepal_area', data=train, hue='species')


# - Takeaway: We fail to reject null. There is no significant difference in the mean area of setosa and versicolor

# # Titanic

# In[435]:


titanic = get_titanic_data(get_connection)
titanic = prep_titanic(titanic)
titanic.head()


# In[436]:


titanic.info()


# In[437]:


titanic['embark_town'].value_counts()


# In[458]:


from prepare import train_val_test

train, val, test = train_val_test(titanic, 'survived')


# In[459]:


num_vars = titanic.select_dtypes(exclude=['object', 'uint8'])
cat_vars = titanic.select_dtypes(include=['object', 'uint8'])


# In[460]:


for col in num_vars:
    sns.barplot(x='embark_town', y=col, data=train)
    plt.axhline(train[col].mean(), label='Overall Mean')
    plt.legend()
    plt.show()


# In[461]:


for col in num_vars:
    sns.barplot(x='survived', y=col, data=train)
    plt.axhline(train[col].mean(), label='Overall Mean')
    plt.show()


# In[462]:


for col in cat_vars:
    sns.barplot(x='survived', y=col, data=train)
    plt.show()


# In[463]:


for col in num_vars:
    sns.barplot(x='pclass', y=col, data=train)
    plt.axhline(train[col].mean(), label='Overall Mean')
    plt.show()


# - Takeaways: 
# - Females had a much higher chance of survival.
# - The higher the class someone is the higher chance of survival they have.
# - If someone is alone they have a lower chance of survival.
# - The fare for getting on at Cherbourg is significantly higher than the mean.
# - Those getting on at Cherbourg had a higher chance of surviving than the mean.
# 
# 
# - Binning the fares would help to compare it against other columns and data.

# ## Telco

# In[474]:


from acquire import get_telco_data

telco = get_telco_data(get_connection)
telco = prep_telco(telco)
telco.head()


# In[475]:


train, val, test = train_val_test(telco, 'gender')


# In[476]:


num_vars_telco = telco.select_dtypes(exclude=['object', 'uint8'])
cat_vars_telco = telco.select_dtypes(include=['object', 'uint8'])


# In[477]:


for col in cat_vars_telco:
    sns.barplot(x=col, y='tenure', data=train)
    plt.axhline(train['tenure'].mean(), label='Tenure Mean')
    plt.legend()
    plt.show()


# - Takeaways:
# - Those with a partner tend to stay longer than those without.
# - Those with dependents have a higher tenure than those without.
# - Those with add-ons stay longer.
# - Those with automatic payments have a higher tenure.
# - Month-to-month contract types have a higher churn rate.
# 

# In[483]:


sns.barplot(x='churn_Yes', y='senior_citizen', data=train)
plt.title('Senior Citizen Churn Rate')
plt.axhline(train['churn_Yes'].mean(), label='Mean Churn Rate')
plt.legend()


# In[484]:


for col in num_vars_telco:
   sns.barplot(x='churn_Yes', y=col, data=train)
   plt.axhline(train[col].mean(), label='Overall Mean')
   plt.legend()
   plt.show()


# - Takeaways:
# - Those with higher monthly charges are more likely to churn.
# - Senior citizens are more likely to churn.
# - Combining the payment types into groups of automatic and manual would help to compare against other data.
# 

# In[ ]:




