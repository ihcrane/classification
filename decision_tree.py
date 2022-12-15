#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from prepare import train_val_test
import acquire


# In[20]:


titanic = pd.read_csv('titanic.csv')        # importing data
titanic.head()


# In[21]:


def prep_titanic(titanic):
    '''
    cleaning titantic data and creating dummies
    '''
    titanic.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    
    return titanic


# In[22]:


titanic = prep_titanic(titanic)      # calling cleaning function
titanic.head()


# In[23]:


titanic.info()


# In[24]:


titanic['survived'].value_counts()           # determining which survival outcome is most common


# In[25]:


titanic['base'] = 0             # creating baseline column with most common survival outcome
titanic.info()


# In[26]:


(titanic['base'] == titanic['survived']).mean()       # determine accuracy of baseline


# In[27]:


titanic = titanic.drop(columns=['embark_town', 'sex'])      # dropping string columns which already have 
                                                            # dummies created for them


# In[28]:


titanic.head()
titanic.info()


# In[29]:


train, val, test = train_val_test(titanic, 'survived')       # splitting data


# In[30]:


train.shape, val.shape, test.shape                   # checking to make sure split correctly


# In[31]:


X_train = train.drop(columns='survived')            # creating split for decision tree classifier
y_train = train['survived']

X_val = val.drop(columns='survived')
y_val = val['survived']

X_test = test.drop(columns='survived')
y_test = test['survived']


# In[32]:


seed = 42

clf = DecisionTreeClassifier(max_depth=3, random_state=42)        # creating decision tree classifier


# In[33]:


clf.fit(X_train, y_train)                    # fitting clf to data


# In[34]:


clf.score(X_train, y_train)                # determine accuracy of clf on the train data


# In[35]:


val_preds = clf.predict(X_val)              # creating prediction from the trained clf on the validate data


# In[36]:


plot_confusion_matrix(clf, X_train, y_train)


# In[421]:


print(classification_report(y_val, val_preds))          # viewing outcome of the decision tree classifier 
                                                        # on the validate data


# - A depth of 18 appears to give the best results for the train sample with an accuracy 93%.
# - A depth of 3 appears to give the best results for the validate sample with an accuracy of 82-83%.

# In[416]:


train_acc = []                      # creating variables
val_acc = []
depth = []
test_acc = []

for i in range(3, 25):
    seed = 42               
    clf = DecisionTreeClassifier(max_depth=i, random_state=42)     # creating decision tree classifier
    
    clf.fit(X_train, y_train)
    
    t_acc = (clf.score(X_train, y_train) * 100)        # scoring accuracy for train and saving to a list
    train_acc.append(t_acc)
    
    v_acc = (clf.score(X_val, y_val) * 100)        # scoring accuracy for validate and saving to a list
    val_acc.append(v_acc)
    depth.append(i)
    
    tt_acc = (clf.score(X_test, y_test) * 100)        # scoring accuracy for test and saving to a list
    test_acc.append(tt_acc)
    




plt.plot(depth, train_acc, color='rebeccapurple', label='Train Accuracy')
plt.plot(depth, val_acc, label='Validate Accuracy')                     # creating plot for train, val and test
plt.plot(depth, test_acc, label='Test Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')                       # labeling axies
plt.legend()
plt.show()
    


# In[39]:


y_preds = clf.predict(X_train)


# In[41]:


TN, FP, FN, TP = confusion_matrix(y_train, y_preds).ravel()


# In[42]:


TN, FP, FN, TP


# In[43]:


negative_cases = TN + FP
positive_cases = FN + TP


# In[44]:


print(f'Negative Cases: {negative_cases}')
print(f'Positive Cases: {positive_cases}')
print(y_train.value_counts())


# In[47]:


for i in range(1, 21):
    trees = DecisionTreeClassifier(max_depth=i, random_state=123)
    trees = trees.fit(X_train, y_train)
    y_preds = trees.predict(X_train)
    report = classification_report(y_train, y_preds, output_dict=True)
    print(f'Tree with max depth of {i}')
    print(pd.DataFrame(report))
    print()


# In[48]:


metrics = []

for i in range(1, 25):
    tree = DecisionTreeClassifier(max_depth=i, random_state=123)
    tree = tree.fit(X_train, y_train)
    in_sample_accuracy = tree.score(X_train, y_train)
    out_of_sample_accuracy = tree.score(X_val, y_val)
    
    output= {'max_depth':i, 'train_accuracy':in_sample_accuracy, 'validate_accuracy':out_of_sample_accuracy}
    
    metrics.append(output)
    
df = pd.DataFrame(metrics)
df['difference'] = df['train_accuracy'] - df['validate_accuracy']
df


# In[51]:


plt.figure(figsize=(12,6))
plt.plot(df.max_depth, df.train_accuracy, marker = 'o', label='Train')
plt.plot(df.max_depth, df.validate_accuracy, marker = 'o', label='Validate')
plt.legend()
plt.show()


# In[223]:


confusion_matrix(y_val, val_preds)          # creating confusion matrix from val data


# In[224]:


class_names = np.array(clf.classes_).astype('str').tolist()


# In[279]:


plt.figure(figsize=(20, 14))
                                               # creating decision tree figure
plot_tree(clf, feature_names=X_train.columns, 
          class_names = np.array(clf.classes_).astype('str').tolist(), rounded=True)

plt.show()


# ## Telco Database

# In[53]:


telco = pd.read_csv('telco.csv')


# In[54]:


def prep_telco(telco):
    '''
    cleaning telco data and creating dummies for data
    also converting total charges column to floats to manipulate data more effectively
    '''
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


# In[55]:


telco = prep_telco(telco)
telco.info()


# In[56]:


telco.drop(columns=['partner', 'dependents', 'phone_service', 'multiple_lines', 
                    'online_security', 'online_backup', 'device_protection',
                    'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing',
                    'churn', 'internet_service_type', 'contract_type','payment_type', 'gender'], inplace=True)

                    # dropping object columns now that dummies have been created


# In[57]:


telco.info()


# In[58]:


train_tel, val_tel, test_tel = train_val_test(telco, 'churn_Yes')   # splitting data 


# In[59]:


train_tel.shape, val_tel.shape, test_tel.shape         # checking split sizes


# In[60]:


X_train_tel = train_tel.drop(columns='churn_Yes')      # splitting data for decision tree
y_train_tel = train_tel['churn_Yes']

X_val_tel = val_tel.drop(columns='churn_Yes')
y_val_tel = val_tel['churn_Yes']

X_test_tel = test_tel.drop(columns='churn_Yes')
y_test_tel = test_tel['churn_Yes']


# In[61]:


seed = 42

clf = DecisionTreeClassifier(max_depth=3, random_state=42)      # creating decision tree classifier


# In[62]:


clf.fit(X_train_tel, y_train_tel)                 # fitting clf to data


# In[63]:


clf.score(X_train_tel, y_train_tel)            # scoring accuracy of clf on train data


# In[64]:


val_preds_tel = clf.predict(X_val_tel)          # predicting validate outcomes


# In[65]:


print(classification_report(y_val_tel, val_preds_tel))        # printing metrics of validate data and clf


# In[66]:


plt.figure(figsize=(20, 14))
                                                  # creating decision tree figure
plot_tree(clf, feature_names=X_train_tel.columns, 
          class_names = np.array(clf.classes_).astype('str').tolist(), rounded=True)

plt.show()


# In[71]:


train_tel_acc = []                      # creating variables
val_tel_acc = []
depth = []
test_tel_acc = []

for i in range(3, 25):
    seed = 42               
    clf = DecisionTreeClassifier(max_depth=i, random_state=42)     # creating decision tree classifier
    
    clf.fit(X_train_tel, y_train_tel)
    
    t_tel_acc = (clf.score(X_train_tel, y_train_tel) * 100)        # scoring accuracy for train and saving to a list
    train_tel_acc.append(t_tel_acc)
    
    v_tel_acc = (clf.score(X_val_tel, y_val_tel) * 100)        # scoring accuracy for validate and saving to a list
    val_tel_acc.append(v_tel_acc)
    depth.append(i)
    
    



plt.figure(figsize=(12,6))
plt.plot(depth, train_tel_acc, marker='o', label='Train Accuracy')
plt.plot(depth, val_tel_acc, marker='o', label='Validate Accuracy')    # creating plot for train and val
plt.xlabel('Depth')
plt.ylabel('Accuracy')                       # labeling axies
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




