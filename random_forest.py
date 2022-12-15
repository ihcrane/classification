#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from acquire import get_titanic_data
from prepare import prep_titanic, train_val_test


# In[66]:


titanic = prep_titanic(get_titanic_data())
titanic.head()


# In[67]:


titanic.drop(columns=['sex', 'embark_town'], inplace=True)


# In[68]:


train, val, test = train_val_test(titanic, 'survived')
train.shape, val.shape, test.shape


# In[69]:


X_train = train.drop(columns='survived')
y_train = train['survived']

X_val = val.drop(columns='survived')
y_val = val['survived']

X_test = test.drop(columns='survived')
y_test = test['survived']


# - Fit the Random Forest classifier to your training sample and transform (i.e. make predictions on the training sample) setting the random_state accordingly and setting min_samples_leaf = 1 and max_depth = 10.
# 
# - Evaluate your results using the model score, confusion matrix, and classification report.
# 
# - Print and clearly label the following: Accuracy, true positive rate, false positive rate, true negative rate, false negative rate, precision, recall, f1-score, and support.
# 
# - Run through steps increasing your min_samples_leaf and decreasing your max_depth.
# 
# - What are the differences in the evaluation metrics? Which performs better on your in-sample data? Why?
# 
# - After making a few models, which one has the best performance (or closest metrics) on both train and validate?

# In[70]:


seed = 42

rf = RandomForestClassifier(max_depth=10, random_state=42, min_samples_leaf=1)


# In[71]:


rf.fit(X_train, y_train)


# In[72]:


rf.score(X_train, y_train)


# In[73]:


train_preds = rf.predict(X_train)


# In[74]:


TN, FP, FN, TP = confusion_matrix(y_train, train_preds).ravel()


# In[75]:


TN, FP, FN, TP


# In[76]:


print(classification_report(y_train, train_preds))


# In[130]:


train_acc = []
val_acc = []
depth = []
leaf = []

for i in range(1, 11):
    
    for n in range(1,6):
        rf = RandomForestClassifier(max_depth=i, random_state=seed, min_samples_leaf=n)
    
        rf.fit(X_train, y_train)
        t_acc = (rf.score(X_train, y_train) * 100)
        train_acc.append(t_acc)
    
        v_acc = (rf.score(X_val, y_val) * 100)
        val_acc.append(v_acc)
    
        depth.append(i)
        leaf.append(n)
    

acc = pd.DataFrame({'depth':depth, 'leaf':leaf, 'train_acc':train_acc, 'val_acc':val_acc})
acc.head(10)


# In[131]:


acc['difference'] = acc['train_acc'] - acc['val_acc']


# In[132]:


acc.sort_values('train_acc', ascending=False).head(1)


# In[133]:


acc.sort_values('val_acc', ascending=False).head()


# In[134]:


acc[(acc['difference'] > -.5) & (acc['difference']<1)].sort_values('difference')


# In[135]:


for i in range(1, 6):
    points = acc[acc['leaf']==i]
    
    plt.plot(points.depth, points.train_acc, marker='o', label='Train Accuracy')
    plt.plot(points.depth, points.val_acc, marker='o', label='Validate Accuracy')
    plt.title(f'Min Sample Leaf of {i}')
    plt.legend()
    plt.show()
        


# **Takeaways**
# 
# - For in sample data the best model is Max Depth = 10 and Min Sample Leaf = 1
# - For out of sample data the best model is Max Depth = 4 and Min Sample Leaf = 4
# - The closest metrics are at Max Depth = 5 and Min Sample leaf of 5

# In[ ]:




