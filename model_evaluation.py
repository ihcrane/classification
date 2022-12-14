#!/usr/bin/env python
# coding: utf-8

# 1. bottom left
# 2. top right
# 3. the model is accurate 80% of the time

# In[71]:


tp = 46
tn = 34
fp = 13
fn = 7


# In[72]:


accuracy = (tp + tn) / (tp+tn+fn+fp)
precision = (tp) / (tp+fp)
recall = tp / (tp+fn)


# In[73]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[74]:


df = pd.read_csv('c3.csv')
df.head(20)


# In[75]:


model1_acc = (df.model1 == df.actual).mean()
model2_acc = (df.model2 == df.actual).mean()
model3_acc = (df.model3 == df.actual).mean()

print(model1_acc)
print(model2_acc)
print(model3_acc)


# In[76]:


subset = df[df['actual']=='Defect']

model1_recall = (subset.model1 == subset.actual).mean()
model2_recall = (subset.model2 == subset.actual).mean()
model3_recall = (subset.model3 == subset.actual).mean()

print(model1_recall)
print(model2_recall)
print(model3_recall)

# the best way to determine number of defects and have the recalled is use the Recall metric
# model 3 has the highest recall


# In[91]:


subset1 = df[df['model1']=='Defect']
subset2 = df[df['model2']=='Defect']
subset3 = df[df['model3']=='Defect']

model1_prec = (subset1.model1 == subset1.actual).mean()
model2_prec = (subset2.model2 == subset2.actual).mean()
model3_prec = (subset3.model3 == subset3.actual).mean()

print(model1_prec)
print(model2_prec)
print(model3_prec)

# the best way to determine which ducks were defects and 
# which ones to send on the vacation would be the Precision metric
# model 1 has the highest precision


# In[78]:


df1 = pd.read_csv('gives_you_paws.csv')
df1.head()


# In[79]:


df1['actual'].value_counts()


# In[80]:


df1['base'] = 'dog'
df1.head()


# In[81]:


model1_acc = (df1.model1 == df1.actual).mean()
model2_acc = (df1.model2 == df1.actual).mean()
model3_acc = (df1.model3 == df1.actual).mean()
model4_acc = (df1.model4 == df1.actual).mean()
baseline = (df1.base == df1.actual).mean()

print(model1_acc)
print(model2_acc)
print(model3_acc)
print(model4_acc)
print(baseline)

# Models 1 and 4 have the highest accuracy and both are above the baseline


# In[92]:


subset = df1[df1['actual']=='dog']

model1_recall = (subset.model1 == subset.actual).mean()
model2_recall = (subset.model2 == subset.actual).mean()
model3_recall = (subset.model3 == subset.actual).mean()
model4_recall = (subset.model4 == subset.actual).mean()

print(model1_recall)
print(model2_recall)
print(model3_recall)
print(model4_recall)

# the best model when dealing with only dog pics would be model 4 which has a Recall of 95.57


# In[93]:


subset = df1[df1['actual']=='cat']

model1_recall = (subset.model1 == subset.actual).mean()
model2_recall = (subset.model2 == subset.actual).mean()
model3_recall = (subset.model3 == subset.actual).mean()
model4_recall = (subset.model4 == subset.actual).mean()

print(model1_recall)
print(model2_recall)
print(model3_recall)
print(model4_recall)

# the best model when dealing with only cat pics would be model 2 which has a Recall of 89.06


# In[84]:


from sklearn.metrics import accuracy_score

print('Model 1 accuracy: ', accuracy_score(df1['actual'], df1['model1']))
print('Model 2 accuracy: ', accuracy_score(df1['actual'], df1['model2']))
print('Model 3 accuracy: ', accuracy_score(df1['actual'], df1['model3']))
print('Model 4 accuracy: ', accuracy_score(df1['actual'], df1['model4']))


# In[85]:


from sklearn.metrics import precision_score

print('Model 1 precision: ', precision_score(df1['actual'], df1['model1'], pos_label='dog'))
print('Model 2 precision: ', precision_score(df1['actual'], df1['model2'], pos_label='dog'))
print('Model 3 precision: ', precision_score(df1['actual'], df1['model3'], pos_label='dog'))
print('Model 4 precision: ', precision_score(df1['actual'], df1['model4'], pos_label='dog'))


# In[86]:


from sklearn.metrics import recall_score

print('Model 1 recall: ', recall_score(df1['actual'], df1['model1'], pos_label='dog'))
print('Model 2 recall: ', recall_score(df1['actual'], df1['model2'], pos_label='dog'))
print('Model 3 recall: ', recall_score(df1['actual'], df1['model3'], pos_label='dog'))
print('Model 4 recall: ', recall_score(df1['actual'], df1['model4'], pos_label='dog'))


# In[87]:


from sklearn.metrics import classification_report

print(classification_report(df1['actual'], df1['model1']))


# In[88]:


print(classification_report(df1['actual'], df1['model2']))


# In[89]:


print(classification_report(df1['actual'], df1['model3']))


# In[90]:


print(classification_report(df1['actual'], df1['model4']))

