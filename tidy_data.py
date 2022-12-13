#!/usr/bin/env python
# coding: utf-8

# # Tidy Data

# In[22]:


import numpy as np
import pandas as pd
from scipy import stats
from env import get_connection
import seaborn as sns
import matplotlib.pyplot as plt

url = get_connection('tidy_data')
att = pd.read_sql('SELECT * FROM attendance', url)
att.head()


# In[23]:


att = att.rename(columns={'Unnamed: 0':'name'})
att.head()


# In[24]:


att = att.melt(id_vars='name', var_name='attendence')


# In[26]:


att.head(10)


# In[27]:


att['value'] = att['value'].map({'P':1, 'H':.5, 'T':.9, 'A':0})
att.head(10)


# In[29]:


att.groupby('name').mean()


# ## Exercise 2

# In[35]:


coffee = pd.read_sql('SELECT * FROM coffee_levels', url)
coffee.head(10)


# In[36]:


coffee['coffee_carafe'].unique()


# In[45]:


coffee_new = pd.pivot(data=coffee, index='hour', columns='coffee_carafe')
coffee_new


# This is not the best format for the data. It is hard to read and understand.

# ## Exercise 3

# In[63]:


cake = pd.read_sql('SELECT * FROM cake_recipes', url)
cake.head()


# In[64]:


recipe = cake['recipe:position'].str.split(':', expand=True)


# In[65]:


cake = pd.merge(left=cake, right=recipe, left_index=True, right_index=True)


# In[69]:


cake.drop(columns=['recipe:position'], inplace=True)


# In[70]:


cake


# In[73]:


cake.rename(columns={0:'recipe', 1:'position'}, inplace=True)


# In[85]:


cake_new = cake.melt(id_vars=['recipe','position'], var_name='temp')
cake_new.head()


# In[88]:


cake_recipe = cake_new.drop(columns=['temp', 'position'])
cake_recipe.head()


# In[87]:


cake_recipe.groupby('recipe').mean()


# In[89]:


cake_temp = cake_new.drop(columns=['recipe', 'position'])
cake_temp.head()


# In[90]:


cake_temp.groupby('temp').mean()


# In[97]:


cake_new.sort_values(by='value', ascending=False)


# In[ ]:




