#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
from statsmodels.compat import lzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


bed = pd.read_csv(r"C:\Users\ASUS\Desktop\price optimization\retail_price.csv")


# In[3]:


bed.head(10)


# In[4]:


bed.isnull().sum()


# In[5]:


bed_model = ols(" total_price ~ freight_price ", data=bed).fit()


# In[6]:


print(bed_model.summary())


# In[7]:


bed_model = ols(" freight_price ~ unit_price ", data=bed).fit()


# In[8]:


print(bed_model.summary())


# In[9]:


bed_model = ols(" total_price ~ unit_price ", data=bed).fit()


# In[10]:


print(bed_model.summary())


# In[11]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(bed_model, fig=fig)


# In[12]:


fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_ccpr_grid(bed_model, fig=fig)


# In[13]:


fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(bed_model, 'unit_price', fig=fig)


# In[14]:


bed['month_year'] = pd.to_datetime(bed['month_year'], format="mixed")


# In[15]:


bed.head()


# In[16]:


from pandas.tseries.offsets import *
bed['Date'] = bed.apply(lambda x:(x['month_year'] + BQuarterBegin(x['qty'])), axis=1)


# In[17]:


bed.drop(['month_year', 'ps1','ps3','product_id',], axis=1, inplace=True)


# In[18]:


bed.head(10)


# In[19]:


endog = bed['qty']


# In[20]:


exog = sm.add_constant(bed['total_price'])


# In[21]:


mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())


# In[22]:


res.plot_recursive_coefficient(range(mod.k_exog), alpha=None, figsize=(10,6));


# In[23]:


fig = res.plot_cusum(figsize=(10,6));


# In[24]:


endog = bed['freight_price']


# In[25]:


exog = sm.add_constant(bed['unit_price'])


# In[26]:


mod = sm.RecursiveLS(endog, exog)
res = mod.fit()

print(res.summary())


# In[27]:


res.plot_recursive_coefficient(range(mod.k_exog), alpha=None, figsize=(10,6));


# In[28]:


fig = res.plot_cusum(figsize=(10,6));


# In[ ]:




