#!/usr/bin/env python
# coding: utf-8

# # Week3-day2-Data Wrangling day3_15092

# In[13]:


import pandas as pd
import numpy as np


# In[14]:


df=pd.read_csv("pew.csv")


# In[15]:


df.head()


# In[16]:


df.shape


#  There are 18 obs and 11 vars

# In[17]:


df.tail()


# #### Information on dataset

# In[18]:


#df.describe()


# #### Melting Datset-can get all 9 income columns into one var. called "income"

# In[19]:


df_new=pd.melt(df,id_vars="religion",var_name="Income",value_name="count")


# In[20]:


df_new


# In[21]:


df_new.shape


# ## Concrete dataset

# In[23]:


df1=pd.read_csv("concrete.csv")


# In[24]:


df1


# In[28]:


waterdf=pd.DataFrame(df1,columns=["water"])


# In[29]:


waterdf.head()


# In[41]:


x1=df1[["water"]]  ## easy way to convert to dataframe
x1


# ### Extract cement,slag,ash,water,strength, and create concrete_subset

# In[36]:


concrete_subset=df1[['cement','slag','ash','water','strength']]
concrete_subset1=df1.iloc[:,1:4] ## cactually from cols 2 to 4 


# In[37]:


concrete_subset1


# In[48]:


type(concrete_subset)  ## its a dataframe


# In[40]:


third_row=df1[2:3]  ##third row  OR

third_row_iloc=df1.iloc[2:3,:]  ##third row


# In[41]:


third_row


# In[42]:


third_row_iloc


# In[58]:


df1.loc[2]  ## extracting the column details of the 3rd row


# #### Extract rows 20th to 30th

# In[45]:


df1.iloc[19:30,:] ## actually returns from 20-30


# In[49]:


df1.loc[19:30,:] ## with "loc" get exactly upto 30 (from 20-31) (it kinda locks to what we see)


# In[70]:


twoenty_to_thirty=df1[19:30]  ## extracting twoenty_to_thirty
twoenty_to_thirty


# In[80]:


df1.loc[df1['age']==28]


# ### extract 1,2,3rd rows and 1,3,5,7,9th columns

# In[ ]:


##can't use this using just" loc" .Have to use "iloc"


# In[93]:


rc_ss=df1.iloc[0:3,[0,2,4,6,8]]


# In[94]:


rc_ss


# # Feature scaling

# There are two types
# 
# 1.Z-score Transformation  
#   2.Min-max Transformation

# In[105]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
std_concrete=scaler.fit_transform(rc_ss)


# In[104]:


std_concrete


# ## Min-Max 

# In[106]:


from sklearn.preprocessing import  MinMaxScaler
scaler=MinMaxScaler()
min_max_concrete= scaler.fit_transform(rc_ss) ## when we need vals between 0-1 we use min-max scaler


# In[107]:


min_max_concrete

