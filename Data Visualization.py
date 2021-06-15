#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('./train.csv')


# In[3]:


data['Label'].value_counts()


# It depicts the movements along with the count of each movement in train.csv file. This table 
# shows that movements have been almost equally distributed except for the lay movement. Below Bar Chart shows graphical representation. 
# 

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[6]:


import seaborn as sns
sns.set(style="darkgrid")
ax = sns.countplot(x="Label", data=data)
plt.xticks(rotation = 90)


# In[7]:


label = pd.read_csv("./Labels.csv")


# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
correlation = pd.read_csv("./Correlation.csv")
f = plt.figure(figsize=(19, 15))
plt.matshow(correlation.corr(), fignum=f.number)
plt.xticks(range(correlation.shape[1]), correlation.columns, fontsize=14, rotation=45)
plt.yticks(range(correlation.shape[1]), correlation.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# A correlation matrix is a table showing correlation coefficients between variables. 
# Each cell in the table indicates the correlation between two sensor variables. 
# A correlation matrix is used to summarize data, as an input into a extra superior analysis

# In[9]:


Correlation = pd.read_csv("./Correlation.csv")


# In[10]:


import matplotlib.pyplot as plt
Correlation.hist(bins=50, figsize=(20,15))
plt.show()


# A histogram is the most regularly used diagram to show frequency distributions. 
# In this sensor data set, histogram will graphically summarizes on the sensor information and 
# how frequently each sensor is used.

# In[ ]:




