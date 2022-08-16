#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings(action='ignore')


# In[ ]:


demo = pd.read_csv('Data/LPOINT_BIG_COMP_01_DEMO.csv')
demo.head()


# In[ ]:


pdde = pd.read_csv('Data/LPOINT_BIG_COMP_02_PDDE.csv')
pdde.head()


# In[ ]:


# 상품 분류정보 - 상품 구매 정보와 조인 
clac = pd.read_csv('Data/LPOINT_BIG_COMP_04_PD_CLAC.csv')
clac.head()


# ## Pdde 테이블과 Clac 테이블 join

# In[ ]:


pdde_clac = pdde.merge(clac, on = '상품_구분', how = 'inner')
pdde_clac.head()


# ## Demo 테이블과 Pdde_Clac 테이블 join

# In[ ]:


make_buy = a.merge(demo, on = '고객번호')
make_buy.head()


# ## Make_buy 데이터와 RFM 등급 산출 데이터 병합 

# In[ ]:


rfm = pd.read_csv('구매테이블_RFM_등급_산출')


# In[ ]:


r =rfm[['고객번호','최근방문일_R','방문빈도_F','구매금액_M','등급']]
df = make_buy.merge(r, on = '고객번호')
df.head()


# ## Lpay 데이터와 Demo 데이터 병합 

# In[ ]:


lpay_buy = demo.merge(lpay, on ='고객번호' )
lpay_buy.head()


# ## Lpay 데이터와 Lpay RFM 등급 산출 데이터 병합 

# In[ ]:


l_pay_rfm = pd.read_csv('Data/lpay_RFM_df.csv')
l_pay_rfm = l_pay_rfm[['고객번호','등급']]

df = lpay_buy.merge(lpay_rfm, on = '고객번호')
df.head()

