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

import matplotlib
matplotlib.rc('font', family='AppleGothic')
matplotlib.rc('axes', unicode_minus=False)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# 그래프 틀 변경
plt.rcParams['axes.unicode_minus'] = False
sns.set(font_scale = 1)  
plt.style.use(['fivethirtyeight'])
pd.set_option('display.max_columns', None)

# 한글폰트 사용
import os

if os.name == 'posix':

    plt.rc("font", family="AppleGothic")
else :

    plt.rc("font", family="Malgun Gothic")


# ## LPOINT_BIG_COMP_01_DEMO 데이터 

# In[2]:


demo = pd.read_csv('Data/LPOINT_BIG_COMP_01_DEMO.csv')
demo.head()


# In[3]:


# 컬럼명 변경 

demo.rename(columns = {'zon_hlv':'거주지_대분류'}, inplace  =True )
demo.rename(columns = {'ma_fem_dv':'성별'} ,inplace  =True )
demo.rename(columns = {'cust':'고객번호'},inplace  =True  )
demo.rename(columns = {'ages':'연령'},inplace  =True  )


# In[4]:


# Null값 체크  

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(demo)


# In[13]:


# 연령 컬럼의 문자열 처리 

demo['연령'] = demo['연령'].str.replace('대',' ')
demo = demo.astype({'연령': 'int'})


# In[14]:


# 성별 전환 0 = 여성, 남성 = 1 

demo['성별'] = demo['성별'].str.replace('여성','0')
demo['성별'] = demo['성별'].str.replace('남성','1')


# In[24]:


demo.head(1)


# ## LPOINT_BIG_COMP_02_PDDE 데이터 

# In[5]:


pdde = pd.read_csv('Data/LPOINT_BIG_COMP_02_PDDE.csv')
pdde.head()


# In[6]:


# 컬럼명 변경 

pdde.rename(columns = {'rct_no':'장바구니_식별번호'}, inplace  =True )
pdde.rename(columns = {'chnl_dv':'온오프_구분'} ,inplace  =True )
pdde.rename(columns = {'cop_c':'제휴사_구분'},inplace  =True  )
pdde.rename(columns = {'br_c':'구매점포_구분'},inplace  =True  )
pdde.rename(columns = {'pd_c':'상품_구분'},inplace  =True  )
pdde.rename(columns = {'de_dt':'구매일자'},inplace  =True  )
pdde.rename(columns = {'de_hr':'구매시간'},inplace  =True  )
pdde.rename(columns = {'buy_am':'구매금액'},inplace  =True  )
pdde.rename(columns = {'buy_ct':'구매수량'},inplace  =True  )
pdde.rename(columns = {'cust':'고객번호'},inplace  =True  )


# In[8]:


# Null값 체크  

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(pdde)


# In[9]:


# 구매 점포 97% 이상 null 값 존재로 제거 

pdde.drop(['구매점포_구분'], axis =1, inplace = True)


# In[16]:


# 구매일자 컬럼 데이터 형식으로 변경 

from datetime import datetime

pdde['구매일자']= pdde['구매일자'].astype(str)
pdde['구매일자']=  pd.to_datetime(pdde['구매일자'],  format='%Y-%m-%d') 


# In[17]:


# 날짜 추가 생성 

pdde["년"] = pdde['구매일자'].dt.year
pdde["월"] = pdde['구매일자'].dt.month
pdde["일"] = pdde['구매일자'].dt.day


# In[25]:


pdde.head(1)


# ## LPOINT_BIG_COMP_06_LPAY 데이터

# In[10]:


# 엘페이 이용내용 
lpay = pd.read_csv('Data/LPOINT_BIG_COMP_06_LPAY.csv')
lpay.head()


# In[11]:


# 컬럼명 변경 

lpay.rename(columns = {'rct_no':'구매_식별번호'}, inplace  =True )
lpay.rename(columns = {'cop_c':'제휴사_구분'},inplace  =True  )
lpay.rename(columns = {'chnl_dv':'온오프_구분'} ,inplace  =True )
lpay.rename(columns = {'de_dt':'제휴사_이용일자'},inplace  =True  )
lpay.rename(columns = {'de_hr':'구매시간'},inplace  =True  )
lpay.rename(columns = {'buy_am':'구매금액'},inplace  =True  )
lpay.rename(columns = {'cust':'고객번호'},inplace  =True  )


# In[28]:


# 제휴사_구분 컬럼 변경

lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A01','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A02','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A03','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A04','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A05','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('A06','유통사')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('B01','숙박업종')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('C01','엔터')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('C02','엔터')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('E01','렌탈')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('L00','기타제휴')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('L01','비제휴')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('D01','F&B')
lpay['제휴사_구분'] =lpay['제휴사_구분'].str.replace('D02','F&B')


# In[20]:


# Null값 체크  

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(pdde)


# In[19]:


# 제휴사_이용일자 컬럼 데이터 형식으로 변경 

from datetime import datetime

lpay['제휴사_이용일자']= lpay['제휴사_이용일자'].astype(str)
lpay['제휴사_이용일자']=  pd.to_datetime(lpay['제휴사_이용일자'],  format='%Y-%m-%d') 


# In[29]:


lpay.head(1)


# ## LPOINT_BIG_COMP_04_PD_CLAC

# In[21]:


clac = pd.read_csv('Data/LPOINT_BIG_COMP_04_PD_CLAC.csv')
clac.head()


# In[22]:


clac.rename(columns = {'pd_c':'상품_구분'}, inplace  =True )
clac.rename(columns = {'pd_nm':'상품_소분류'}, inplace  =True )
clac.rename(columns = {'clac_hlv_nm':'상품_중분류'}, inplace  =True )
clac.rename(columns = {'clac_mcls_nm':'상품_대분류'}, inplace  =True )


# In[23]:


# Null값 체크  

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(pdde)


# In[27]:


clac.head(1)

