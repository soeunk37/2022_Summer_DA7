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


# ## 최종 데이터 EDA

# In[56]:


data = pd.read_csv('Data/final_ta_0801.csv')

# 온라인 구매 데이터 사용 
data = data
df = data.copy()
df.head()


# In[49]:


df.drop(['온오프_구분','최근구매일_R'], axis = 1, inplace =True)


# In[50]:


# 제휴사처리 
df = df[df.P_등급 != 0]  


# In[51]:


# 데이터가 임포트되며 처리 요망 , 날짜 전처리 

from datetime import datetime

df['구매일자'] = df['구매일자'].str.replace('/',' ')
df['구매일자'] = df['구매일자'].str.replace(' ','')
df['구매일자'] = str(20)+df['구매일자']

df['구매일자']= df['구매일자'].astype(str)
df['구매일자']=  pd.to_datetime(df['구매일자'],  format='%Y-%m-%d') 


# In[6]:


# 최근 구매일 오류수정  
l = pd.DataFrame(df.groupby('고객번호')['구매일자'].max().reset_index())

current_day = pd.to_datetime('20210101') 
time_diff = l['구매일자'] - current_day
l['최근구매일_R'] = time_diff

l.drop('구매일자' , axis = 1, inplace = True)
df = df.merge(l, on = '고객번호' )


# In[52]:


# 결측치 확인 

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

missing_col = check_missing_col(df)


# In[53]:


# 결측치 평균값으로 대체
m = df['상품중분류_정규화'].mean()
df['상품중분류_정규화'].fillna(m, inplace = True)


# In[54]:


# 엘페이 사용유저 

lpay_df = df[(df.L_등급==1)|(df.L_등급==2)|(df.L_등급==3)|(df.L_등급==4)|(df.L_등급==5)]
lpay_df.head(1)


# In[10]:


display(lpay_df.info(), lpay_df.shape)


# In[11]:


lpay_df.select_dtypes(include=['int64','float64']).columns


# In[12]:


lpay_df.select_dtypes(include=['object']).columns


# In[13]:


# 상관성 측적 
corr= lpay_df.corr(method='pearson')
plt.rcParams["figure.figsize"] = (20,12)
sns.heatmap(corr,
           annot = True, fmt = '.0%',cmap = 'winter',
           vmin = -1, vmax=1 )


# In[14]:


# 수치 데이터 분포 확인 

lpay_df.iloc[:,:-1].select_dtypes(include=['float64','int64']).hist(figsize=(15, 15),color='blue', bins=50, xlabelsize=8, ylabelsize=8)
plt.show()


# In[15]:


# 성별 분포

X = lpay_df['성별'].value_counts()

fig , ax  = plt.subplots(figsize = (10,8))
wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 5}
colors = ["blue","gray"]

ax.pie(x = X.values, labels = X.index, autopct='%.1f%%'
      ,  startangle=260, counterclock=False,  wedgeprops=wedgeprops, textprops ={'size' :20}, colors = colors
      )

# autopct는 부채꼴 안에 표시될 숫자의 형식을 지정합니다. 소수점 한자리까지 표시하도록 설정했습니다.
# startangle는 부채꼴이 그려지는 시작 각도를 설정합니다.
# counterclock=False로 설정하면 시계 방향 순서로 부채꼴 영역이 표시됩니다
plt.legend()
plt.show()


# In[16]:


# 연령별 구매수 

X = lpay_df["연령"].value_counts()
fig ,ax  = plt.subplots(figsize = (15,5))
sns.barplot( X.index, X.values , ax = ax,color = "b")
plt.xticks(rotation = 45)
ax.set_title("연령별 구매수 ")


plt.show()


# In[17]:


# 구매일자 별 구매수

X = lpay_df["구매일자"].value_counts()
fig ,ax  = plt.subplots(figsize = (25,5))
sns.lineplot( X.index, X.values , ax = ax,color = "b")
plt.xticks(rotation = 45)
ax.set_title("구매일자 별 구매수 ")
ax.set_ylabel("구매수")

plt.show()


# In[18]:


# 월별 구매수
X = lpay_df["월"].value_counts()
fig ,ax = plt.subplots(figsize = (25,5))
sns.lineplot( X.index, X.values , ax = ax,color = "b")
ax.set_title(" 월별 구매수 ")
ax.set_ylabel("구매수")
plt.show()


# In[19]:


# 일별 구매수 

y = lpay_df["일"].value_counts()
fig ,ax = plt.subplots(figsize = (25,5))
sns.lineplot( y.index, y.values , ax = ax,color = "b")
ax.set_title(" 일별 구매수 ")
ax.set_ylabel("구매수")
plt.show()


# In[20]:


# 구매시간에 따른 구매 수 

X = lpay_df['구매시간'].value_counts()
fig ,ax = plt.subplots(figsize = (8,8))
sns.barplot( X.index, X.values , ax = ax,color = "b")
ax.set_title(" 구매시간 따른 구매수  ")
ax.set_ylabel("구매수")
plt.show()


# ## 코호트 분석 

# In[21]:


# 코호트 분석 
cohort = lpay_df[['고객번호','구매일자']]
cohort['월_주문'] = cohort['구매일자'].dt.strftime('%Y-%m')
cohort.set_index('고객번호', inplace = True)


# In[22]:


# 고객 각각의 첫 구매기간 추출 

cohort['첫구매월'] = cohort.groupby(level = 0)['구매일자'].min().apply(lambda x :x.strftime('%y-%m'))
cohort.reset_index(inplace= True)
cohort.head()


# In[23]:


# 첫 구매일과 구매날짜를 기준으로 고객 수, 주문 수, 총매출 합계 계산 

g = cohort.groupby(['첫구매월','월_주문'])

c = g.agg({'고객번호': pd.Series.nunique})

c.rename(columns ={ '고객번호': '고객수'}, inplace = True)
c.head()


# In[24]:


# 년월 - 년월의 패턴을 년원 - 소요기간(월) 로 변환 

def cohort_period(cohort) :
    cohort['코호트_기간'] = np.arange(len(cohort)) +1 
    return cohort


# In[25]:


c = c.groupby(level = 0).apply(cohort_period)
c.head()


# In[26]:


# 리텐션 결과를 비율로 나타내기 위해 각각 첫 구매일에 따른 고객수 도출 

c.reset_index(inplace = True)
c.set_index(['첫구매월','코호트_기간'], inplace = True)

c_group_size = c['고객수'].groupby(level = 0).first()
c_group_size.head()


# In[27]:


user_retention = c['고객수'].unstack(0).divide(c_group_size, axis = 1)
user_retention.head(10)


# In[28]:


fig ,ax = plt.subplots(figsize = (12,10))

sns.heatmap(user_retention.T, mask = user_retention.T.isnull(), annot = True, fmt = '.0%',cmap= 'cool')
ax.set_title(" Cohorts : User_Retention  ")
plt.show()


# In[29]:


Day_df = lpay_df[['일','고객번호']]


# In[30]:


R = Day_df.drop_duplicates()
G = R.groupby(['일']).size()


# In[31]:


G.sort_values(ascending =False).head(5)


# In[32]:


fig , ax = plt.subplots( figsize = (10, 8))

ax.plot(G, color = 'b')
ax.set_title('엘페이_DAU')
plt.show()


# In[33]:


Day_df = lpay_df[['월','고객번호']]


# In[34]:


R = Day_df.drop_duplicates()
G = R.groupby(['월']).size()


# In[35]:


fig , ax = plt.subplots( figsize = (10, 8))

ax.plot(G,color= 'b' )
ax.set_title('엘페이_MAU')
plt.show()


# In[36]:


# 제휴사별_사용금액 

x = lpay_df.groupby('제휴사_구분')['구매금액_X'].mean().round().sort_values(ascending = False)

fig ,ax = plt.subplots(figsize = (10,5))
sns.barplot( x.index, x.values , ax = ax, color = 'b')
ax.set_title(" 제휴사별_사용금액 ")
plt.show()


# In[37]:


# 거주지별_사용금액

x = lpay_df.groupby('거주지_대분류')['구매금액_X'].mean().round().sort_values(ascending = False)

fig ,ax = plt.subplots(figsize = (10,5))
sns.barplot( x.index, x.values , ax = ax, color = 'b')
ax.set_title(" 제휴사별_사용금액 ")
plt.show()


# In[38]:


# 주중주말
x = lpay_df.groupby('주중주말')['구매금액_X'].mean().round().sort_values(ascending = False)

fig ,ax = plt.subplots(figsize = (10,5))
sns.barplot( x.index, x.values , ax = ax, color = 'b')
ax.set_title(" 주중주말_사용금액 ")
plt.show()


# In[39]:


# 요일별 인당 평균금액 높은 고객 10명 

x = lpay_df.groupby('고객번호')['요일별_인당_평균결제금액'].mean().round().sort_values(ascending = False).head(10)

fig ,ax = plt.subplots(figsize = (10,5))
sns.barplot( x.index, x.values , ax = ax, color = 'b')
ax.set_title(" 요일별 인당 평균금액 높은 고객 10명 ")
plt.xticks(rotation =90)
plt.show()

