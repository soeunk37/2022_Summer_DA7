#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[10]:


make_buy = pd.read_csv('Data/make_buy.csv',index_col =0)
make_buy.head()


# ## 구매테이블 RFM 등급 파생변수 생성
# 
# * Recency : 얼마나 최근에 구매했는가
# * Frequency : 얼마나 자주 구매했는가
# * Monetary : 얼마나 많은 금액을 지출했는가
# ---- 
# 
# 즉, 사용자별로 얼마나 최근에, 얼마나 자주, 얼마나 많은 금액을 지출했는지에 따라 사용자들의 분포를 확인 하거나 사용자 그룹(또는 등급)을 나누어 분류 하는 분석 기법입니다. 구매 가능성이 높은 고객을 선정할 때 용이한 데이터 분석방법이라고 알려져 있고, 또 사용자들의 평소 구매 패턴을 기준으로 분류를 진행하기 때문에 각 사용자 그룹의 특성에 따라 차별화된 마케팅 메세지를 전달할 수 있습니다. 

# In[ ]:


from tqdm import tqdm

customer_id = list(make_buy['고객번호'].unique())
monetary_df = pd.DataFrame() 
monetary_df['고객번호'] = customer_id 
 
    
monetary_data = [] 
for ci in tqdm(customer_id,position=0,desc='Calculating amount of individual customer'):
    
    temp = make_buy.query('고객번호==@ci') 
    amount = sum(temp['구매금액']) 
    monetary_data.append(amount)

monetary_df['구매금액'] = monetary_data 

# 각 고객별 최근방문일
temp_recency_df = make_buy[['고객번호','구매일자']].drop_duplicates() 
recency_df = temp_recency_df.groupby('고객번호')['구매일자'].max().reset_index() 
recency_df = recency_df.rename(columns={'구매일자':'최근방문일'})
 
# 각 고객별 방문횟수
temp_frequency_df = make_buy[['고객번호','구매일자']].drop_duplicates() 
frequency_df = temp_frequency_df.groupby('고객번호')['구매일자'].count().reset_index() 
frequency_df = frequency_df.rename(columns={'구매일자':'방문빈도'})

# 데이터를 고객아이디를 기준으로 병합
rfm_df = pd.merge(recency_df,frequency_df,how='left',on='고객번호')
rfm_df = pd.merge(rfm_df,monetary_df,how='left',on='고객번호')


# In[ ]:


# 데이터저장 
rfm_df.to_csv('rfm.csv')


# In[ ]:


# 해석 방향 동일하게 하기 위해 2021-01-01 기준으로 적용 , 즉 최근 방문일에 큰 점수 

current_day = pd.to_datetime('20210101') 
time_diff = rfm_df['최근방문일']-current_day ##최근방문일과 기준 날짜의 시간 차이
time_in_seconds = [x.total_seconds() for x in time_diff] #시간 차이를 초단위로 계산
rfm_df['최근방문일'] = time_in_seconds # 변환된 데이터를 다시 삽입


# In[ ]:


def get_score(level, data):
    '''
    Description :
    level안에 있는 원소를 기준으로
    1 ~ len(level)+ 1 까지 점수를 부여하는 함수
    
    Parameters :
    level = 튜플 또는 리스트 타입의 숫자형 데이터이며 반드시 오름차순으로 정렬되어 있어야함.
    예 - [1,2,3,4,5] O, [5,4,3,2,1] X, [1,3,2,10,4] X 
    data = 점수를 부여할 데이터. 순회가능한(iterable) 데이터 형식
    return :
    점수를 담고 있는 리스트 반환
    '''
    score = [] 
    for j in range(len(data)): 
        for i in range(len(level)): 
            if data[j] <= level[i]: 
                score.append(i+1) 
                break 
            elif data[j] > max(level): 
                score.append(len(level)+1) 
                break 
            else: 
                continue 
    return score
    
def get_rfm_grade(df, num_class, rfm_tick_point, rfm_col_map, suffix=None):
    '''
    Description :
    개별 고객에 대한 최근방문일/방문횟수/구매금액 데이터가 주어졌을때
    최근방문일/방문횟수/구매금액 점수를 계산하여 주어진 데이터 오른쪽에 붙여줍니다.
    
    Parameters :
    df = pandas.DataFrame 데이터
    num_class = 등급(점수) 개수
    rfm_tick_point = 최근방문일/방문횟수/구매금액에 대해서 등급을 나눌 기준이 되는 값
                    'quantile', 'min_max' 또는 리스트를 통하여 직접 값을 정할 수 있음.
                    단, 리스트 사용시 원소의 개수는 반드시 num_class - 1 이어야함.
                    quatile = 데이터의 분위수를 기준으로 점수를 매김
                    min_max = 데이터의 최소값과 최대값을 동일 간격으로 나누어 점수를 매김
    rfm_col_map = 최근방문일/방문횟수/구매금액에 대응하는 칼럼명
    예 - {'R':'Recency','F':'Frequency','M':'Monetary'}
    suffix = 최근방문일/방문횟수/구매금액에 대응하는 칼럼명 뒤에 붙는 접미사
    Return : 
    pandas.DataFrame
    '''
    
    
    from sklearn import preprocessing
    
    
    if not isinstance(df, pd.DataFrame): 
        print('데이터는 pandas.DataFrame 객체여야 합니다.')
        return
    
    if isinstance(rfm_tick_point, dict) == False or isinstance(rfm_col_map, dict) == False: 
        print(f'rfm_tick_point와 rfm_col_map은 모두 딕셔너리여야합니다.')
        return
    
    if len(rfm_col_map) != 3:
        print(f'rfm_col_map인자는 반드시 3개의 키를 가져야합니다. \n현재 rfm_col_map에는 {len(rfm_col_map)}개의 키가 있습니다.')
        return
    
    if len(rfm_tick_point) != 3: 
        print(f'rfm_tick_point인자는 반드시 3개의 키를 가져야합니다. \n현재 rfm_col_map에는 {len(rfm_col_map)}개의 키가 있습니다.')
        return
    
    if set(rfm_tick_point.keys()) != set(rfm_col_map.keys()): 
        print(f'rfm_tick_point와 rfm_col_map은 같은 키를 가져야 합니다.')
        return
    
    if not set(rfm_col_map.values()).issubset(set(df.columns)):
        not_in_df = set(rfm_col_map.values())-set(df.columns)
        print(f'{not_in_df}이 데이터 칼럼에 있어야 합니다.')
        return
    
    for k, v in rfm_tick_point.items():
        if isinstance(v, str):
            if not v in ['quantile','min_max']:
                print(f'{k}의 값은 "quantile" 또는 "min_max"중에 하나여야 합니다.')
                return
        elif isinstance(v,list) or isinstance(v,tuple):
            if len(v) != num_class-1:
                print(f'{k}에 대응하는 리스트(튜플)의 원소는 {num_class-1}개여야 합니다.')
                return
    
    if suffix:
        if not isinstance(suffix, str):
            print('suffix인자는 문자열이어야합니다.')
            return
        
    #최근방문일/방문횟수/구매금액 점수 부여
    for k, v in rfm_tick_point.items():
        if isinstance(v,str):
            if v == 'quantile':
                
                ## 데이터 변환
                scale = preprocessing.StandardScaler()
                temp_data = np.array(df[rfm_col_map[k]]) 
                temp_data = temp_data.reshape((-1,1)) 
                temp_data = scale.fit_transform(temp_data) 
                temp_data = temp_data.squeeze() 
 
                ## 분위수 벡터
                quantiles_level = np.linspace(0,1,num_class+1)[1:-1] 
                quantiles = [] 
                for ql in quantiles_level:
                    quantiles.append(np.quantile(temp_data,ql)) 
                    
            else: ## min_max인 경우
               
                temp_data = np.array(df[rfm_col_map[k]])
 
                ## 등분점 계산
                quantiles = np.linspace(np.min(temp_data),np.max(temp_data),num_class+1)[1:-1] 
        else: 
            temp_data = np.array(df[rfm_col_map[k]])
            quantiles = v 
        score = get_score(quantiles, temp_data) 
        new_col_name = rfm_col_map[k]+'_'+k 
        if suffix:
            new_col_name = rfm_col_map[k]+'_'+suffix
        df[new_col_name] = score 
    return df


# In[ ]:


# 4분위수로 구간 나눔 
rfm_tick_point={'R':'quantile','F':'quantile','M':'quantile'}
rfm_col_map={'R':'최근방문일','F':'방문빈도','M':'구매금액'}
 
result = get_rfm_grade(df=rfm_df, num_class=5, rfm_tick_point=rfm_tick_point, rfm_col_map=rfm_col_map)

# 데이터 저장 
result.to_csv('result.csv',index=False)


# ## RFM 점수 도출 
# 
# 1. 매출 기여도의 표준편차를 최대화하는 가중치를 구함
# 2. 가중치와 RFM점수를 이용하여 고객별로 등급을 부여
# 3. 각 등급렬 매출기여도를 확인 

# In[ ]:


rfm_s = pd.read_csv('result.csv')
rfm_s.head()


# In[ ]:


rfm_s = rfm_s[['고객번호','구매금액','최근방문일_R','방문빈도_F','구매금액_M']]


# In[ ]:


def get_score(level, data, reverse = False):
    score = [] 
    for j in range(len(data)): 
        for i in range(len(level)): 
            if data[j] <= level[i]: 
                score.append(i+1) 
                break 
            elif data[j] > max(level): 
                score.append(len(level)+1) 
                break 
            else: 
                continue
    if reverse:
        return [len(level)+2-x for x in score]
    else:
        return score 
 
grid_number = 100 
weights = []
for j in range(grid_number+1):
    weights += [(i/grid_number,j/grid_number,(grid_number-i-j)/grid_number)
                  for i in range(grid_number+1-j)]
num_class = 5 
class_level = np.linspace(1,5,num_class+1)[1:-1] 
total_amount_of_sales = rfm_s['구매금액'].sum()


# In[ ]:


# 등급을 총 5개로 정함 

num_class = 5
class_level = np.linspace(1, 5, num_class+1)[1 : -1]
print(class_level)


# In[ ]:


from tqdm import tqdm

max_std = 0

for w in tqdm(weights,position=0,desc = '[Finding Optimal weights]'):
   
    score = w[0]*rfm_s['최근방문일_R'] +                         w[1]*rfm_s['방문빈도_F'] +                         w[2]*rfm_s['구매금액_M'] 
    rfm_s['등급'] = get_score(class_level,score,True) 
    
    ## 등급별로 구매금액을 집계
    grouped_rfm_score = rfm_s.groupby('등급')['구매금액'].sum().reset_index()
    
   
    grouped_rfm_score = grouped_rfm_score.sort_values('등급')
    
    temp_monetary = list(grouped_rfm_score['구매금액'])
    if temp_monetary != sorted(temp_monetary,reverse=True):
        continue
    
    # 클래스별 구매금액을 총구매금액으로 나누어 클래스별 매출 기여도 계산
    grouped_rfm_score['구매금액'] = grouped_rfm_score['구매금액'].map(lambda x : x/total_amount_of_sales)
    std_sales = grouped_rfm_score['구매금액'].std() 
    if max_std <= std_sales:
        max_std = std_sales 
        optimal_weights = w  


# In[ ]:


score = optimal_weights[0]*rfm_s['최근방문일_R'] +         optimal_weights[1]*rfm_s['방문빈도_F'] +         optimal_weights[2]*rfm_s['구매금액_M'] ## 고객별 점수 계산
 
rfm_s['등급'] = get_score(class_level,score,True) ## 고객별 등급 부여


# In[ ]:


#클래스별 고객 수 계산
temp_rfm_score1 = rfm_s.groupby('등급')['고객번호'].count().reset_index().rename(columns={'고객번호':'고객수'})
 
# 클래스별 구매금액(매출)계산
temp_rfm_score2 = rfm_s.groupby('등급')['구매금액'].sum().reset_index()
 
# 클래스별 매출 기여도 계산
temp_rfm_score2['기여도'] = temp_rfm_score2['구매금액'].map(lambda x : x/total_amount_of_sales)
 
# 데이터 결합
result_df = pd.merge(temp_rfm_score1,temp_rfm_score2,how='left',on=('등급'))


# In[ ]:


# 데이터 저장 
result_df.to_csv('rfm_result_df.csv')


# ## Lpay RFM 등급 파생변수 생성

# In[14]:


lpay = pd.read_csv('Data/LPOINT_BIG_COMP_06_LPAY.csv')
lpay.rename(columns = {'rct_no':'구매_식별번호'}, inplace  =True )
lpay.rename(columns = {'cop_c':'제휴사_구분'},inplace  =True  )
lpay.rename(columns = {'chnl_dv':'온오프_구분'} ,inplace  =True )
lpay.rename(columns = {'de_dt':'제휴사_이용일자'},inplace  =True  )
lpay.rename(columns = {'de_hr':'구매시간'},inplace  =True  )
lpay.rename(columns = {'buy_am':'구매금액'},inplace  =True  )
lpay.rename(columns = {'cust':'고객번호'},inplace  =True  )
lpay.head()


# In[ ]:


from tqdm import tqdm

customer_id = list(lpay['고객번호'].unique()) 
monetary_df = pd.DataFrame() 
monetary_df['고객번호'] = customer_id 
 
    
monetary_data = [] 
for ci in tqdm(customer_id,position=0):
    
    temp = lpay.query('고객번호==@ci') 
    amount = sum(temp['구매금액']) 
    monetary_data.append(amount)

monetary_df['구매금액'] = monetary_data 


# 각 고객별 최근방문일
recency_df = lpay[['고객번호','제휴사_이용일자']].drop_duplicates() 
recency_df = recency_df.groupby('고객번호')['제휴사_이용일자'].max().reset_index() 
recency_df = recency_df.rename(columns={'제휴사_이용일자':'최근방문일'})
 
# 각 고객별 방문횟수
frequency_df = lpay[['고객번호','제휴사_이용일자']].drop_duplicates() 
frequency_df = frequency_df.groupby('고객번호')['제휴사_이용일자'].count().reset_index() 
frequency_df = frequency_df.rename(columns={'제휴사_이용일자':'방문빈도'})

# 데이터를 고객아이디를 기준 병합 
rfm_df = pd.merge(recency_df,frequency_df,how='left',on='고객번호')
rfm_df = pd.merge(rfm_df,monetary_df,how='left',on='고객번호')


# In[ ]:


# 데이터 저장 
rfm_df.to_csv('lay_rmf_')


# In[ ]:


# 해석 방향 동일하게 하기 위해 2021-01-01 기준으로 적용 , 즉 최근 방문일에 큰 점수 

current_day = pd.to_datetime('20210101') 
time_diff = rfm_df['최근방문일']-current_day ## 최근방문일과 기준 날짜의 시간 차이
time_in_seconds = [x.total_seconds() for x in time_diff] ## 시간 차이를 초단위로 계산
rfm_df['최근방문일'] = time_in_seconds ## 변환된 데이터를 다시 삽입


# In[ ]:


def get_score(level, data):
    
    score = [] 
    for j in range(len(data)): 
        for i in range(len(level)): 
            if data[j] <= level[i]: 
                score.append(i+1) 
                break 
            elif data[j] > max(level): 
                score.append(len(level)+1) 
                break 
            else: 
                continue 
    return score
    
def get_rfm_grade(df, num_class, rfm_tick_point, rfm_col_map, suffix=None):
    '''
    Description :
    개별 고객에 대한 최근방문일/방문횟수/구매금액 데이터가 주어졌을때
    최근방문일/방문횟수/구매금액 점수를 계산하여 주어진 데이터 오른쪽에 붙여줍니다.
    
    Parameters :
    df = pandas.DataFrame 데이터
    num_class = 등급(점수) 개수
    rfm_tick_point = 최근방문일/방문횟수/구매금액에 대해서 등급을 나눌 기준이 되는 값
                    'quantile', 'min_max' 또는 리스트를 통하여 직접 값을 정할 수 있음.
                    단, 리스트 사용시 원소의 개수는 반드시 num_class - 1 이어야함.
                    quatile = 데이터의 분위수를 기준으로 점수를 매김
                    min_max = 데이터의 최소값과 최대값을 동일 간격으로 나누어 점수를 매김
    rfm_col_map = 최근방문일/방문횟수/구매금액에 대응하는 칼럼명
    예 - {'R':'Recency','F':'Frequency','M':'Monetary'}
    suffix = 최근방문일/방문횟수/구매금액에 대응하는 칼럼명 뒤에 붙는 접미사
    Return : 
    pandas.DataFrame
    '''
    ##### 필요모듈 체크
    
    from sklearn import preprocessing
    
    
    # 최근방문일/방문횟수/구매금액 점수 부여
    for k, v in rfm_tick_point.items():
        if isinstance(v,str):
            if v == 'quantile':
                #  데이터 변환
                scale = preprocessing.StandardScaler()
                temp_data = np.array(df[rfm_col_map[k]])
                temp_data = temp_data.reshape((-1,1)) 
                temp_data = scale.fit_transform(temp_data) 
                temp_data = temp_data.squeeze() 
 
                ## 분위수 벡터
                quantiles_level = np.linspace(0,1,num_class+1)[1:-1] 
                quantiles = [] 
                for ql in quantiles_level:
                    quantiles.append(np.quantile(temp_data,ql)) 
            else: 
                ## min_max인 경우
                temp_data = np.array(df[rfm_col_map[k]])
 
                ## 등분점 계산
                quantiles = np.linspace(np.min(temp_data),np.max(temp_data),num_class+1)[1:-1] 
        else: ## 직접 구분값을 넣어주는 경우
            temp_data = np.array(df[rfm_col_map[k]])
            quantiles = v 
        score = get_score(quantiles, temp_data) #구분값을 기준으로 점수를 부여하고 리스트로 저장한다.
        new_col_name = rfm_col_map[k]+'_'+k # 점수값을 담는 변수의 이름
        if suffix:
            new_col_name = rfm_col_map[k]+'_'+suffix
        df[new_col_name] = score 
    return df


# In[ ]:


# 4분위수로 구간 나눔 
rfm_tick_point={'R':'quantile','F':'quantile','M':'quantile'}
rfm_col_map={'R':'최근방문일','F':'방문빈도','M':'구매금액'}
 
result = get_rfm_grade(df=rfm_df, num_class=5, rfm_tick_point=rfm_tick_point, rfm_col_map=rfm_col_map)

# 데이터 저장 
result.to_csv('result.csv',index=False)


# ## RFM 점수 도출 
# 
# 1. 매출 기여도의 표준편차를 최대화하는 가중치를 구함
# 2. 가중치와 RFM점수를 이용하여 고객별로 등급을 부여
# 3. 각 등급렬 매출기여도를 확인 

# In[ ]:


rfm_r = result
rfm_s = rfm_r[['고객번호','구매금액','최근방문일_R','방문빈도_F','구매금액_M']]


# In[ ]:


def get_score(level, data, reverse = False):
    '''
    Description :
    level안에 있는 원소를 기준으로
    1 ~ len(level)+ 1 까지 점수를 부여하는 함수
    
    Parameters :
    level = 튜플 또는 리스트 타입의 숫자형 데이터이며 반드시 오름차순으로 정렬되어 있어야함.
    예 - [1,2,3,4,5] O, [5,4,3,2,1] X, [1,3,2,10,4] X 
    data = 점수를 부여할 데이터. 순회가능한(iterable) 데이터 형식
    reverse = 점수가 높을 때 그에 해당하는 값을 낮게 설정하고 싶을 때 True
    return :
    점수를 담고 있는 리스트 반환
    '''
    score = [] 
    for j in range(len(data)): 
        for i in range(len(level)): 
            if data[j] <= level[i]: 
                score.append(i+1) 
                break 
            elif data[j] > max(level): 
                score.append(len(level)+1) 
                break 
            else: 
                continue
    if reverse:
        return [len(level)+2-x for x in score]
    else:
        return score 
 
grid_number = 150 
weights = []
for j in range(grid_number+1):
    weights += [(i/grid_number,j/grid_number,(grid_number-i-j)/grid_number)
                  for i in range(grid_number+1-j)]
num_class = 5 
class_level = np.linspace(1,5,num_class+1)[1:-1] 
total_amount_of_sales = rfm_s['구매금액'].sum() 


# In[ ]:


max_std = 
for w in tqdm(weights,position=0,desc = '[Finding Optimal weights]'):
   
    score = w[0]*rfm_s['최근방문일_R'] +                         w[1]*rfm_s['방문빈도_F'] +                         w[2]*rfm_s['구매금액_M'] 
    #점수를 이용하여 고객별 등급 부여
    rfm_s['등급'] = get_score(class_level,score,True) 
    
    # 등급별로 구매금액을 집계
    grouped_rfm_score = rfm_s.groupby('등급')['구매금액'].sum().reset_index()
    
    # 제약조건 추가 - 등급이 높은 고객들의 매출이 낮은 등급의 고객들 보다 커야함
    grouped_rfm_score = grouped_rfm_score.sort_values('등급')
    
    temp_monetary = list(grouped_rfm_score['구매금액'])
    if temp_monetary != sorted(temp_monetary,reverse=True):
        continue
    
    # 클래스별 구매금액을 총구매금액으로 나누어 클래스별 매출 기여도 계산
    grouped_rfm_score['구매금액'] = grouped_rfm_score['구매금액'].map(lambda x : x/total_amount_of_sales)
    std_sales = grouped_rfm_score['구매금액'].std() 
    if max_std <= std_sales:
        max_std = std_sales 
        optimal_weights = w  


# In[ ]:


score = optimal_weights[0]*rfm_s['최근방문일_R'] +         optimal_weights[1]*rfm_s['방문빈도_F'] +         optimal_weights[2]*rfm_s['구매금액_M'] 
 
rfm_s['등급'] = get_score(class_level,score,True) 


# In[ ]:


#클래스별 고객 수 계산
temp_rfm_score1 = rfm_s.groupby('등급')['고객번호'].count().reset_index().rename(columns={'고객번호':'고객수'})
 
#클래스별 구매금액(매출)계산
temp_rfm_score2 = rfm_s.groupby('등급')['구매금액'].sum().reset_index()
 
#클래스별 매출 기여도 계산
temp_rfm_score2['기여도'] = temp_rfm_score2['구매금액'].map(lambda x : x/total_amount_of_sales)
 
#데이터 결합
result_df = pd.merge(temp_rfm_score1,temp_rfm_score2,how='left',on=('등급'))


# ## 상품_중분류_정규화 파생변수 도출 

# In[16]:


add_df = pd.read_csv('add_df.csv', index_col= 0)
add_df.head()


# In[ ]:


df_goods = add_df.groupby(['상품_중분류']).sum()[['구매금액']].reset_index()

def normalize(df):
    result = df.copy()
    for 구매금액 in df.columns:
        max_value = df['구매금액'].max()
        min_value = df['구매금액'].min()
        result['구매금액_정규화'] = (df['구매금액'] - min_value) / (max_value - min_value)
    return result

df_goods2 = normalize(df_goods)
    
df_goods3 = df_goods2[['상품_중분류','구매금액_정규화']]
df_goods3.rename(columns={'구매금액_정규화':'상품중분류_정규화'},inplace=True)

res = pd.merge(add_df,df_goods3,how='left',on='상품_중분류')
res2 = res.drop(['상품_소분류','상품_중분류'],axis=1)


# ## 주중주말 파생변수 도출 

# In[ ]:


add_df['주중구분'] = add_df['구매일자'].dt.dayofweek

# 0-4 : 주중 / 5,6 :주말


# In[ ]:


add_df['주중주말']= add_df.apply(lambda x : '주말' if x['주중구분']>=5 else '주중', axis=1)

