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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report


# In[116]:


data = pd.read_csv('Data/final_ta_0801.csv')

# 온라인 구매 데이터 사용 
data = data[data['온오프_구분'] ==2] 
df = data.copy()
df.head()


# In[117]:


# 제휴사처리 
df = df[df.P_등급 != 0]  


# In[118]:


# 데이터가 임포트되며 처리 요망 , 날짜 전처리 

from datetime import datetime

df['구매일자'] = df['구매일자'].str.replace('/',' ')
df['구매일자'] = df['구매일자'].str.replace(' ','')
df['구매일자'] = str(20)+df['구매일자']

df['구매일자']= df['구매일자'].astype(str)
df['구매일자']=  pd.to_datetime(df['구매일자'],  format='%Y-%m-%d') 


# In[119]:


# 사용하지 않은 컬럼 제거 

df.drop(['LPAY_결제액수','LPAY_결제횟수','온오프_구분','최근구매일_R'], axis = 1, inplace =True)


# In[120]:


# 최근 구매일 오류수정  
l = pd.DataFrame(df.groupby('고객번호')['구매일자'].max().reset_index())

current_day = pd.to_datetime('20210101') 
time_diff = l['구매일자'] - current_day
l['최근구매일_R'] = time_diff

l.drop('구매일자' , axis = 1, inplace = True)
df = df.merge(l, on = '고객번호' )


# In[121]:


df


# In[7]:


# 날짜 유닉스 형태로 변경
df['구매일자'] = df['구매일자'].map(datetime.toordinal)


# In[8]:


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


# In[9]:


# 결측치 평균값으로 대체
m = df['상품중분류_정규화'].mean()
df['상품중분류_정규화'].fillna(m, inplace = True)


# ## StandardScaler

# In[10]:


df['구매일자']= df['구매일자'].astype(int)
df['구매시간']= df['구매시간'].astype(int)
df['구매금액_X']= df['구매금액_X'].astype(int)
df['성별']= df['성별'].astype(int)
df['연령']= df['연령'].astype(int)
df['년']= df['년'].astype(int)
df['월']= df['월'].astype(int)
df['일']= df['일'].astype(int)
df['최근구매일_R']= df['최근구매일_R'].astype(int)
df['상품중분류_정규화']= df['상품중분류_정규화'].astype(int)
df['주중주말']= df['주중주말'].astype(int)
df['요일별_인당_평균결제금액']= df['요일별_인당_평균결제금액'].astype(int)
df['P_결제금액']= df['P_결제금액'].astype(int)
df['P_결제횟수']= df['P_결제횟수'].astype(int)
df['P_결제액수평균']= df['P_결제액수평균'].astype(int)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df[[
 '구매일자',
 '구매시간',
 '구매금액_X',
 '성별',
 '연령',
 '년',
 '월',
 '일',
 '상품중분류_정규화',
 '주중주말',
 '요일별_인당_평균결제금액',
    '최근구매일_R' ,
    'P_결제금액',
 'P_결제횟수',
 'P_결제액수평균']])

df[[
 '구매일자',
 '구매시간',
 '구매금액_X',
 '성별',
 '연령',
 '년',
 '월',
 '일',
 '상품중분류_정규화',
 '주중주말',
 '요일별_인당_평균결제금액',
    '최근구매일_R',
'P_결제금액',
 'P_결제횟수',
 'P_결제액수평균']] = scaler.transform(df[[
 '구매일자',
 '구매시간',
 '구매금액_X',
 '성별',
 '연령',
 '년',
 '월',
 '일',
 '상품중분류_정규화',
'최근구매일_R',
 '주중주말',
 '요일별_인당_평균결제금액',
'P_결제금액',
 'P_결제횟수',
 'P_결제액수평균']])


# ## LabelEncoder

# In[12]:


le = LabelEncoder()
df['고객번호']=  le.fit_transform(df['고객번호'])
df['제휴사_구분']=  le.fit_transform(df['제휴사_구분'])
df['거주지_대분류']=  le.fit_transform(df['거주지_대분류'])
df['구매타입']=  le.fit_transform(df['구매타입'])


# In[13]:


# 데이터 확인 

df.head(1)


# ## Modeling

# In[49]:


from xgboost import XGBClassifier
import xgboost as xgb
from catboost import Pool, CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss
import optuna
from optuna.samplers import TPESampler


# In[15]:


train_df = df[(df.L_등급==1)|(df.L_등급==2)|(df.L_등급==3)|(df.L_등급==4)|(df.L_등급==5)]
test_df = df[df.L_등급 == 0].drop(['L_등급'],axis=1)


# In[16]:


# 엘페이 유저 사용 
target = 'L_등급'
X = train_df.drop('L_등급', axis =1)
y = train_df[target]
X_t = test_df


# In[25]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                  random_state=2017, 
                                                  test_size=0.3, shuffle=True)


# In[18]:


# 피쳐 확인 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest( k=10)
bestfeatures

fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
print(featureScores.nlargest(10,'Score'))  


# In[22]:


num_folds = 5
scoring = 'accuracy'

models = []
models.append(('CART', DecisionTreeClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))


# In[23]:


seed = 42
results = []
names = []

for name, model in models :
    kfold = KFold(n_splits= num_folds , random_state= seed , shuffle= True)
    cv_results = cross_val_score(model , X_train, y_train , cv = kfold , scoring= scoring)
    
    results.append(cv_results)
    names.append(name)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ### Accuracy  

# In[26]:


# 점수 높은 RF로 예측 

rf = RandomForestClassifier()
rf.fit(X_train ,y_train)

rf_pred = rf.predict(X_valid)
accuracy_score(y_valid,rf_pred )


# ### classification_report

# In[28]:


# 오버 피팅이 있는것으로 보임 

from sklearn.metrics import classification_report

print(classification_report(y_valid, rf_pred))


# ### Logloss

# In[62]:


from sklearn.metrics import log_loss

rf_proba = rf.predict_proba(X_valid)
log_score = log_loss(y_valid, rf_proba)
log_score


# ### Roc_Auc 

# In[102]:


from sklearn.metrics import roc_auc_score

pred = rf.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# 랜덤 포레스트는 오버피팅의 양상을 보이고 있기 때문에 사용하지 않음 

# ## Catboost 

# In[31]:


train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_valid, label=y_valid.values) 


# In[40]:


cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    random_strength=0.1,
    depth=8,
    loss_function='MultiClass',
     custom_loss=['AUC', 'Accuracy'],
    leaf_estimation_method='Newton'
)


# In[41]:


cat.fit(train_pool,plot=True,eval_set=test_pool)


# ### Accuracy

# In[42]:


p = cat.predict(X_valid)
accuracy_score(y_valid, p)


# ### classification_report

# In[43]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, p))


# ### Logloss

# In[63]:


from sklearn.metrics import log_loss

cat_proba = cat.predict_proba(X_valid)
log_score = log_loss(y_valid, cat_proba)
log_score


# ### Roc_Auc 

# In[103]:


from sklearn.metrics import roc_auc_score

pred = cat.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# ## LightGBM

# In[44]:


lgbm = LGBMClassifier(objective = 'multiclass',metric = 'multi_logloss', ) 

lgbm.fit(X_train, y_train)


# ### Accuracy  

# In[45]:


lgbm_pred = lgbm.predict(X_valid)
accuracy_score(y_valid, lgbm_pred)


# ### classification_report

# In[46]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, lgbm_pred))


# ### Logloss

# In[64]:


from sklearn.metrics import log_loss

lgbm_proba = lgbm.predict_proba(X_valid)
log_score = log_loss(y_valid, lgbm_proba)
log_score


# ### Roc_Auc 

# In[104]:


from sklearn.metrics import roc_auc_score

pred = lgbm.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# ## XGB

# In[50]:


xgb_c = XGBClassifier(learning_rate=0.1,
                    n_estimators=100,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multi:softprob',
                    nthread=4,
                    num_class=9,
                    seed=42)


# In[56]:


xgb_c.fit(X_train, y_train)


# ### Accuracy  

# In[57]:


xgb_pred = xgb_c.predict(X_valid)
accuracy_score(y_valid, xgb_pred)


# ### classification_report

# In[58]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, xgb_pred))


# ### Logloss

# In[65]:


from sklearn.metrics import log_loss

xgb_proba = xgb_c.predict_proba(X_valid)
log_score = log_loss(y_valid, xgb_proba)
log_score


# ### Roc_Auc 

# In[105]:


from sklearn.metrics import roc_auc_score

pred = xgb_c.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# ## 점수가 유의미한 Catboost와 LightGBM 데이터 불균형 해소 
# ## 하이퍼 파라미터 튜닝  및 결과 확인 
# ### SMOTE 처리 후 비교 

# In[68]:


from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

X_SMOTE_train, X_SMOTE_valid, y_SMOTE_train, y_SMOTE_valid = train_test_split(X_SMOTE, y_SMOTE, test_size=0.3, random_state=2017)


# In[69]:


SMOTE_train_pool = Pool(data=X_SMOTE_train, label=y_SMOTE_train)
SMOTE_test_pool = Pool(data=X_SMOTE_valid, label=y_SMOTE_valid.values) 


# In[71]:


# cat_features =[0,1,2,5,6,7,8,15,18]

SMOTE_cat = CatBoostClassifier(
    iterations=300,
    random_seed=63,
    learning_rate=0.5,
    custom_loss=['AUC', 'Accuracy']
)
SMOTE_cat.fit(
    SMOTE_train_pool,
    #. cat_features=cat_features,
    eval_set= SMOTE_test_pool,
    verbose=False,
    plot=True
)


# ### Accuracy   - SMOTE 처리 후 , 기존 0.84744  ->  0.95102로 상승

# In[73]:


SMOTE_cat_pred = SMOTE_cat.predict(X_valid)
accuracy_score(y_valid, SMOTE_cat_pred)


# In[74]:


from sklearn.metrics import classification_report

print(classification_report(y_valid, SMOTE_cat_pred))


# ### Logloss - SMOTE 처리 후 , 기존 0.5837184 ->  0.2294976 로 감소

# In[84]:


from sklearn.metrics import log_loss

SMOTE_cat_proba = SMOTE_cat.predict_proba(X_SMOTE_valid)
log_score = log_loss(y_SMOTE_valid, SMOTE_cat_proba)
log_score


# ### Roc_Auc 

# In[106]:


from sklearn.metrics import roc_auc_score

pred = SMOTE_cat.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# In[76]:


def plot_feature_importance(importance,names,model_type):
    
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    plt.figure(figsize=(10,8))

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.title(model_type + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')


# In[79]:


# 피쳐 중요도 확인 
plot_feature_importance(SMOTE_cat.get_feature_importance(),X_SMOTE_valid.columns,'CATBOOST')


# ## LightGBM

# In[80]:


lgbm = LGBMClassifier(objective = 'multiclass',metric = 'multi_logloss', ) 

lgbm.fit(X_SMOTE_train, y_SMOTE_train)


# In[81]:


lgbm_SMOTE_pred = lgbm.predict(X_SMOTE_valid)
accuracy_score(y_SMOTE_valid, lgbm_SMOTE_pred)


# In[83]:


from sklearn.metrics import classification_report

print(classification_report(y_SMOTE_valid, lgbm_SMOTE_pred))


# In[85]:


from sklearn.metrics import log_loss

SMOTE_lgbm_proba = lgbm.predict_proba(X_SMOTE_valid)
log_score = log_loss(y_SMOTE_valid, SMOTE_lgbm_proba)
log_score


# In[108]:


from sklearn.metrics import roc_auc_score

pred = lgbm.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# In[115]:


# 모델저장 
import pickle
pickle.dump(SMOTE_cat, open("model_cat_boost.pkl", "wb"))


# ## LightGMB 랜덤 서치 적용

# In[89]:


# LightGBM

param_grid = {
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(30, 150)),
    'learning_rate': [0.01,0.1,0.5],
    'subsample_for_bin': [20000,50000,100000,120000,150000],
    'min_child_samples': [20,50,100,200,500],
    'colsample_bytree': [0.6,0.8,1],
    "max_depth": [5,10,50,100]
}


# In[ ]:


# RDSearch lgbm
from sklearn.model_selection import RandomizedSearchCV
lgbm_cv = RandomizedSearchCV(lgbm, param_grid, cv=5, n_jobs=-1, verbose=2) 
lgbm_cv.fit(X_SMOTE_train, y_SMOTE_train) 
lgbm_cv.best_params_ 


# In[90]:


lgbm_tuned = LGBMClassifier(boosting_type = 'gbdt',
                            subsample_for_bin= 100000,
                            num_leaves = 57,
                            min_child_samples= 200,
                            max_depth = 10,
                            learning_rate = 0.1,
                            colsample_bytree= 1,
                            objective = 'multiclass',
                            metric = 'multi_logloss')

lgbm_tuned.fit(X_SMOTE_train, y_SMOTE_train) 


# ### Accuracy   -  SMOTE와 튜닝 후 , 기존 0.93285 -> 0.01 % 정도 감소 

# In[91]:


lgbm_tuned_pred = lgbm_tuned.predict(X_SMOTE_valid)
accuracy_score(y_SMOTE_valid, lgbm_tuned_pred)


# ### classification_report  - SMOTE와 튜닝 후 , 정확도, 재현율, F1-Score에서 유의미한 증가가 보임 

# In[94]:


from sklearn.metrics import classification_report


print(classification_report(y_SMOTE_valid, lgbm_tuned_pred))


# ### Logloss - SMOTE와 튜닝 후 , 기존 0.349955 ->  0.330446 로 감소

# In[95]:


lgbm_tuned_proba = lgbm_tuned.predict_proba(X_SMOTE_valid)
log_score = log_loss(y_SMOTE_valid, preds_l)
log_score


# In[109]:


from sklearn.metrics import roc_auc_score

pred = lgbm_tuned.predict_proba(X_valid)
roc_auc_score(y_valid, pred, multi_class= 'ovr')


# In[100]:


# 모델저장 
import pickle
pickle.dump(lgbm_tuned, open("model_lightgbm.pkl", "wb"))


# ## 테스트 데이터 예측  - X_t 

# In[96]:


# catboost 

cat_preds = SMOTE_cat.predict(X_t)
cat_preds


# In[99]:


# LightGBM
lgbm_preds = lgbm_tuned.predict(X_t)
lgbm_preds

