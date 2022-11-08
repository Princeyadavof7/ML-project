#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essential packages
import pandas as pd
import numpy as np

# Plot packages
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots

# Pre-processing packages
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Packages for handling missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[3]:


# Packages for Machine Learning
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,recall_score,precision_score
import lazypredict
from lazypredict.Supervised import LazyClassifier


# Models 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Model optimization package
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice


# In[4]:


data = pd.read_csv("bank.csv",sep = ',')
data.head()


# In[5]:


data.info()


# In[6]:


miss_val = data.isnull().any().sum()
miss_val


# In[7]:


data_categorical = data.select_dtypes(include='object')
data_numeric = data.select_dtypes(include='int64')
col_cat = data_categorical.columns 
col_num = data_numeric.columns


# In[8]:


data_categorical = data.select_dtypes(include='object')
data_numeric = data.select_dtypes(include='int64')
col_cat = data_categorical.columns 
col_num = data_numeric.columns


# In[9]:


n = len(col_cat)
fig, ax = plt.subplots(round(n/2), 2, figsize = (30,50), sharey = False)
axes = ax.flatten()

for i in range(0,n):
    sns.countplot(data = data, x = col_cat[i], ax = axes[i])


# In[10]:


data['poutcome'] = data['poutcome'].replace({'unknown':np.NaN, 'other':np.NaN})
data['contact'] = data['contact'].replace('unknown',np.NaN)


# In[11]:


n = len(col_num)

fig, axes = plt.subplots(n, 2, figsize=(15*2,10*n), sharey=False)
for i in range(0,n):
    sns.histplot(data=data,x=col_num[i],bins=20, kde = True, ax=axes[i,0])
    axes[i,0].set_title(col_num[i])
    
    sns.boxplot(x = data[col_num[i]], ax = axes[i,1])
    axes[i,1].set_title(col_num[i])


# In[12]:


data[col_num].describe()


# In[13]:


px.box(data, x="deposit", y ="balance", width=800, height=600)


# In[14]:


data['balance_range'] = pd.cut(data['balance'], bins=[5000,10000,20000,40000, np.inf], labels=['5000_to_10000','10000_to_20000','20000_to_40000','over_40000']) 
aux = data.groupby(['balance_range','deposit']).count()
aux['value'] = round(aux.iloc[:,0]*100/data.shape[0],2).to_frame()
aux.reset_index(inplace=True)
fig = px.bar(aux, x='balance_range', y = 'value', color='deposit', height=800,width=800,text = 'value')
fig.update_traces(textfont_size=20, textangle=0, textposition="outside")
fig.update_layout(title_text = "Distribution of balance", 
                       yaxis = dict(title='Percentage of data',titlefont_size=16,tickfont_size=14),
                       xaxis = dict(title='balance',titlefont_size=16,tickfont_size=14))
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[15]:


fig = px.box(data, x="deposit", y ="duration", width=1200, height=600)
fig.show()


# In[16]:


data.drop(data[data.balance>20000].index, inplace=True)
data.reset_index(inplace = True, drop =True)
data.drop('balance_range', axis=1, inplace = True)
data


# In[17]:


sns.pairplot(data, hue = 'deposit', dropna= True)


# In[18]:


n = col_cat.shape[0]-1
fig, axes = plt.subplots(n, figsize=(15,10*n), sharey=False)

for i in range(0,n):
    sns.countplot(data=data, x=col_cat[i], hue='deposit',ax = axes[i]) 
    axes[i].set_title(col_cat[i])


# In[19]:


def test_miss_validation(dataframe, miss_name_col,corr_name_col):
    rows = []
    for i in miss_name_col:
        cols = []
        for j in corr_name_col:
            miss_val = dataframe[dataframe[i]==False].groupby(j)[i].count()
            true_val = dataframe[dataframe[i]==True].groupby(j)[i].count()
            
            # Checking if the two lists have the same number of categories
            len_1 = len(true_val)
            len_2 = len(miss_val)
            if len_1 <len_2:
                mux = tuple(miss_val.index)
                true_val = true_val.reindex(mux, fill_value=0)

            if len_1 >=len_2:
                mux = tuple(true_val.index)
                miss_val = miss_val.reindex(mux, fill_value=0)
            
            
            table_validation = np.transpose([true_val.values, miss_val.values])
            p_value = chi2_contingency(table_validation)[1]
            cols.append(round(p_value,6))
        
        rows.append(cols)
    
    t_results = np.array(rows)
    df = pd.DataFrame(t_results, columns = corr_name_col, index =miss_name_col)
    return df


# In[20]:


miss_names = ['poutcome_miss','contact_miss']
cat_test = col_cat.to_list()
cat_test.remove('poutcome')
cat_test.remove('contact')

data['poutcome_miss'] = data['poutcome']
data['poutcome_miss'] = False
data.loc[data[data['poutcome'].isnull()].index, "poutcome_miss"] = True 
##

data['contact_miss'] = data['contact']
data['contact_miss'] = False
data.loc[data[data['contact'].isnull()].index, "contact_miss"] = True


# In[21]:


result_p = test_miss_validation(data, miss_names, cat_test)
result_p


# In[22]:


data[col_cat].head()


# In[23]:


data_encoded = data.copy()
data_encoded.drop(columns = ['poutcome_miss','contact_miss'],inplace=True)
data_encoded = pd.get_dummies(data_encoded, columns = ['job', 'marital'])
data_encoded.head()


# In[24]:


data_encoded['education'] = data_encoded['education'].astype('category')
data_encoded['education']=data_encoded['education'].cat.codes
data_encoded['education']


# In[25]:


label_encoder = LabelEncoder()
data_encoded['month'] = label_encoder.fit_transform(data_encoded['month'])+1 # usando label encoder ele começa a transformação usando 0 não o 1

# Transformando a variável 'month' em sin e cos
max_val = data_encoded['month'].max()
data_encoded['month_sin'] = np.sin(2 * np.pi *data_encoded['month']/max_val)
data_encoded['month_cos'] = np.cos(2 * np.pi * data_encoded['month']/max_val)

# Transformando a variável 'day' em sin e cos
max_val = data_encoded['day'].max()
data_encoded['day_sin'] = np.sin(2 * np.pi *data_encoded['day']/max_val)
data_encoded['day_cos'] = np.cos(2 * np.pi * data_encoded['day']/max_val)

data_encoded.drop(columns = ['day','month'],inplace=True)

ax = data_encoded.plot.scatter('month_sin', 'month_cos').set_aspect('equal')


# In[26]:


name_cat =col_cat.to_list()
name_cat.remove('education')
name_cat.remove('job')
name_cat.remove('marital')
name_cat.remove('month')

data_encoded['poutcome'] = data_encoded['poutcome'].replace({np.NaN:'unknown'})
data_encoded['contact'] = data_encoded['contact'].replace({np.NaN:'unknown'})

label_encoder = LabelEncoder()

for i in name_cat:
    data_encoded[i] = label_encoder.fit_transform(data_encoded[i])
    
data_encoded['poutcome'].value_counts()


# In[27]:


data_encoded['poutcome'] = data_encoded['poutcome'].replace(2,np.NaN)
data_encoded['contact'] = data_encoded['contact'].replace(2,np.NaN)


# In[28]:


scaler = MinMaxScaler()
data_scaler = pd.DataFrame(scaler.fit_transform(data_encoded), columns = data_encoded.columns)
data_scaler.drop(columns = ['deposit'],inplace=True)
data_scaler.head()


# In[29]:


imputer = IterativeImputer(imputation_order='ascending',max_iter=1000,random_state=0,n_nearest_features=None)

data_end = pd.DataFrame(imputer.fit_transform(data_scaler),columns = data_scaler.columns)

data_end['poutcome'] = data_end['poutcome'].round(decimals=0)
data_end['contact'] = data_end['contact'].round(decimals=0)
data_end['poutcome'].value_counts()


# In[30]:


data_end['contact'].value_counts()


# In[31]:


data_end['deposit'] = data_encoded['deposit']
fig, axes = plt.subplots(1, figsize=(20,20))
pearson_matrix = data_end.corr(method='pearson')
fig = sns.heatmap(pearson_matrix, annot=True,linewidths=.1)


# In[32]:


axes = plt.subplots(1, figsize=(20,20))
spearman_matrix = data_end.corr(method='spearman')
fig = sns.heatmap(spearman_matrix, annot=True,linewidths=.1)


# In[33]:


data_end.drop(columns=['pdays','marital_single','deposit'],inplace = True)


# In[34]:


X = data_end
y= data_encoded['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =0)


# In[35]:


lazy_model = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=precision_score)
models,predictions = lazy_model.fit(X_train, X_test, y_train, y_test)
models


# In[36]:


model_RF = Pipeline([('RF', RandomForestClassifier(random_state=0))])


# In[37]:


def Objective_RF(trial):
    RF__max_depth = trial.suggest_int('RF__max_depth',200,800)
    RF__n_estimators = trial.suggest_int('RF__n_estimators',100,600)
    RF__min_samples_split = trial.suggest_int('RF__min_samples_split',2,20)
    RF__min_samples_leaf = trial.suggest_int('RF__min_samples_leaf',1,10)    
    RF__max_leaf_nodes = trial.suggest_int('RF__max_leaf_nodes',300,800)
    
    params = {
        'RF__max_depth': RF__max_depth,
        'RF__n_estimators':  RF__n_estimators,
        'RF__min_samples_split' : RF__min_samples_split,
        'RF__min_samples_leaf' : RF__min_samples_leaf,
        'RF__max_leaf_nodes' : RF__max_leaf_nodes
    }
    
    
    model = model_RF.set_params(**params)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    pre = precision_score(y_test,prediction)
    
    
    return pre


# In[38]:


study_RF = optuna.create_study(directions = ["maximize"])
study_RF.optimize(Objective_RF, n_trials = 200, show_progress_bar = True)


# In[39]:


plot_optimization_history(study_RF)


# In[40]:


plot_slice(study_RF)


# In[41]:


study_RF.best_trial.values


# In[42]:


model = model_RF.set_params(**study_RF.best_params)
model.fit(X_train,y_train)
prediction = model.predict(X_test)


matrix=confusion_matrix(y_test,prediction)
df_cm = pd.DataFrame(matrix,index=['0','1'],columns=['0','1'])
s = sns.heatmap(df_cm, annot= True,fmt='g')
s.set(xlabel='Target', ylabel='Predicted')

report=classification_report(y_test,prediction)
print(report)


# In[43]:


model_XGB = Pipeline([('XGB', XGBClassifier(random_state=0,eval_metric = 'auc') )])


# In[44]:


def Objective_XGB(trial):
    n_estimators = trial.suggest_int('XGB__n_estimators',50,300)
    max_depth = trial.suggest_int('XGB__max_depth',10,400)
    max_leaves = trial.suggest_int('XGB__max_leaves',0,100)
    learning_rate = trial.suggest_float('XGB__learning_rate',0.0001,1)
    min_child_weight = trial.suggest_float('XGB__min_child_weight',0.001,10)
    max_delta_step = trial.suggest_float('XGB__max_delta_step',0.0001,1)
    reg_alpha = trial.suggest_float('XGB__reg_alpha',0,20)
    
    params = {
        'XGB__n_estimators' : n_estimators,
        'XGB__max_depth' : max_depth,
        'XGB__max_leaves' : max_leaves,
        'XGB__learning_rate' : learning_rate,
        'XGB__min_child_weight' : min_child_weight,
        'XGB__max_delta_step' : max_delta_step,
        'XGB__reg_alpha' : reg_alpha
    }
    
    model = model_XGB.set_params(**params)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    pre = precision_score(y_test,prediction)
    
    
    return pre  


# In[45]:


study_XGB = optuna.create_study(directions = ["maximize"])
study_XGB.optimize(Objective_XGB, n_trials = 200, show_progress_bar = True)


# In[46]:


plot_optimization_history(study_XGB)


# In[47]:


plot_slice(study_XGB)


# In[48]:


study_XGB.best_trial.values


# In[49]:


model = model_XGB.set_params(**study_XGB.best_params)

model.fit(X_train,y_train)
prediction = model.predict(X_test)

matrix=confusion_matrix(y_test,prediction)
df_cm = pd.DataFrame(matrix,index=['0','1'],columns=['0','1'])
s = sns.heatmap(df_cm, annot= True,fmt='g')
s.set(xlabel='Target', ylabel='Predicted')
report=classification_report(y_test,prediction)
print(report)


# In[50]:


model_LGBM = Pipeline([('LGBM', LGBMClassifier(random_state=0, metric = "logloss"))])


# In[51]:


def Objective_LGBM(trial):
    
    LGBM__boosting_type = trial.suggest_categorical('LGBM__boosting_type',['gbdt','dart','goss']) #pode dar erro por ter modelos diferentes
    LGBM__num_leaves = trial.suggest_int('LGBM__num_leaves',10,100)
    LGBM__max_depth = trial.suggest_int('LGBM__max_depth',-1,99,step=10)
    LGBM__learning_rate = trial.suggest_float('LGBM__learning_rate',0.001,1)
    LGBM__num_iterations = trial.suggest_int('LGBM__num_iterations',50,500,step = 50)
    LGBM__reg_alpha = trial.suggest_float('LGBM__reg_alpha',0,10)
    LGBM__reg_lambda = trial.suggest_float('LGBM__reg_lambda',0,10)
    
    
    params = {
        'LGBM__boosting_type' : LGBM__boosting_type,
        'LGBM__num_leaves' : LGBM__num_leaves,
        'LGBM__max_depth' : LGBM__max_depth,
        'LGBM__learning_rate' : LGBM__learning_rate,
        'LGBM__num_iterations' : LGBM__num_iterations,
        'LGBM__reg_alpha' : LGBM__reg_alpha,
        'LGBM__reg_lambda' : LGBM__reg_lambda, 
    }
    model = model_LGBM.set_params(**params)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    pre = precision_score(y_test,prediction)
    
    
    return pre


# In[52]:


study_LGBM = optuna.create_study(directions = ["maximize"])
study_LGBM.optimize(Objective_LGBM, n_trials = 200, show_progress_bar = True)


# In[53]:


plot_optimization_history(study_LGBM)


# In[54]:


plot_slice(study_LGBM)


# In[55]:


study_LGBM.best_trial.values


# In[56]:


model = model_LGBM.set_params(**study_LGBM.best_params)
model.fit(X_train,y_train)
prediction = model.predict(X_test)

# Verificando a precisão do modelo
matrix=confusion_matrix(y_test,prediction)
df_cm = pd.DataFrame(matrix,index=['0','1'],columns=['0','1'])
s = sns.heatmap(df_cm, annot= True,fmt='g')
s.set(xlabel='Target', ylabel='Predicted')

report=classification_report(y_test,prediction)
print(report)


# In[57]:


study_XGB.best_params


# In[58]:


from xgboost import plot_importance
model_final = XGBClassifier(random_state =0, learning_rate = 0.698720134632314, n_estimators = 231, max_depth = 343, max_leaves = 67, min_child_weight = 0.8341364234312232,
                           max_delta_step = 0.07651183216848476, reg_alpha = 7.249554627471188)

model_final.fit(X_train,y_train)
fig, ax = plt.subplots(figsize=(50,10))
plt.bar(range(len(model_final.feature_importances_)), model_final.feature_importances_,tick_label = data_end.columns)


# In[ ]:




