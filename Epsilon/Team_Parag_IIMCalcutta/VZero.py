#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np


# In[37]:


data = pd.read_csv('Train.csv')
submit = pd.read_csv('Test.csv')


# In[38]:


data.head()


# In[39]:


data.isnull().sum()


# In[40]:


data.dtypes


# **Data Merge**

# In[41]:


submit['Default'] = None
all_data = data.append(submit, ignore_index = True)
print(len(data), len(submit), len(all_data))


# **Preprocessing**

# In[42]:


all_data['Default'] = all_data['Default'].apply(lambda x: 0 if x =='No' else x)
all_data['Default'] = all_data['Default'].apply(lambda x: 1 if x =='Yes' else x)


# In[43]:


all_data['Term'].fillna(np.mean(all_data.Term), inplace = True)
all_data['Credit_score'].fillna(np.mean(all_data.Credit_score), inplace = True)
all_data['Amount'].fillna(np.mean(all_data.Amount), inplace = True)
all_data['Checking_amount'].fillna(np.mean(all_data.Checking_amount), inplace = True)
all_data['Saving_amount'].fillna(np.mean(all_data.Saving_amount), inplace = True)
all_data['Emp_duration '].fillna(np.mean(all_data['Emp_duration '] ), inplace = True)
all_data['Marital_status '] = all_data['Marital_status '].fillna(all_data['Marital_status '].value_counts().index[0])
all_data['Car_loan'] = all_data['Car_loan'].fillna(all_data['Car_loan'].value_counts().index[0])
all_data['Personal_loan'] = all_data['Personal_loan'].fillna(all_data['Personal_loan'].value_counts().index[0])
all_data['Home_loan'] = all_data['Home_loan'].fillna(all_data['Home_loan'].value_counts().index[0])
all_data['Education_loan'] = all_data['Education_loan'].fillna(all_data['Education_loan'].value_counts().index[0])
all_data['No_of_credit_acc'] = all_data['No_of_credit_acc'].fillna(all_data['No_of_credit_acc'].value_counts().index[0])


# In[44]:


all_data.isnull().sum()


# In[45]:


all_data.head()


# In[46]:


all_data['Car_loan'] = [0 if x == 'No' else 1 for x in all_data['Car_loan']]
all_data['Education_loan'] = [0 if x == 'No' else 1 for x in all_data['Education_loan']]
all_data['Home_loan'] = [0 if x == 'No' else 1 for x in all_data['Home_loan']]
all_data['Personal_loan'] = [0 if x == 'No' else 1 for x in all_data['Personal_loan']]
all_data['Marital_status '] = [0 if x == 'Single' else 1 for x in all_data['Marital_status ']]
all_data['Emp_status'] = [0 if x == 'unemployed' else 1 for x in all_data['Emp_status']]
all_data['Gender'] = [0 if x == 'Female' else 1 for x in all_data['Gender']]


# In[47]:


all_data.dtypes


# **Seperating Data**

# In[48]:


df_train=all_data[all_data['Default'].isnull()==False].copy()
df_test=all_data[all_data['Default'].isnull()==True].copy()

print(df_train.shape,df_test.shape)


# In[49]:


X,y=df_train.drop(['ID','Default'],axis=1),df_train['Default']
Xtest=df_test.drop(['ID','Default'],axis=1)


# **TrainTestSplit**

# In[50]:


from sklearn.model_selection import StratifiedKFold,train_test_split
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.15,random_state = 1996,stratify=y)


# **LDA**

# In[51]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 5)
X_new = lda.fit_transform(X_train,y_train)


# In[52]:


new = pd.DataFrame(X_new)
new['Label'] = y_train
print(np.min(new[new['Label']==1]),np.max(new[new['Label']==1]))
print(np.min(new[new['Label']==0]),np.max(new[new['Label']==0]))


# In[53]:


import matplotlib.pyplot as plt 
plt.xlabel('LD1')
#plt.ylabel('LD2')
plt.scatter(
    X_new[:,0],
    X_new[:,0],
    c=y_train,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='w'
)


# In[54]:


lda.explained_variance_ratio_


# In[55]:


lda_out = lda.predict(X_val)
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
accuracy = accuracy_score(lda_out, y_val)
confusion_matrix = confusion_matrix(lda_out, y_val)
print(classification_report(lda_out,y_val))
print(accuracy)
print(confusion_matrix)


# **Models**

# **Random Forest**

# In[56]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 350)
clf.fit(X_train,y_train)
sel = SelectFromModel(clf)
sel.fit(X_train, y_train)
selected_feat= X_train.columns[(sel.get_support())]
clf.fit(X_train[selected_feat], y_train)
predict = clf.predict(X_val[selected_feat])


# In[57]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(predict, y_val)
confusion_matrix = confusion_matrix(predict, y_val)
print(classification_report(predict,y_val))
print(accuracy)
print(confusion_matrix)


# In[58]:


selected_feat


# **AllFeatures**

# In[59]:


clf = RandomForestClassifier(n_estimators = 350)
clf.fit(X_train,y_train)
predict = clf.predict(X_val)


# In[60]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(predict, y_val)
confusion_matrix = confusion_matrix(predict, y_val)
print(classification_report(predict,y_val))
print(confusion_matrix)


# 
# **Xgboost**

# In[61]:


from xgboost import XGBClassifier
xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.8, gamma=0.4,
       learning_rate=0.3, max_delta_step=0, max_depth=10,
       min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
       nthread=None, random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgb1.fit(X_train, y_train)
predict = xgb1.predict(X_val)


# In[62]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(predict, y_val)
confusion_matrix = confusion_matrix(predict, y_val)
print(accuracy)
print(classification_report(predict,y_val))
print(confusion_matrix)


# **SVM**

# In[63]:


from sklearn import svm


# In[64]:


clf = svm.SVC(kernel='linear')
clf.fit(X_train[selected_feat],y_train)
predict = clf.predict(X_val[selected_feat])


# In[65]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(predict, y_val)
confusion_matrix = confusion_matrix(predict, y_val)
print(classification_report(predict, y_val))
print(confusion_matrix)


# **TrainingOnFullData**

# In[31]:


''''
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 350)
clf.fit(X,y)
sel = SelectFromModel(clf)
sel.fit(X, y)
selected_feat= X.columns[(sel.get_support())]
clf.fit(X[selected_feat], y)
predict = clf.predict(Xtest[selected_feat])
''''''


# In[66]:


lda = LinearDiscriminantAnalysis(n_components = 5)
X_new = lda.fit_transform(X,y)


# In[67]:


predict = lda.predict(Xtest)


# **Submit_csv**

# In[68]:


sub = pd.DataFrame(df_test['ID'])
sub['Default'] = predict
sub['Default'] = ['Yes' if x == 1 else 'No' for x in sub['Default']]
sub.to_csv('submit.csv', index=False)


# In[ ]:




