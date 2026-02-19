#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


titanic_data=pd.read_csv("train.csv")
titanic_data_test=pd.read_csv("test.csv")


# In[3]:


titanic_data


# In[4]:


titanic_data.head()


# In[5]:


titanic_data.describe()


# In[6]:


titanic_data.describe(include=['O'])


# In[7]:


g_values=titanic_data.loc[:,["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"]]


# In[8]:


sns.heatmap(g_values.corr(), cmap="YlGnBu")
plt.show()


# In[9]:


sns.lineplot(g_values.corr())
plt.show()


# In[10]:


sns.histplot(g_values.corr())
plt.show()


# In[11]:


sns.scatterplot(g_values.corr())
plt.show()


# In[12]:


pd.DataFrame(titanic_data.groupby(['Pclass'])['Survived'].mean())


# In[13]:


pd.DataFrame(titanic_data.groupby(['Sex'])['Survived'].mean())


# In[14]:


pd.DataFrame(titanic_data.groupby(['SibSp'])['Survived'].mean())


# In[15]:


pd.DataFrame(titanic_data.groupby(['Parch'])['Survived'].mean())


# In[16]:


titanic_data['Family_Size']=titanic_data['Parch']+titanic_data['SibSp']+1 # 1 include the person itself too
titanic_data_test['Family_Size']=titanic_data['Parch']+titanic_data['SibSp']+1


# In[17]:


pd.DataFrame(titanic_data.groupby(['Family_Size'])['Survived'].mean())


# In[18]:


family_map={1:'Alone',2:'Pair',3:'Trio',4:'Very Small',5:'Small',6:'Medium',7:'Medium Large',8:'Large',11:'Very Large'}
titanic_data['Family_Size_Grouped']=titanic_data['Family_Size'].map(family_map)
titanic_data_test['Family_Size_Grouped']=titanic_data_test['Family_Size'].map(family_map)


# In[19]:


pd.DataFrame(titanic_data.groupby(['Family_Size_Grouped'])['Survived'].mean())


# In[20]:


pd.DataFrame(titanic_data.groupby(['Embarked'])['Survived'].mean())


# In[21]:


sns.displot(titanic_data,x='Age',col='Survived',binwidth=10,height=5)


# In[22]:


titanic_data['Age_Cut']=pd.qcut(titanic_data['Age'],8)
titanic_data_test['Age_Cut']=pd.qcut(titanic_data_test['Age'],8)


# In[23]:


pd.DataFrame(titanic_data.groupby(['Age_Cut'],observed=True)['Survived'].mean())


# In[24]:


titanic_data.loc[titanic_data['Age'] <= 16, 'Age'] = 0
titanic_data.loc[(titanic_data['Age'] > 16)&(titanic_data['Age'] <= 20.125), 'Age'] = 1
titanic_data.loc[(titanic_data['Age'] > 20.125)&(titanic_data['Age'] <= 24.0), 'Age'] = 2
titanic_data.loc[(titanic_data['Age'] > 24.0)&(titanic_data['Age'] <= 28.0), 'Age'] = 3
titanic_data.loc[(titanic_data['Age'] > 28.0)&(titanic_data['Age'] <= 32.312), 'Age'] = 4
titanic_data.loc[(titanic_data['Age'] > 32.312)&(titanic_data['Age'] <= 38.0), 'Age'] = 5
titanic_data.loc[(titanic_data['Age'] > 38.0)&(titanic_data['Age'] <= 47.0), 'Age'] = 6
titanic_data.loc[(titanic_data['Age'] > 47.0)&(titanic_data['Age'] <= 80.0), 'Age'] = 7
titanic_data.loc[titanic_data['Age'] >80, 'Age'] = 8

titanic_data_test.loc[titanic_data_test['Age'] <= 16, 'Age'] = 0
titanic_data_test.loc[(titanic_data_test['Age'] > 16)&(titanic_data_test['Age'] <= 20.125), 'Age'] = 1
titanic_data_test.loc[(titanic_data_test['Age'] > 20.125)&(titanic_data_test['Age'] <= 24.0), 'Age'] = 2
titanic_data_test.loc[(titanic_data_test['Age'] > 24.0)&(titanic_data_test['Age'] <= 28.0), 'Age'] = 3
titanic_data_test.loc[(titanic_data_test['Age'] > 28.0)&(titanic_data_test['Age'] <= 32.312), 'Age'] = 4
titanic_data_test.loc[(titanic_data_test['Age'] > 32.312)&(titanic_data_test['Age'] <= 38.0), 'Age'] = 5
titanic_data_test.loc[(titanic_data_test['Age'] > 38.0)&(titanic_data_test['Age'] <= 47.0), 'Age'] = 6
titanic_data_test.loc[(titanic_data_test['Age'] > 47.0)&(titanic_data_test['Age'] <= 80.0), 'Age'] = 7
titanic_data_test.loc[titanic_data_test['Age'] >80, 'Age'] = 8


# In[25]:


titanic_data.head()


# In[26]:


sns.displot(titanic_data,x='Fare',col='Survived',binwidth=75,height=5)


# In[27]:


titanic_data['Fare_Cut']=pd.qcut(titanic_data['Fare'],6)
titanic_data_test['Fare_Cut']=pd.qcut(titanic_data_test['Fare'],6)


# In[28]:


pd.DataFrame(titanic_data.groupby(['Fare_Cut'],observed=True)['Survived'].mean())


# In[29]:


titanic_data.loc[titanic_data['Fare'] <= 7.775, 'Fare'] = 0
titanic_data.loc[(titanic_data['Fare'] > 7.775)&(titanic_data['Fare'] <= 8.662), 'Fare'] = 1
titanic_data.loc[(titanic_data['Fare'] > 8.662)&(titanic_data['Fare'] <= 14.454), 'Fare'] = 2
titanic_data.loc[(titanic_data['Fare'] > 14.454)&(titanic_data['Fare'] <= 26.0), 'Fare'] = 3
titanic_data.loc[(titanic_data['Fare'] > 26.0)&(titanic_data['Fare'] <= 52.369), 'Fare'] = 4
titanic_data.loc[(titanic_data['Fare'] > 52.369)&(titanic_data['Fare'] <=512.329), 'Fare'] = 5
titanic_data.loc[titanic_data['Fare'] >512.329, 'Fare'] = 6

titanic_data_test.loc[titanic_data_test['Fare'] <= 7.775, 'Fare'] = 0
titanic_data_test.loc[(titanic_data_test['Fare'] > 7.775)&(titanic_data_test['Fare'] <= 8.662), 'Fare'] = 1
titanic_data_test.loc[(titanic_data_test['Fare'] > 8.662)&(titanic_data_test['Fare'] <= 14.454), 'Fare'] = 2
titanic_data_test.loc[(titanic_data_test['Fare'] > 14.454)&(titanic_data_test['Fare'] <= 26.0), 'Fare'] = 3
titanic_data_test.loc[(titanic_data_test['Fare'] > 26.0)&(titanic_data_test['Fare'] <= 52.369), 'Fare'] = 4
titanic_data_test.loc[(titanic_data_test['Fare'] > 52.369)&(titanic_data_test['Fare'] <=512.329), 'Fare'] = 5
titanic_data_test.loc[titanic_data_test['Fare'] >512.329, 'Fare'] = 6


# In[30]:


titanic_data["Name"].str.split(pat=",",expand=True)


# In[31]:


titanic_data["Name"].str.split(pat=",",expand=True)[1].str.split(pat=".",expand=True)


# In[32]:


titanic_data["Name"].str.split(pat=",",expand=True)[1].str.split(pat=".",expand=True)[0]


# In[33]:


titanic_data_test["Title"]=titanic_data_test["Name"].str.split(pat=",",expand=True)[1].str.split(pat=".",expand=True)[0].apply(lambda x : x.strip())                                                                                                                    



# In[34]:


titanic_data["Title"]=titanic_data["Name"].str.split(pat=",",expand=True)[1].str.split(pat=".",expand=True)[0].apply(lambda x: x.strip())


# In[35]:


pd.DataFrame(titanic_data.groupby(['Title'],observed=True)['Survived'].mean())


# In[36]:


'''
->Military Ranks: Capt (Captain), Col (Colonel), Major
->Academic/Professional Titles:- Don, Dr (Doctor), Rev (Reverend)
->Nobility/Formal Honorifics:- Jonkheer, Lady, Sir, the Countess
->General Honorifics:- Master, Miss, Mlle (Mademoiselle), Mme (Madame), Mr (Mister), Mrs (Missus), Ms'''
'''
->French Title:- Mlle (Mademoiselle), Mme (Madame) 
->Dutch Titles:- Jonkheer
->Spanish/Italian Titles:- Don'''


# In[37]:


title={"Adult men" :['Mr'],"Unmarried women and girls":["Miss, Ms, Mlle"],"Married women":["Mrs, Mme"],"Young boys":['Master'],"Professional or military rank":["Capt, Col, Major, Dr, Rev"],"High-status nobility":["Jonkheer, Don, Sir, Lady, the Countess"]}


# In[38]:


pd.DataFrame(title)


# In[39]:


titanic_data["Title"]=titanic_data["Title"].replace({
  "Mr": "Adult men",
  "Miss": "Unmarried women and girls",
  "Ms": "Unmarried women and girls",
  "Mlle": "Unmarried women and girls",
  "Mrs": "Married women",
  "Mme": "Married women",
  "Master": "Young boys",
  "Capt": "Military",
  "Col": "Military",
  "Major": "Military",
  "Dr": "Professional",
  "Rev": "Professional",
  "Jonkheer": "High-status nobility",
  "Don": "High-status nobility",
  "Sir": "High-status nobility",
  "Lady": "High-status nobility",
  "the Countess": "High-status nobility"
})
titanic_data_test["Title"]=titanic_data_test["Title"].replace({
  "Mr": "Adult men",
  "Miss": "Unmarried women and girls",
  "Ms": "Unmarried women and girls",
  "Mlle": "Unmarried women and girls",
  "Mrs": "Married women",
  "Mme": "Married women",
  "Master": "Young boys",
  "Capt": "Military",
  "Col": "Military",
  "Major": "Military",
  "Dr": "Professional",
  "Rev": "Professional",
  "Jonkheer": "High-status nobility",
  "Don": "High-status nobility",
  "Sir": "High-status nobility",
  "Lady": "High-status nobility",
  "the Countess": "High-status nobility"
})


# In[40]:


pd.DataFrame(titanic_data.groupby(['Title'],observed=True)['Survived'].agg(['count','mean']))


# In[41]:


titanic_data["Name_Length"]=titanic_data["Name"].apply(lambda x:len(x))
titanic_data_test["Name_Length"]=titanic_data_test["Name"].apply(lambda x:len(x))


# In[42]:


sns.kdeplot(titanic_data["Name_Length"][(titanic_data["Survived"]==0)&(titanic_data["Name"].notnull())],color="Pink",fill=True)


# In[43]:


sns.kdeplot(titanic_data["Name_Length"][(titanic_data["Survived"]==1)&(titanic_data["Name"].notnull())],color="Blue",fill=True)


# In[44]:


g=sns.kdeplot(titanic_data["Name_Length"][(titanic_data["Survived"]==0)&(titanic_data["Name"].notnull())],color="Pink",fill=True)
g=sns.kdeplot(titanic_data["Name_Length"][(titanic_data["Survived"]==1)&(titanic_data["Name"].notnull())],ax=g,color="Blue",fill=True)
g.set_ylabel("Frequency")
g=g.legend(['Not Survived','Survived'])


# In[45]:


titanic_data['Name_LengthGB']=pd.qcut(titanic_data['Name_Length'],8)
titanic_data_test['Name_LengthGB']=pd.qcut(titanic_data_test['Name_Length'],8)


# In[46]:


pd.DataFrame(titanic_data.groupby(['Name_LengthGB'],observed=True)['Survived'].mean())


# In[47]:


titanic_data.loc[titanic_data['Name_Length'] <= 18.0, 'Name_Length'] = 0
titanic_data.loc[(titanic_data['Name_Length'] > 18.0) & (titanic_data['Name_Length'] <= 20.0), 'Name_Length'] = 1
titanic_data.loc[(titanic_data['Name_Length'] > 20.0) & (titanic_data['Name_Length'] <= 23.0), 'Name_Length'] = 2
titanic_data.loc[(titanic_data['Name_Length'] > 23.0) & (titanic_data['Name_Length'] <= 25.0), 'Name_Length'] = 3
titanic_data.loc[(titanic_data['Name_Length'] > 25.0) & (titanic_data['Name_Length'] <= 27.25), 'Name_Length'] = 4
titanic_data.loc[(titanic_data['Name_Length'] > 27.25) & (titanic_data['Name_Length'] <= 30.0), 'Name_Length'] = 5
titanic_data.loc[(titanic_data['Name_Length'] > 30.0) & (titanic_data['Name_Length'] <= 38.0), 'Name_Length'] = 6
titanic_data.loc[titanic_data['Name_Length'] > 38.0, 'Name_Length'] = 7

titanic_data_test.loc[titanic_data_test['Name_Length'] <= 18.0, 'Name_Length'] = 0
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 18.0) & (titanic_data_test['Name_Length'] <= 20.0), 'Name_Length'] = 1
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 20.0) & (titanic_data_test['Name_Length'] <= 23.0), 'Name_Length'] = 2
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 23.0) & (titanic_data_test['Name_Length'] <= 25.0), 'Name_Length'] = 3
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 25.0) & (titanic_data_test['Name_Length'] <= 27.25), 'Name_Length'] = 4
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 27.25) & (titanic_data_test['Name_Length'] <= 30.0), 'Name_Length'] = 5
titanic_data_test.loc[(titanic_data_test['Name_Length'] > 30.0) & (titanic_data_test['Name_Length'] <= 38.0), 'Name_Length'] = 6
titanic_data_test.loc[titanic_data_test['Name_Length'] > 38.0, 'Name_Length'] = 7


# In[48]:


titanic_data.head()


# In[49]:


titanic_data_test.head()


# In[50]:


titanic_data["Ticket_Number"]=titanic_data["Ticket"].apply(lambda x:pd.Series({'Ticket':x.split()[-1]}))
titanic_data_test["Ticket_Number"]=titanic_data_test["Ticket"].apply(lambda x:pd.Series({'Ticket':x.split()[-1]}))


# In[51]:


pd.DataFrame(titanic_data.groupby(['Ticket_Number'],observed=True)['Survived'].agg(['count','mean']))


# In[52]:


titanic_data["Ticket_Number_Count"]=titanic_data.groupby('Ticket_Number')["Ticket_Number"].transform('count')
titanic_data_test["Ticket_Number_Count"]=titanic_data_test.groupby('Ticket_Number')["Ticket_Number"].transform('count')


# In[53]:


pd.DataFrame(titanic_data.groupby(['Ticket_Number_Count'],observed=True)['Survived'].agg(['count','mean']))


# In[54]:


titanic_data["Ticket"].str.split(pat=" ",expand=True)


# In[55]:


titanic_data["Ticket_Location"]=np.where(titanic_data["Ticket"].str.split(pat=" ",expand=True)[1].notna(),titanic_data["Ticket"].str.split(pat=" ",expand=True)[0].apply(lambda x:x.strip()),'Blank')
titanic_data_test["Ticket_Location"]=np.where(titanic_data_test["Ticket"].str.split(pat=" ",expand=True)[1].notna(),titanic_data_test["Ticket"].str.split(pat=" ",expand=True)[0].apply(lambda x:x.strip()),'Blank')


# In[56]:


titanic_data["Ticket_Location"].value_counts()


# In[57]:


titanic_data['Ticket_Location'] = titanic_data['Ticket_Location'].replace({
    'SOTON/O.Q.':'SOTON/OQ',
    'C.A.':'CA',
    'CA.':'CA',
    'SC/PARIS':'SC/Paris',
    'S.C./PARIS':'SC/Paris',
    'A/4.':'A/4',
    'A/5.':'A/5',
    'A.5.':'A/5',
    'A./5.':'A/5',
    'W./C.':'W/C',
})

titanic_data_test['Ticket_Location'] = titanic_data_test['Ticket_Location'].replace({
    'SOTON/O.Q.':'SOTON/OQ',
    'C.A.':'CA',
    'CA.':'CA',
    'SC/PARIS':'SC/Paris',
    'S.C./PARIS':'SC/Paris',
    'A/4.':'A/4',
    'A/5.':'A/5',
    'A.5.':'A/5',
    'A./5.':'A/5',
    'W./C.':'W/C',
})


# In[58]:


pd.DataFrame(titanic_data.groupby(['Ticket_Location'],observed=True)['Survived'].agg(['count','mean']))


# In[59]:


titanic_data['Cabin'] = titanic_data['Cabin'].fillna('U')
titanic_data['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'x' for i in titanic_data['Cabin']])

titanic_data_test['Cabin'] = titanic_data_test['Cabin'].fillna('U')
titanic_data_test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'x' for i in titanic_data_test['Cabin']])


# In[60]:


pd.DataFrame(titanic_data.groupby(['Cabin'],observed=True)['Survived'].agg(['count','mean']))


# In[61]:


titanic_data['Cabin_Assigned'] = titanic_data['Cabin'].apply(lambda x: 0 if x in ['U'] else 1)
titanic_data_test['Cabin_Assigned'] = titanic_data_test['Cabin'].apply(lambda x: 0 if x in ['U'] else 1)


# In[62]:


pd.DataFrame(titanic_data.groupby(['Cabin_Assigned'],observed=True)['Survived'].agg(['count','mean']))


# In[63]:


titanic_data


# In[64]:


correlation_matrix = titanic_data.corr(numeric_only=True)


# In[65]:


sns.heatmap(correlation_matrix.corr(), cmap="viridis")
plt.show()


# In[66]:


titanic_data.info()


# In[67]:


titanic_data_test.info()


# In[68]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data_test['Age'].fillna(titanic_data_test['Age'].mean(), inplace=True)
titanic_data_test['Fare'].fillna(titanic_data_test['Fare'].mean(), inplace=True)


# In[69]:


titanic_data_test.info()


# In[70]:


titanic_data_test['Age_Cut']=pd.qcut(titanic_data_test['Age'],8)

titanic_data_test.loc[titanic_data_test['Age'] <= 16, 'Age'] = 0
titanic_data_test.loc[(titanic_data_test['Age'] > 16)&(titanic_data_test['Age'] <= 20.125), 'Age'] = 1
titanic_data_test.loc[(titanic_data_test['Age'] > 20.125)&(titanic_data_test['Age'] <= 24.0), 'Age'] = 2
titanic_data_test.loc[(titanic_data_test['Age'] > 24.0)&(titanic_data_test['Age'] <= 28.0), 'Age'] = 3
titanic_data_test.loc[(titanic_data_test['Age'] > 28.0)&(titanic_data_test['Age'] <= 32.312), 'Age'] = 4
titanic_data_test.loc[(titanic_data_test['Age'] > 32.312)&(titanic_data_test['Age'] <= 38.0), 'Age'] = 5
titanic_data_test.loc[(titanic_data_test['Age'] > 38.0)&(titanic_data_test['Age'] <= 47.0), 'Age'] = 6
titanic_data_test.loc[(titanic_data_test['Age'] > 47.0)&(titanic_data_test['Age'] <= 80.0), 'Age'] = 7
titanic_data_test.loc[titanic_data_test['Age'] >80, 'Age'] = 8


# In[71]:


titanic_data_test.info()


# In[72]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier


from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV


# In[73]:


ohe = OneHotEncoder(sparse_output=False)
ode = OrdinalEncoder
SI = SimpleImputer(strategy='most_frequent')


# In[74]:


ode_cols = ['Family_Size_Grouped']
ohe_cols = ['Sex', 'Embarked']


# In[75]:


X = titanic_data.drop(['Survived', 'SibSp', 'Parch'], axis=1)
y = titanic_data['Survived']
X_test = titanic_data_test.drop(['Age_Cut', 'Fare_Cut','SibSp', 'Parch'], axis=1)


# In[76]:


from sklearn.preprocessing import StandardScaler
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)


# In[77]:


ordinal_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(handle_unknown = 'ignore', sparse_output=False))
])
col_trans = ColumnTransformer(transformers=[
    ('impute', SI, ['Age']),
    ('ord_pipeline', ordinal_pipeline, ode_cols),
    ('ohe_pipeline', ohe_pipeline, ohe_cols),
   # ('passthrough', 'passthrough', ['Pclass', 'TicketNumberCounts', 'Cabin_Assigned', 'Name_Size', 'Age', 'Fare'])
     ('passthrough', 'passthrough', ['Pclass', 'Cabin_Assigned', 'Name_Length', 'Age', 'Fare', 'Ticket_Number_Count'])
    ],
    remainder='drop',
    n_jobs=-1)


# In[78]:


rfc = RandomForestClassifier()


# In[79]:


param_grid = {
    'n_estimators': [150, 200, 300, 500],
    'min_samples_split': [5, 10, 15],
    'max_depth': [10, 13, 15, 17, 20],
    'min_samples_leaf': [2, 4, 5, 6],
    'criterion': ['gini', 'entropy'],
}
abc = AdaBoostClassifier(estimator=LogisticRegression(max_iter=1000))


# In[80]:


CV_rfc = GridSearchCV(estimator=rfc,
                      param_grid=param_grid,
                      cv=StratifiedKFold(n_splits=5),
                     verbose=0)


# In[81]:


pipefinalrfc = make_pipeline(col_trans, CV_rfc)
pipefinalrfc.fit(X_train, y_train)


# In[82]:


dtc = DecisionTreeClassifier()
param_grid = {
    'min_samples_split': [5, 10, 15],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
}
CV_dtc = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[83]:


pipefinaldtc = make_pipeline(col_trans, CV_dtc)
pipefinaldtc.fit(X_train, y_train)


# In[84]:


knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1,2],
}
CV_knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[85]:


pipefinalknn = make_pipeline(col_trans, CV_knn)
pipefinalknn.fit(X_train, y_train)


# In[86]:


svc = SVC(probability=True)
param_grid = {
    'C': [100,10, 1.0, 0.1, 0.001, 0.001],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
}
CV_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[87]:


pipefinalsvc = make_pipeline(col_trans, CV_svc)
pipefinalsvc.fit(X_train, y_train)


# In[96]:


lr = LogisticRegression(max_iter=1000)
param_grid = {
    'C': [100,10, 1.0, 0.1, 0.001, 0.001],
}
abc = AdaBoostClassifier(estimator=LogisticRegression(max_iter=1000))
CV_lr = GridSearchCV(estimator=lr,
                     param_grid=param_grid, 
                     cv=StratifiedKFold(n_splits=5),
                    verbose=0)


# In[97]:


pipefinallr= make_pipeline(col_trans, CV_lr)
pipefinallr.fit(X_train, y_train)


# In[98]:


gnb = GaussianNB()
param_grid = {
    'var_smoothing': [0.00000001, 0.000000001, 0.00000001],
}
CV_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[99]:


pipefinalgnb= make_pipeline(col_trans, CV_gnb)
pipefinalgnb.fit(X_train, y_train)


# In[100]:


xg = XGBClassifier()
param_grid = {
     'booster': ['gbtree', 'gblinear','dart'],
}
CV_xg = GridSearchCV(estimator=xg, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[101]:


pipefinalxg= make_pipeline(col_trans, CV_xg)
pipefinalxg.fit(X_train, y_train)


# In[104]:


abc = AdaBoostClassifier()
dtc_2 = DecisionTreeClassifier(criterion = 'entropy', max_depth=10,min_samples_leaf=4, min_samples_split=10)  
svc_2 = SVC(probability=True, C=10, kernel='rbf') 
lr_2 = LogisticRegression(C=0.1) 
lr_3 = LogisticRegression(C=0.2) 
lr_4 = LogisticRegression(C=0.05) 
param_grid = {
    'estimator': [dtc_2, svc_2, lr_2], 
    'n_estimators':  [5, 10, 25, 50, 100],
    'learning_rate': [(0.97 + x / 100) for x in range(1, 7)]  
}
CV_abc = GridSearchCV(estimator=abc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[105]:


pipefinalabc= make_pipeline(col_trans, CV_abc)
pipefinalabc.fit(X_train, y_train)


# In[106]:


etc = ExtraTreesClassifier()
param_grid = {
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "n_estimators" :[100,300],
}
CV_etc = GridSearchCV(estimator=etc, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[107]:


pipefinaletc= make_pipeline(col_trans, CV_etc)
pipefinaletc.fit(X_train, y_train)


# In[108]:


GBC = GradientBoostingClassifier()
param_grid = {
              'n_estimators' : [300, 400, 500],
              'learning_rate': [ 0.1, 0.3, 0.6, 1.0],
              'max_depth': [8, 10, 12],
              'min_samples_leaf': [50, 100, 120, 150],
              'max_features': [0.1, 0.3, 0.5] 
              }
CV_gbc = GridSearchCV(estimator=GBC, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))


# In[110]:


pipefinalgbc= make_pipeline(col_trans, CV_gbc)
pipefinalgbc.fit(X_train, y_train)


# In[111]:


vc1 = VotingClassifier([('gbc', CV_gbc.best_estimator_),
                        ('etc', CV_etc.best_estimator_),
                          ('nb', CV_gnb.best_estimator_)
                         ], voting='hard', weights=[1,2,3] )
vc2 = VotingClassifier([('abc', CV_abc.best_estimator_),
                        ('etc', CV_etc.best_estimator_),
                          ('nb', CV_gnb.best_estimator_)
                         ], voting='hard', weights=[1,2,3])


# In[112]:


pipefinalcv1 = make_pipeline(col_trans, vc1)
pipefinalcv2 = make_pipeline(col_trans, vc2)
pipefinalcv1.fit(X_train, y_train)


# In[113]:


pipefinalcv2.fit(X_train, y_train)


# In[114]:


Y_pred = pipefinalrfc.predict(X_test)
Y_pred2 = pipefinaldtc.predict(X_test)
Y_pred3 = pipefinalknn.predict(X_test)
Y_pred4 = pipefinalsvc.predict(X_test)
Y_pred5 = pipefinallr.predict(X_test)
Y_pred6 = pipefinalgnb.predict(X_test)
Y_pred7 = pipefinalxg.predict(X_test)
Y_pred8 = pipefinalabc.predict(X_test)
Y_pred9 = pipefinaletc.predict(X_test)
Y_pred10 = pipefinalgbc.predict(X_test)
Y_pred11 = pipefinalcv1.predict(X_test)
Y_pred12 = pipefinalcv2.predict(X_test)


# In[115]:


submission = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred
})

submission2 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred2
})

submission3 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred3
})

submission4 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred4
})

submission5 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred5
})

submission6 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred6
})

submission7 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred7
})

submission8 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred8
})

submission9 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred9
})

submission10 = pd.DataFrame({
    'PassengerId': titanic_data_test['PassengerId'],
    'Survived': Y_pred10
})

submission11 = pd.DataFrame({
        "PassengerId": titanic_data_test["PassengerId"],
        "Survived": Y_pred11
})

submission12 = pd.DataFrame({
        "PassengerId": titanic_data_test["PassengerId"],
        "Survived": Y_pred12
})


# In[118]:


submission.to_csv('submission929_1.csv', index=False)
submission2.to_csv('submission929_2.csv', index=False)
submission3.to_csv('submission929_3.csv', index=False)
submission4.to_csv('submission929_4.csv', index=False)
submission5.to_csv('submission929_5.csv', index=False)
submission6.to_csv('submission929_6.csv', index=False)
submission7.to_csv('submission929_7.csv', index=False)
submission8.to_csv('submission101_8.csv', index=False)
submission9.to_csv('submission101_9.csv', index=False)
submission10.to_csv('submission101_10.csv', index=False)
submission11.to_csv('submission101_11.csv', index=False)
submission12.to_csv('submission101_12.csv', index=False)


# In[ ]:




