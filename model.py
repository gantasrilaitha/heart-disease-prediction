import pandas as pd
import sys
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('h.csv')
df.rename(columns={"class":"target"},inplace=True)
df['target'].replace(['absent','present'],[0,1],inplace=True)

df=pd.get_dummies(df)
print(df)
x=df.drop('target',axis=1)
y=df['target']
#print(x,y)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=35)
logr=LogisticRegression()
logr.fit(train_x,train_y)
pickle.dump(logr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
result=model.predict(test_x)
print(result)








