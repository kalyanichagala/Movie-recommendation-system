import pandas as pd


df=pd.read_csv("Classification_Model_data.csv")

X_train=df.drop(df[-1],axis=1)
y_train=df[-1]

X_test=df.drop(df[-1],axis=1)
y_train=df[-1]

X_val=df.drop(df[-1],axis=1)
y_val=df[-1]



