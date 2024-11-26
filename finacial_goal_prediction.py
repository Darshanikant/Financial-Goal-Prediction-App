import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_excel("family_financial_and_transactions_data.xlsx")

le=LabelEncoder() # le to labelencoder
le

# fit_tramsform lebel encoder to the dataset having categorical value
label_enc_col=['Category'] # select the column to applylabel encoder
# let apply
data[label_enc_col]=data[label_enc_col].apply(le.fit_transform)


# dependent and independet variable
x=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# built regression to fit the train data
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train) 

#
#prediction
y_pred=lr.predict(x_test)
print("Prediction:- ",y_pred)


#bais
bais=lr.score(x_train,y_train)
print("bais value:- ",bais)

#variance
var=lr.score(x_test,y_test)
print("The variance:- ",var)


# pickel the code
import pickle
filename='financial goal.pkl'
with open(filename,"wb") as file:
    pickle.dump(lr,file)


#scaler file
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)
   
print('Model has been picked saved in financial goal.pkl ')

# check the path
import os
print(os.getcwd())
