import os
os.environ['KERAS_BACKEND' ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import pandas as pd
import keras
import sklearn
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense,Dropout
from keras.layers.core import Dropout, Activation
import time
from keras.models import Model
from keras.layers.merge import concatenate
NAME = "Shared_input_layer"
data = pd.read_csv('data.csv')
df0,df1 = data.shape[0], data.shape[1]
data = data.drop(['id'], axis=1)
data = data.drop(['Unnamed: 32'], axis=1)
labelencoder= LabelEncoder()
X = data.drop(['diagnosis'],axis=1)
pd.DataFrame(X[:5])

y = data['diagnosis']
y = labelencoder.fit_transform(y)
pd.DataFrame(y[:5])

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3)
batch_size = 75
dropout = 0.64

visible = Input(shape=(30,))
m1 = Dense(6, activation='sigmoid')(visible)
m1 = Dense(6, activation='sigmoid')(m1)
m1 = Dropout(dropout)(m1)

m2 = Dense(6, activation='sigmoid')(visible)
m2 = Dense(6, activation='sigmoid')(m2)

m3 = Dense(6, activation='sigmoid')(visible)
m3 = Dense(6, activation='sigmoid')(m3)
m3 = Dropout(dropout)(m3)

merge = concatenate([m1,m2,m3],axis=1)

output = Dense(1, activation='sigmoid')(merge)
model = Model(inputs=visible, outputs=output)
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
model.fit(X_train,y_train,batch_size=batch_size,epochs=500,validation_split=0.47,callbacks=[tensorboard])


model.evaluate(X_test,y_test)
print(y_test[:30])
print(model.predict(X_test[:30]))
model.save('breast_cancer_model.h5')
