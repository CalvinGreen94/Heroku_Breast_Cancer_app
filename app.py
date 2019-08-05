import numpy as np
from flask import Flask, session,abort,request, jsonify, render_template,redirect,url_for,flash
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
import stripe
import datetime
import keras
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Breast_Cancer',methods=['POST'])
def Breast_Cancer():
    model = load_model('models/breast_cancer_model.h5')
    int_features = [[int(x) for x in request.form.values()]]
    mini=StandardScaler()
    # int_features = mini.fit_transform(int_features)
    # print(int_features)
    final_features = [np.array(int_features)]
    prediction_proba = model.predict(final_features)
    prediction = (prediction_proba > 0.5)
    return render_template('index.html', prediction_text='THANK YOU FOR YOUR PURCHASE, \n FOR THE DATA YOU ENTERED {}\n IT IS PREDICTED {} \n THAT THE CUSTOMER WILL LEAVE BANK.'.format(int_features,prediction))

if __name__ == "__main__":
    app.run(debug=True, port=8080) #debug=True,host="0.0.0.0",port=50000
