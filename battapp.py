#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:49:58 2022

@author: batta
"""
import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('knn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('characteristics.html')

@app.route('/predict',methods=['Get','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    def int_to_word(decision):
        word_dict = {0:"Not buy", 1:"Buy"}
        return word_dict[decision]
    output = int_to_word(prediction[0])
    return render_template('characteristics.html', prediction_text='Purchase decision will be {}'.format(output))

if __name__ == "__main__":
    app.run(port = 3232,debug=True)
    