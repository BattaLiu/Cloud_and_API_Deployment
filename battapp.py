#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 00:49:58 2022

@author: batta
"""
import pandas as pd
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)
model = pickle.load(open('knn_model.pkl', 'rb'))

@app.route('/')
def home():
    data = "Hello world!"
    return jsonify({'data':data})

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = request.args.get('age')
    estimated_salary = request.args.get('estimated_salary')
    test_df = pd.DataFrame({'Age': [age], 'Income': [estimated_salary]})
    prediction = model.predict(test_df)
    def int_to_word(decision):
        word_dict = {0:"Not buy", 1:"Buy"}
        return word_dict[decision]
    output = int_to_word(prediction[0])
    return jsonify({'Purchase Decision:': output})

if __name__ == "__main__":
    app.run(port = 3232,debug=True)
    