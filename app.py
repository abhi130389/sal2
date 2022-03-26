# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:48:43 2022

@author: shash
"""
from flask import Flask, jsonify
import joblib

filename = 'process/model/sal_model.pkl'

app = Flask(__name__)
@app.route('/')
def index():
    return "Square Prediction"

@app.route('/sal/<int:x>', methods = ['GET'])
def predict(x):
    loaded_model = joblib.load(filename)
    y=loaded_model,predict([[x]])[0]
    sal=jsonify({'square': round(y,2)})
    return sal

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=80)