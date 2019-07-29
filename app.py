# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:57:41 2019

@author: sayadav
"""

from flask import Flask
import os  

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello world"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 33507))
    app.run(host='0.0.0.0', port=port)
