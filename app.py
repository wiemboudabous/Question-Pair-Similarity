from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap   
from cleaning_text import vectorize 
from cleaning_text import apply_clean 
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from joblib import dump
from joblib import load
import joblib
#import pickle
app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form=request.form
    if request.method=='POST':
        namequery1 = request.form['input1']
        namequery2 = request.form['input2']
        #print(type(namequery1)
        cleaned2=apply_clean(namequery2)
        cleaned1=apply_clean(namequery1)
        vec1=vectorize(str(cleaned1))
        vec2=vectorize(str(cleaned2))
        x=np.vstack((vec1,vec2))
        x_test=x.reshape(1,192)
        model = joblib.load("Pickle_RL_Model.pkl")
        y_predict = model.predict(x_test)
    return render_template('result.html',namequery1=namequery1,namequery2=namequery2,y_predict=y_predict)




if __name__ == '__main__':
	app.run(debug=True)