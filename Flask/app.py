from flask import Flask, render_template, url_for, request
import numpy as np
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random
from src.API import *
from src.gbc_predict import * 


app = Flask(__name__)
with open('../models/GBCmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/GBCmodelScaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

client = EventAPIClient1()
row = client.collect()
data =pd.DataFrame(row)
@app.route('/')
@app.route('/home')
def home():
    
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")



@app.route('/predict')
def predict(data=data):
    row = client.collect()
    data = data.append(pd.DataFrame(row), ignore_index=True)
    
    
    
    X = get_example_X_y(data, scaler)
    pred =model.predict_proba(X)
    
    data['prediction'] = np.around(pred[:,1], decimals=4)
    data['risk'] = data['prediction'].apply(lambda x: "High" if x > .6 else "Medium" if x > .3 else "Low")
    df=data[['name','prediction','risk']]
        

        
    return render_template('results.html', table=df.to_html(), titles=df.columns.values ,pred=0, name=data['name'].values )
    
    #return render_template("results.html", column_names=df.columns.values, row_data=list(df.values.tolist()),
     #                    zip=zip)




if __name__=="__main__":

    
    
    app.run(debug=True)