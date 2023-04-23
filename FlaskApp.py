#!/usr/bin/env python
# coding: utf-8

# In[2]:


dataset.head()


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib
# Importing the data set
dataset = pd.read_csv(r'C:\Users\I551126\Desktop\sales.csv')

# separate feature & target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the data set into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# we can use either pickle or dump to save the model
#pickle.dump(regressor, open('model.pkl','wb'))
joblib.dump(regressor, 'model.pkl')


# In[15]:


# import flask
from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__,template_folder='C:/Users/I551126/Desktop/templates')

loaded_model = joblib.load('model.pkl')
@app.route("/")
def root():
    return render_template("index.html")
    #return redirect("index.html", code=500)
@app.route("/predict", methods=['POST'])
def predict_sales():

    if request.method == 'POST':
        exp = request.form['exp']
        X = [[float(exp)]]
        [prediction] = loaded_model.predict(X)
        salary = round(prediction, 2)
    msg = "Standard salary for provided experience of  " + str(exp) + " years, would be: ₹ " + str(salary) + "/-- "

    return render_template("index.html", prediction_text= msg)
if __name__ == '__main__':
    app.run(debug=False)

