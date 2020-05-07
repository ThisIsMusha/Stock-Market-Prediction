import math
import pandas_datareader as web
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
from datetime import datetime
from flask import Flask, request, jsonify, render_template 
plt.style.use('fivethirtyeight')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    edate= datetime.today().strftime('%Y-%m-%d')
    days = request.form['days']

    df = web.DataReader('AAPL', data_source='yahoo',start='1980-12-12', end=datetime.today().strftime('%Y-%m-%d'))
    #get the close price
    df = df[['Close']]
    #print(df)

    # A variable for predicting 'n' days out into the future
    forecast_out = int(days)
    #create another column (the target dependent varible) shifted 'n' units up
    df['Prediction'] = df[['Close']].shift(-forecast_out)
    #print(df)

    ### Create the independant dataset (x)
    # convert the datafreame to a numpy array
    X = np.array(df.drop(['Prediction'],1))
    #remove the last 'n' rows
    X = X[:-forecast_out]
    #print(X)

    ### Create the dependent dataset (y)
    # convert the dataframe to a numpy array(All the values including NaN's)
    Y = np.array(df['Prediction'])
    # get all of the y values except the last 'n' rows
    Y = Y[:-forecast_out]
    #print(Y)

    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


    # Create and train Linear regression model
    model = LinearRegression()
    # train the model
    model.fit(x_train, y_train)

    


    ## testing model: Score returns the coefficient of determination R^2 of the prediction
    ## best score is 1.0
    model_confidence = model.score(x_test, y_test)
    #print("lr confidence: ", lr_confidence)


    #forcase data
    x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
    #print(x_forecast)

    model_prediction = model.predict(x_forecast)

    #date = edate,"\n"
    #pred = "Predicted closing value $ :",model_prediction,"\n"
    #mod = "Model Score : ",model_confidence,"\n"

    return render_template('index.html',date_text='Current Date: {}'.format(edate), prediction_text='Predicted closing value $ {}'.format(model_prediction), model_text='Model Score: {}'.format(model_confidence))

#return render_template('index.html', prediction_text='{}\n Predicted closing value $ {}\n Model Score {}'.format(edate,model_prediction,model_confidence))

if __name__ == "__main__":
    app.run(debug=True)