from flask import Flask,Response
from flask import render_template
from flask import request,redirect, url_for

import numpy as np
import pandas as pd

import datetime

from fbprophet import Prophet

import pickle

path = "city_day.csv"
city_day = pd.read_csv(path).sort_values(by = ['Date', 'City'])
city_day.Date = city_day.Date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))
city_day = city_day.sort_values(by = 'Date')
city_day.corr().AQI.sort_values(ascending = False)
# adding all the features with corr less than 0.4

city_day['B_X_O3_NH3'] = city_day['Benzene'] +\
city_day['Xylene'] + city_day['O3'] + city_day['NH3']

city_day['ParticulateMatters'] = city_day['PM2.5'] + city_day['PM10']

corr_with_AQI = city_day.corr().AQI.sort_values(ascending = False)

corr_with_AQI
most_polluted = city_day[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by = 'AQI', ascending = False)
most_polluted

most_polluted = city_day[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by = 'AQI', ascending = False)

cities = most_polluted.index
params = most_polluted.columns

def first_date(city, parameter):
    df = city_day[(city_day.City == city)]
    df = df[df[parameter].notnull()]
    if len(df) != 0:
        return df.iloc[0].Date.strftime('%Y-%m-%d')
    else: return('no_measurement')
        
        
for city in cities:
    for param in params:
        most_polluted.loc[city, str(param) + '_date'] = first_date(city, param)
        
most_polluted.head()

city_day['Year_Month'] = city_day.Date.apply(lambda x : x.strftime('%Y-%m'))

df = city_day.groupby(['Year_Month']).sum().reset_index()

metrices = corr_with_AQI[corr_with_AQI>0.5].index

cities = ['Ahmedabad','Delhi','Bengaluru','Mumbai','Hyderabad','Chennai']

filtered_city_day = city_day[city_day['Date'] >= '2019-01-01']
AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['Date','City','AQI','AQI_Bucket']]
AQI.head()

def tell_me_null(df):
    num_null = df.isnull().sum().sort_values(ascending = False)
    percentage_null = round(df.isnull().sum().sort_values(ascending = False)/len(df) * 100, 1)
    return pd.DataFrame(np.c_[num_null, percentage_null], index = num_null.index,  columns = ['# of Null', 'Percentage'])

tell_me_null(city_day)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Forecast/',methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        givencity1 = request.form['cities']
        givencity = city_day[(city_day.AQI.notnull()) & (city_day.City == givencity1)]
        corr = givencity.corr().AQI.sort_values(ascending = False)
        related = list(corr[corr>0.6].index)
        inter = givencity.loc[:, related].interpolate(method = 'linear')
        givencity.loc[:, related] = inter
        givencity_aqi = givencity[['Date','AQI']]
        givencity_aqi.reset_index(inplace = True,drop = True)
        train_df = givencity_aqi
        train_df.rename(mapper = {'Date':'ds','AQI':'y'},axis =1,inplace = True)

        if(givencity1=='delhi'):
            with open(r'modeldelhi', 'rb') as file:  
                model = pickle.load(file)
        elif(givencity1=='kolkata'):
            with open(r'modelkolkata', 'rb') as file:  
                model = pickle.load(file)
        elif(givencity1=='chennai'):
            with open(r'modelchennai', 'rb') as file:  
                model = pickle.load(file)
        elif(givencity1=='mumbai'):
            with open(r'modelmumbai', 'rb') as file:  
                model = pickle.load(file)
        elif(givencity1=='hyderabad'):
            with open(r'modelhyderabad', 'rb') as file:  
                model = pickle.load(file)                

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        predictions_df=pd.DataFrame(forecast,columns=['ds','yhat'])
        predictions = predictions_df.to_csv(index=False)
        return Response(
        predictions,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=predictions.csv"})
                
    return render_template('2.html')

@app.route('/LifeinPandemic/')
def pandemic():
    return render_template('4.html')

if __name__ == '__main__':
    app.run(debug=True,port=8000)