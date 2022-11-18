import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import requests
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('prediction.html',data = {'result' : ""})

#method to predict from deployed model
def predictFromDeploymentModel(userInput):
    API_KEY = ""
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    payload_scoring = {"input_data": [{"fields": ['yearOfRegistration'	,'powerPS'	,'kilometer'	,'monthOfRegistration'	,'gearbox_labels',	'notRepairedDamage_labels',	'model_labels',	'brand_labels',	'fuelType_labels',	'vehicleType_labels'], "values": [userInput]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/9528ebfb-f57a-4b7b-9684-d745eea3da24/predictions?version=2022-11-08', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    predictions = response_scoring.json()
    return (predictions['predictions'][0]['values'][0][0])

def getMonthNumber(monthName):
    months = {
        'january' : 1,
        'february' : 2,
        'march' : 3,
        'april' : 4,
        'may' : 5,
        'june' : 6,
        'july' : 7,
        'august' : 8,
        'september' : 9,
        'october' : 10,
        'november' : 11,
        'december' : 12,
    }

    return months[monthName]
    

@app.route('/y_predict', methods=['GET', 'POST']) 
def y_predict():
    regyear = int(request.form['regyear'])
    powerps = float(request.form['powerps'])
    kms = float(request.form['kms'])
    regmonth = getMonthNumber(request.form.get('regmonth'))
    gearbox = request.form['gearbox'] 
    damage = request.form['dam']
    model = request.form.get('modeltype')
    brand= request.form.get('brand') 
    fuelType = request.form.get('fuel')
    vehicletype= request.form.get('vehicletype')

    new_row = {'yearOfRegistration':regyear, 'powerPS':powerps, 'kilometer': kms, 'monthOfRegistration':regmonth, 'gearbox': gearbox, 'notRepairedDamage': damage,'model':model, 'brand': brand, 'fuelType': fuelType, 'vehicleType': vehicletype}
    new_df = pd.DataFrame(columns =[ 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType','brand', 'notRepairedDamage']) 
    new_df = new_df.append(new_row, ignore_index = True)


    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

    mapper = {} 
    
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes'+i+'.npy'),allow_pickle=True) 
        tr = mapper[i].transform(new_df[i])
        new_df.loc[:, i+'_labels'] = pd.Series (tr, index=new_df.index) 

    labeled = new_df[['yearOfRegistration','powerPS','kilometer','monthOfRegistration']+[x+'_labels' for x in labels]]
    X = labeled.values
    result = predictFromDeploymentModel(list(X[0]))
    data = {"result" : "Predicted price - $ "+str(round(result,2))}
    return render_template('prediction.html' , data = data)


if __name__ == '__main__':
   app.run(host='localhost', debug=True, threaded=False)

