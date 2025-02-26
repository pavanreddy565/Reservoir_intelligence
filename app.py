from flask import Flask, render_template, jsonify, request
from supabase import create_client, Client
from flask_cors import CORS,cross_origin  # Import CORS
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import Load as ld
import GetWeek as gw


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = Flask(__name__)
CORS(app)

@app.route('/')
def main_page():
    x = ld.Load()
    x.updateData()
    return render_template('index.html')

@app.route('/getWeekStorage', methods=['GET'])
def getWeekStorage():
    gaw = gw.GetWeek()
    df = gaw.getStorage()
    
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y-%m-%d')
    data = df.to_dict(orient='records')
    return jsonify(data)

@app.route('/getWeekInflow', methods=['GET'])
def getWeekInflow():
    gaw = gw.GetWeek()
    df = gaw.getInflow()
    
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y-%m-%d')
    data = df.to_dict(orient='records')
    return jsonify(data) 
@app.route('/getWeekOutflow', methods=['GET'])
def getWeekOutflow():
    gaw = gw.GetWeek()
    df = gaw.getOutflow()
    
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True).dt.strftime('%Y-%m-%d')
    data = df.to_dict(orient='records')
    return jsonify(data) 
    
@app.route('/current_status')
def bar_plot():
    return render_template('current.html')

@app.route('/getForecast',methods=['POST'])
def getForecast():
    data = request.get_json()
    reservoir = data.get('reservoir')
    getValue = data.get('value')
    gd = gw.GetDays(90)
    df = gd. getPredictions(reservoir, getValue)
    data = df.to_dict(orient='records')
    return jsonify(data) 

@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/getDemand', methods=['GET'])
def getDemand():
    gd = gw.GetDemand(30)
    df = gd.getPredictions()
    data = df.to_dict(orient='records')
    return jsonify(data) 
@app.route('/getSupply', methods=['GET'])
def getSupply():
    gs = gw.GetSupply()
    df = gs.getFinal()
    data = df.to_dict(orient='records')
    return jsonify(data)

@app.route('/Home')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/resource')
def resource():
    return render_template('resource.html')
@app.route('/getTable', methods=['GET'])
def getTable():
    gs = gw.GetSupply()
    df = gs.getTable()
    df['Date'] = pd.to_datetime(df['Date'])
    gd = gw.GetDemand(30)
    df_ = gd.getPredictions()
    df_['Date'] = pd.to_datetime(df_['Date'])
    result_df = df.merge(df_,how='left',on='Date')
    result_df.rename(columns={"Outflow":"Demand (cusecs)"},inplace=True)
    data = result_df.to_dict(orient='records')
    return jsonify(data) 
    
if __name__ == '__main__':
    app.run(debug=True)