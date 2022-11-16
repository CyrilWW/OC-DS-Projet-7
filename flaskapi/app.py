# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, render_template, jsonify
import json
import requests
import datetime

app = Flask(__name__)

METEO_API_KEY = "ff879e4c9e1ce2f1e8c3080dca25d4b9"
if METEO_API_KEY is None:
    # URL de test :
    METEO_API_URL = "https://samples.openweathermap.org/data/2.5/forecast?lat=0&lon=0&appid=xxx"
else: 
    # URL avec clé :
    METEO_API_URL = "https://api.openweathermap.org/data/2.5/forecast?lat=48.883587&lon=2.333779&appid=" + METEO_API_KEY

@app.route("/")
def hello():
    return f"Version <b>Flaskapi</b> | {datetime.datetime.now().strftime('%d/%m/%Y @ %Hh%Mm%Ss')}!!"

@app.route('/api/prediction/')
def predict():
    threshold = np.random.uniform(0.5, 1.0)
    risk = np.random.uniform(0., 1.)
    return jsonify({
        'risk': risk,
        'threshold': threshold
        })

@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/meteo/')
def meteo():
    response = requests.get(METEO_API_URL)
    content = json.loads(response.content.decode('utf-8'))

    if response.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API météo n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
        }), 500

    data = [] # On initialise une liste vide
    for prev in content["list"]:
        datetime = prev['dt'] * 1000
        temperature = prev['main']['temp'] - 273.15 # Conversion de Kelvin en °c
        temperature = round(temperature, 2)
        data.append([datetime, temperature])
 
    return jsonify({
      'status': 'ok', 
      'data': data
    })

NEWS_API_KEY = None # Remplacez None par votre clé NEWSAPI, par exemple "4116306b167e49x993017f089862d4xx"

if NEWS_API_KEY is None:
    # URL de test :
    NEWS_API_URL = "https://s3-eu-west-1.amazonaws.com/course.oc-static.com/courses/4525361/top-headlines.json" # exemple de JSON
else:
    # URL avec clé :
    NEWS_API_URL = "https://newsapi.org/v2/top-headlines?sortBy=publishedAt&pageSize=100&language=fr&apiKey=" + NEWS_API_KEY

@app.route('/api/news/')
def get_news():
    response = requests.get(NEWS_API_URL)
    content = json.loads(response.content.decode('utf-8'))
    if response.status_code != 200:
        return jsonify({
            'status': 'error',
            'message': 'La requête à l\'API des articles d\'actualité n\'a pas fonctionné. Voici le message renvoyé par l\'API : {}'.format(content['message'])
        }), 500

    keywords, articles = extract_keywords(content["articles"])
    return jsonify({
        'status'   : 'ok',
        'data'     :{
            'keywords' : keywords[:100], # On retourne uniquement les 100 premiers mots
            'articles' : articles
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
