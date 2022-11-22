# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, render_template, jsonify, request
import json
import requests
import datetime
import pickle
import os
import pandas as pd


print(f"CWD = {os.getcwd()}")



def build_categ_vals(df, root_var_name):
    """Construit la liste de valeurs de la catégorie dans les noms de feature, à partir de la racine.
    
    Tient compte de la convention de nommage des features catégorielles OneHotEncodées : 'AAA%BBB__CCC_NR#MIN'
    
    Parameters
    ----------
    feat : root_var_name
        Racine des noms des features à traiter (issues d'un OneHotEncoder / get_dummies) 
    
    Returns
    -------
    categ_vals : list
        Liste de chaines correspondant aux valeurs de la catégorie root_var_name.
    """
    categ_vals = []
    for col in df.columns:
        if root_var_name in col:
            categ, categ_val = get_categ_val(col)
            categ_vals.append(categ_val)
    return categ_vals

app = Flask(__name__)

# Chargement du modèle, de l'explainer et des données
data_dir = './flaskapi'
file = open(f"{data_dir}/pipeline.pkl","rb")
pipeline = pickle.load(file)
file.close()

file = open(f"{data_dir}/explainer.pkl","rb")
explainer = pickle.load(file)
file.close()

file = open(f"{data_dir}/test_df_all_nan.pkl","rb")
test_df_all_nan = pickle.load(file)
file.close()
test_df_nonan = test_df_all_nan.drop(['TARGET', 'APPROVED'], axis=1)
test_df = test_df_nonan.copy()
test_df_nonan = test_df_nonan.fillna('N/A')
print(test_df_all_nan.shape)
print(test_df.shape)
print(test_df_nonan.shape)

file = open(f"{data_dir}/model_params.pkl","rb")
model_params = pickle.load(file)
file.close()

def compute_people_indicators():
    approved_cat_rates = {}
    people_means = {}

    # Moyennes numériques de la population
    for col in test_df.columns:
        if test_df[col].dtype == 'object':
            people_means[col] = test_df[col].value_counts().index[0] # mode
        else:
            people_means[col] = np.float64(test_df[col].mean())

    # Taux de approved/refused, par classe
    for col in test_df_all_nan.columns:
        # On regarde les colonnes catégorielles, mais pas ayant subi des opérations de moyennage, max, etc (avec un #)
        if '__' in col and not '#' in col:
            suf_df = test_df_all_nan.loc[test_df_all_nan[col]>0]
            if suf_df.shape[0]==0: # Pas de représentant de cette catégorie dans le test_df de la compétition
                approved_cat_rates[col] = np.float64(1.) # Tout le monde a gagné, donc. Cette valeur ne sera pas regardée dans le dashboard
            else:
                rate = suf_df['APPROVED'].sum() / suf_df.shape[0]
                approved_cat_rates[col] = np.float64(rate)
    print(f"approved_cat_rates = {approved_cat_rates}")
    return people_means, approved_cat_rates


# Endpoints
@app.route("/")
def hello():
    return f"<b>Flaskapi</b> | {os.getcwd()} | {datetime.datetime.now().strftime('%d/%m/%Y @ %Hh%Mm%Ss')}!!"

@app.route('/api/params')
def params():
    return jsonify(gui_params) 

@app.route('/api/client')
def client():
    # threshold = np.random.uniform(0.5, 1.0)
    if not 'threshold' in model_params:
        return f"<b>ERROR</b> | No 'threshold' in model prameters!!"

    # On choisit un client au hasard dans test_df
    x_client = test_df_nonan.sample(1).to_dict(orient='records')[0]
    # risk = np.random.uniform(0., 1.)
    return jsonify(x_client)

def my_isna(s):
    if s in ['N/A', 'NaN', '']:
        return True
    else:
        return False

def transform_input_nan_vals(x_client):
    x_client_w_nan = {}
    for key in x_client:
        if my_isna(x_client[key]):
            x_client_w_nan[key] = np.nan
        else:
            x_client_w_nan[key] = x_client[key]
    return x_client_w_nan

@app.route('/api/prediction') #, methods=['GET']
def predict():
    request_data = request.get_json()
    x_client = request_data.get('x_client')
    del(x_client['SK_ID_CURR']) # Ca tombe bien il faut l'enlever
    # Preprocess input X
    x_client_w_nan = transform_input_nan_vals(x_client)
    x_client_df = pd.DataFrame(data=x_client_w_nan, index=[0])

    # Prediction
    y_pred = pipeline.predict_proba(x_client_df)
    print(f"Prédiction OK.")

    # Approbation/refus du prêt
    risk = y_pred[0, 1]
    threshold = np.float64(model_params['threshold'])
    risk = np.float64(risk)

    # Explainer
    client_shap_values = explainer.shap_values(x_client_df)[0].tolist()
    feature_names = x_client_df.columns.tolist()

    # Positionnement du client

    data = {
        'decision': {
            'risk': risk,
            'risk_threshold': 1. - threshold
        },
        'decision_criteria': {
            'client_shap_values': client_shap_values,
            'feature_names': feature_names
        },
        'people_indicators': {
            'people_means': people_means,
            'approved_cat_rates': approved_cat_rates
        }
    }

    return jsonify(data)


@app.route("/dashboard")
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/news')
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
    people_means, approved_cat_rates = compute_people_indicators()
    app.run(host="0.0.0.0", debug=True)
