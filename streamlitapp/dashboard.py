import streamlit as st
import json
import requests as re

"""
Développer un dashboard interactif permettant :
 - aux chargés de relation client d'expliquer de façon la plus transparente possible les décisions d’octroi de crédit 
 - à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

Celui-ci devra contenir au minimum les fonctionnalités suivantes :
 - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour 
   une personne non experte en data science.
 - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
 - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe 
   de clients similaires.
"""
# st.set_page_config( 
#     page_title="Loan Prediction App",
#     page_icon="./loan.png"
# )

st.title("Prêt à dépenser - tableau de bord")
st.subheader("Application web Streamlit utilisant un modèle de Machine Learning servi par une API, prédisant le risque de défaut de paiement d'un client.")

# st.write("""
# ## About
# **Application web Streamlit utilisant un modèle de Machine Learning servi par une API, prédisant le risque de défaut de paiement d'un client.** 
# """)

st.image("header.png")

col1, col2 = st.columns([1, 1])

st.write("""
# Informations client"""
)

st.header('Etat civil')

SK_ID_CURR = st.number_input("""Numéro de client""")
# CODE_GENDER = st.text_input("""Sexe (F|M)""")
CODE_GENDER = st.number_input("""Sexe (0|1)""")

st.header('Informations socio-professionnelles')

st.header('Informations prêt en cours')

st.header('Informations précédent(s) prêt(s)')

if st.button("Décision"):

    values = {
            "SK_ID_CURR": SK_ID_CURR,
            "CODE_GENDER": CODE_GENDER,
        }

    res = re.post(f"http://192.168.1.71:5000/prediction", json=values)
    json_str = json.dumps(res.json())
    resp = json.loads(json_str)

    if CODE_GENDER not in ['0', '1']:
        st.write("Erreur! Entrez le sexe du client.")
    else:
        st.write(f"""### The Price of the {names} is {resp[0]}$.""")