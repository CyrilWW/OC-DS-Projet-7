import streamlit as st
import json
import requests as re
import datetime
import pandas as pd

# Développer un dashboard interactif permettant :
#  - aux chargés de relation client d'expliquer de façon la plus transparente possible les décisions d’octroi de crédit 
#  - à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

# Celui-ci devra contenir au minimum les fonctionnalités suivantes :
#  - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour 
#    une personne non experte en data science.
#  - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
#  - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe 
#    de clients similaires.

# st.set_page_config( 
#     page_title="Loan Prediction App",
#     page_icon="./loan.png"
# )

st.title("Prêt à dépenser - tableau de bord")
st.write("Application web Streamlit utilisant un modèle de Machine Learning servi par une API, prédisant le risque de défaut de paiement d'un client.")

# st.write("""
# ## About
# **Application web Streamlit utilisant un modèle de Machine Learning servi par une API, prédisant le risque de défaut de paiement d'un client.** 
# """)

st.image("header.png")

# col1, col2 = st.columns([1, 1])

st.sidebar.title("Informations client") 
st.sidebar.image("client.png", width=100)
if st.sidebar.button("Nouveau client"):
    # Récupération des infos d'un client random
    pass



st.sidebar.header('Etat civil')
# st.sidebar.write("")


SK_ID_CURR = st.sidebar.number_input("Numéro de client", value=100001, min_value=100001, max_value=456255, format="%d", help="Tel qu'indiqué dans application_{train|test}.csv")
# CODE_GENDER = st.sidebar.text_input("Sexe (F|M)")
CODE_GENDER = st.sidebar.selectbox("Sexe :", ("F", "M")) # TODO mapper en (1, 0)
DAYS_BIRTH = st.sidebar.date_input("Date de naissance :", value=datetime.datetime(1970,1,1))
NAME_FAMILY_STATUS = st.sidebar.selectbox("Statut matrimonial :", ("Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"))
FLAG_PHONE = st.sidebar.radio("Téléphone fixe :", ('Non', 'Oui'))
FLAG_MOBIL = st.sidebar.radio("Téléphone mobile  :", ('Non', 'Oui'))

st.sidebar.header('Informations socio-professionnelles')
CNT_CHILDREN = st.sidebar.number_input("Nombre d'enfants :", value=0, format="%d")
NAME_EDUCATION_TYPE = st.sidebar.selectbox("Niveau d'études :", ("Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"))
AMT_INCOME_TOTAL = st.sidebar.slider("Revenus annuels :", min_value=0, max_value=1000000, step=1000)
NAME_INCOME_TYPE = st.sidebar.selectbox("Type d'emploi :", ("Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"))
DAYS_EMPLOYED = st.sidebar.date_input("Date de prise d'emploi :", value=datetime.datetime(1990,1,1))
ORGANIZATION_TYPE = st.sidebar.selectbox("Domaine professionnel :", ("Business Entity Type 3", "Self-employed", "Other", "Medicine", "Business Entity Type 2", "Government", "School", "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1", "Transport: type 4", "Trade: type 3", "Industry: type 9", "Industry: type 3", "Security", "Housing", "Industry: type 11", "Military", "Bank", "Agriculture", "Police", "Transport: type 2", "Postal", "Security Ministries", "Trade: type 2", "Restaurant", "Services", "University", "Industry: type 7", "Transport: type 3", "Industry: type 1", "Hotel", "Electricity", "Industry: type 4", "Trade: type 6", "Industry: type 5", "Insurance", "Telecom", "Emergency", "Industry: type 2", "Advertising", "Realtor", "Culture", "Industry: type 12", "Trade: type 1", "Mobile", "Legal Services", "Cleaning", "Transport: type 1", "Industry: type 6", "Industry: type 10", "Religion", "Industry: type 13", "Trade: type 4", "Trade: type 5", "Industry: type 8"))
FLAG_OWN_REALTY = st.sidebar.radio("Propriétaire foncier :", ('Non', 'Oui'))
FLAG_OWN_CAR = st.sidebar.radio("Propriétaire d'un véhicule :", ('Non', 'Oui'))

st.sidebar.header('Informations prêt en cours')
NAME_CONTRACT_TYPE = st.sidebar.selectbox("Type de prêt :", ("Cash loans", "Revolving loans"))
AMT_CREDIT = st.sidebar.slider("Montant du prêt :", min_value=0, max_value=5000000, step=1000)
AMT_ANNUITY = st.sidebar.slider("Montant mensuel de l'échéance :", min_value=0, max_value=10000, step=100) # TODO : transformer en vraie mensualité
AMT_GOODS_PRICE = st.sidebar.slider("Montant du bien :", min_value=0, max_value=5000000, step=1000)

st.sidebar.header('Informations précédent(s) prêt(s)')
st.sidebar.subheader("""(...)""")


if st.sidebar.button("Décision"):

    values = {
            "SK_ID_CURR": SK_ID_CURR,
            "CODE_GENDER": CODE_GENDER,
            # etc.
        }

    if CODE_GENDER not in ['F', 'F']:
        st.write("Erreur! Entrez le sexe du client.")
        
    # res = re.post(f"http://192.168.1.71:5000/api/prediction/", json=values)
    res = re.get(f"http://192.168.1.71:5000/api/prediction/", json=values)
    json_str = json.dumps(res.json())
    resp = json.loads(json_str)
    if resp['risk'] > resp['threshold']:
        decision = 'No'
        icon_decision = './no.jpg'
        str_decision = "Prêt refusé"
    else:
        decision = 'Yes'
        icon_decision = './yes.jpg'
        str_decision = "Prêt accordé"

    with st.container():
        st.image(icon_decision, width=80)
        st.subheader(str_decision)

        # Niveau de risque / seuil
        st.markdown(f"Risque de défaut de paiement / seuil de décision : **{resp['risk']:.2f} / {resp['threshold']:.2f}**")

        # Eléments de décision
        st.markdown("**Eléments de décision**")
        if decision == 'Yes':
            st.image('./bar_plot_yes.png')
        else:
            st.image('./bar_plot_no.png')

        # Positionnement du client, variables numériques
        st.subheader("Positionnement du client")
        st.markdown("**Critères numériques**")
        st.image('./radarplot.png', width=600)


        # Positionnement du client, variables catégorielles
        st.markdown("**Critère catégoriel 1**")
        st.image('positionnement_NAME_FAMILY_STATUS.png', width=600)

        st.markdown("**Critère catégoriel 2**")
        st.image('positionnement_NAME_EDUCATION_TYPE.png', width=600)

        st.markdown("**Critère catégoriel 3**")
        st.write("(*Autant de critères catégoriels que dans le radarplot*)")

