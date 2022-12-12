import streamlit as st
import json
import requests as re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler

# Dashboard interactif permettant :
#  - aux chargés de relation client d'expliquer de façon la plus transparente possible les décisions d’octroi de crédit 
#  - à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

# Celui-ci contient les fonctionnalités suivantes :
#  - Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour 
#    une personne non experte en data science.
#  - Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
#  - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe 
#    de clients similaires.

# st.set_page_config( 
#     page_title="Loan Prediction App",
#     page_icon="./loan.png"
# )

# PREDICTION_API_URL = "http://localhost:5000"
# PREDICTION_API_URL = "http://192.168.1.71:5000"
# PREDICTION_API_URL = "http://192.168.1.71:5000"
PREDICTION_API_URL = "http://92.94.205.51:5000"

DATEORIGIN = datetime.datetime(2018,5,17) # Date de la compétition Kaggle
N_CRIT = 15
N_CAT_MAX = 10

IMG_PATH = Path(__file__).parent
print(IMG_PATH)

def get_categ_val(feat):
    """Récupère la valeur de la catégorie dans un nom de feature.
    
    Tient compte de la convention de nommage des features catégorielles OneHotEncodées : 'AAA%BBB__CCC_NR#MIN'
    Exemple : pour ce nom de colonne, la fonction renvoie ainsi 'CCC'.
    
    Parameters
    ----------
    feat : str
        Nom de la feature à traiter 
    
    Returns
    -------
    categ : str
        Chaine correspondant au nom de la catégorie
    categ_val : str
        Chaine correspondant à la valeur de la variable catégorielle
    """
    sep1 = '%'
    sep2 = '__'
    sep3 = '_NR'
    sep4 = '#'
    
    categ = ''
    categ_val = feat
    if sep1 in categ_val:
        categ_val = categ_val.split(sep1)[1]
    if sep2 in categ_val:
        categ = categ_val.split(sep2)[0]
        categ_val = categ_val.split(sep2)[1]
    if sep3 in categ_val:
        categ_val = categ_val.split(sep3)[0]
    if sep4 in categ_val:
        categ_val = categ_val.split(sep4)[0]
    return categ, categ_val


def append_dict_list(dico, key, val):
    """Etend une liste dans un dictionnaire si elle existe, et la crée sinon."""
    if key not in dico:
        dico[key] = []
    dico[key].append(val)
    return dico


def update_double_dict(dico, key1, key2, val):
    if key1 not in dico:
        dico[key1] = {}
    dico[key1][key2] = val
    return dico


def interpret_input_df_client_vars(x_client):
    num_vars = {}
    categ_vars = {}
    mapping = {}
    # On étudie chaque nom de colonne
    for orig_col in x_client: # Le dataframe doit avoir été transmis sous forme de dictionnaire
        # col = orig_col.replace('_NR','')
        col = orig_col
        if '__' in col: # Colonne catégorielle, OneHotEncodée
            categ, categ_val = get_categ_val(col)
            categ_vars = append_dict_list(categ_vars, categ, categ_val)
            mapping = update_double_dict(mapping, categ, categ_val, orig_col)
        else: # Colonne numérique
            num_vars[col] = 1 # Permet de les recenser
            mapping[col] = orig_col
    return num_vars, categ_vars, mapping


def search_key_with_val_in_dict(dico, val):
    """Recherche la clé key dans un dictionnaire dico telle que dico[key] == val"""
    key = None
    for key in dico:
        if dico[key]==val:
            return key


def search_keys_in_multidict_for_val(dico, val):
    key1 = None
    key2 = None
    for k1 in dico:
        if not isinstance(dico[k1], dict):
            continue
        for k2 in dico[k1]:
            if dico[k1][k2] == val:
                return k1, k2
    return key1, key2


def my_isna(s):
    if s in ['N/A', 'NaN', '', np.nan]:
        return True
    else:
        return False


def decode_x_test(x_test, num_vars, categ_vars, mapping):
    """Décode les informations contenues dans un sample, pour l'IHM du dashboard."""
    # st.write(x_test)
    x_gui = {}
    for orig_col in x_test: # Le dataframe doit avoir été transformé en dictionnaire
        # col = orig_col.replace('_NR','')
        col = search_key_with_val_in_dict(mapping, orig_col)
        if col is not None: # Colonne numérique
            if col not in num_vars:
                print(f"Erreur: {col} variable pas trouvée dans {num_vars}")
                x_gui[orig_col] = 'ERROR'
            else:
                if orig_col in ['DAYS_BIRTH', 'DAYS_EMPLOYED']:
                    if my_isna(x_test[orig_col]): x_test[orig_col] = 0
                    x_gui[orig_col] = DATEORIGIN + datetime.timedelta(days=x_test[orig_col])
                else:
                    x_gui[orig_col] = x_test[orig_col]
        else: # Colonne issue d'une variable catégorielle
            categ, categ_val = get_categ_val(orig_col)
            cat_val_list = categ_vars[categ]
            if x_test[orig_col]:
                ind = cat_val_list.index(categ_val)
                x_gui[categ] = ind
            else:
                x_gui[categ] = 0
    return x_gui


def update_x_test(x_test, info_gui, num_vars, categ_vars, mapping):
    """Met à jour un sample avec les infos de l'IHM."""
    for key in info_gui:
        if key in num_vars:
            if key in ['DAYS_BIRTH', 'DAYS_EMPLOYED']:
                if info_gui[key] == DATEORIGIN: 
                    x_test[key] = np.nan
                else:
                    d = info_gui[key]
                    x_test[key] = -(DATEORIGIN - datetime.datetime(d.year, d.month, d.day)).days
                print(x_test[key])
            elif key in ['CODE_GENDER_NR']:
                x_test[key] = 1 if info_gui[key]=='F' else 0
            elif key in ['FLAG_OWN_REALTY_NR', 'FLAG_OWN_CAR_NR']:
                x_test[key] = 1 if info_gui[key]=='Oui' else 0
            else:
                x_test[key] = info_gui[key]
        elif key in categ_vars:
            categ = key
            val = info_gui[categ]
            # OneHot manuel
            for v in categ_vars[categ]:
                col = mapping[categ][v]
                x_test[col] = 0
            col = mapping[categ][val]
            x_test[col] = 1
        else: # Ah, pas d'association possible
            pass
    # Variables feat.eng. du modèle
    x_test['DAYS_EMPLOYED_PERC'] = x_test['DAYS_EMPLOYED'] / x_test['DAYS_BIRTH']
    x_test['INCOME_CREDIT_PERC'] = x_test['AMT_INCOME_TOTAL'] / x_test['AMT_CREDIT']
    x_test['INCOME_PER_PERSON'] = x_test['AMT_INCOME_TOTAL'] / x_test['CNT_FAM_MEMBERS_NR']
    x_test['ANNUITY_INCOME_PERC'] = x_test['AMT_ANNUITY'] / x_test['AMT_INCOME_TOTAL']
    x_test['PAYMENT_RATE'] = x_test['AMT_ANNUITY'] / x_test['AMT_CREDIT']

    return x_test


def build_client_shap_values_agg(feature_names, client_shap_values, num_vars, categ_vars, mapping):
    client_shap_values_dict = {col: val for col, val in zip(feature_names, client_shap_values)}
    client_shap_values_agg = {}
    for feat in client_shap_values_dict:
        if feat in num_vars: # variable numérique
            client_shap_values_agg[feat] = client_shap_values_dict[feat]
        else : # variable catégorielle
            # Recherche de la variable catégorielle à l'origine de la variable courante 
            root_name, categ_val = search_keys_in_multidict_for_val(mapping, feat)
            if root_name is None:
                st.write(f"⚠ Problème dans la construction de client_shap_values_agg pour la variable {feat}")
                continue
            all_categ_cols = categ_vars[root_name]
            if root_name not in client_shap_values_agg:
                client_shap_values_agg[root_name] = 0.
            for col in all_categ_cols:
                client_shap_values_agg[root_name] += np.abs(client_shap_values_dict[mapping[root_name][col]])

    return client_shap_values_agg, client_shap_values_dict


def make_decision_criteria_graph(n_crit, client_shap_values, criteria_names, client_approved, num_vars, categ_vars, mapping, x_gui):
    most_impt_crit_names = [] # TODO faire le dico et l'agglo des shap dès le début
    client_shap_values_agg, client_shap_values_dict = build_client_shap_values_agg(criteria_names, client_shap_values, num_vars, categ_vars, mapping)
    criteria_names_agg = list(client_shap_values_agg.keys())
    shap_values_agg = list(client_shap_values_agg.values())
    df_crit = pd.DataFrame({"Critère": criteria_names_agg,
                            "Importance": shap_values_agg})
    if client_approved:
        df_crit['Importance'] = df_crit['Importance'] # On inverse les signe car les shap values + poussent le risque vers 1 donc la décision à 0, et vice-versa
    else:
        df_crit['Importance'] = - df_crit['Importance'] # On les les signes car les shap values + poussent le risque à 1 donc la décision à 0, et vice-versa
    
    df_crit['Importance abs'] = df_crit['Importance'].abs()
    
    # Traduction en pourcentages
    sum = df_crit['Importance abs'].sum()
    df_crit['Importance'] = df_crit['Importance'] / sum * 100.
    df_crit['Importance abs'] = df_crit['Importance abs'] / sum * 100.

    plt.rcParams['figure.figsize'] = [6, 6]
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig = plt.figure();
    df_crit = df_crit.sort_values(by="Importance abs", ascending=False)#.plot(kind='barh', grid=False, rot=0)
    df_crit_n = df_crit.iloc[:n_crit]
    most_impt_crit_names = df_crit_n['Critère'].values

    colors = ['yellowgreen' if val > 0. else 'red' for val in df_crit_n['Importance'].values]
    sns.barplot(data=df_crit_n, x="Importance", y="Critère", palette=colors)
    plt.title('Importance des critères de décision')
    plt.xlabel("Importance (%)")
    return fig, most_impt_crit_names


def build_radar_data(df, cols=None, label_col=None):
    variables, values, labels = [], {}, []
    
    if cols is None and label_col is None:
        print("Specify at least one arg between `cols` and `label_col`")
        return variables, values, labels
              
    # Label des individus
    if label_col is None:
        labels = [str(ind) for ind in range(df.shape[0])]
    else:
        labels = df[label_col]
        
    # Variables représentées sur le radar
    if cols is None:
        cols = [col for col in df.columns if col != label_col]

    variables = cols
    values = {}
    for ind in range(df.shape[0]):
        vals = df[cols].iloc[ind].tolist()
        # Pour fermer la courbe
        values[labels[ind]] = vals + [vals[0]]
    
    return variables, values, labels


def make_radar_chart(df, cols=None, label_col=None, title='', title_y=1.05):
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    variables, values, labels = build_radar_data(df, cols=cols, label_col=label_col)
    n_vars = len(list(values.values())[0])
    label_loc = np.linspace(start=0, stop=2*np.pi, num=n_vars)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    for key, label in zip(values.keys(), labels):
        ax.plot(label_loc, values[key], label=label)
    plt.title(title, y=title_y)
    label_loc = label_loc[:-1]
    lines, labs = plt.thetagrids(np.degrees(label_loc), labels=variables)
    plt.legend()
    return fig


def make_num_client_positioning(crit_names, crit_means, x_client):
    crit_names2 = [crit for crit in crit_names if not my_isna(x_client[crit])]
    x_client2 = {}
    crit_means2 = {crit: crit_means[crit] for crit in crit_names2}
    x_client2 = {crit: x_client[crit] for crit in crit_names2}
    df1 = pd.DataFrame.from_records([crit_means2])
    df2 = pd.DataFrame.from_records([x_client2])
    df = pd.concat((df1, df2), ignore_index=True)
    df = df.astype(float)
    df_sc = pd.DataFrame(MinMaxScaler(feature_range=(0.05, 1)).fit_transform(df), columns=df.columns)
    for col in df_sc.columns: df_sc[col] = df_sc[col] * 100.
    df_sc['Who'] = ['Population', 'Client']
    fig = make_radar_chart(df_sc, cols=crit_names2, label_col='Who')
    return fig


def get_whole_categ_cols(categ_vars, mapping, x_gui, crit_name=None, root_name=None):
    # Init
    all_categ_cols = None
    all_categ_vals = None
    client_type = None
    # Recherche de la variable catégorielle à l'origine de la variable courante 
    if crit_name is None and root_name is None:
        st.write(f"❌ crit_name et root_name ne peuvent pas être {None} en même temps")
        return root_name, all_categ_cols, all_categ_vals, client_type
    if crit_name is not None:
        root_name, categ_val = search_keys_in_multidict_for_val(mapping, crit_name)
    if root_name not in categ_vars:
        return root_name, all_categ_cols, all_categ_vals, client_type
    client_type = categ_vars[root_name][x_gui[root_name]] # categ_val
    all_categ_vals = categ_vars[root_name]
    all_categ_cols = [mapping[root_name][val] for val in all_categ_vals]
    return root_name, all_categ_cols, all_categ_vals, client_type


def make_cat_client_positioning(crit_names, approved_cat_rates, categ_vars, mapping, x_gui, client_approved=False):
    plt.rcParams['figure.figsize'] = [9, 6]
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    figs = [] 
    treated_root_names = []

    df_approved_cat_rates = pd.DataFrame.from_records([approved_cat_rates])
    if client_approved:
        color_approved = 'forestgreen'
    else:
        color_approved = 'red'
    color_lambda = 'grey'
    
    ind = 0
    for crit in crit_names:
        root_name, all_categ_cols, all_categ_vals, client_type = get_whole_categ_cols(categ_vars, mapping, x_gui, root_name=crit) # Nouvelle politique d'affichage
        if root_name is None:
            st.write(f"⚠ Critère {root_name} absent des infos disponibles !")
            continue

        # On ne refait pas les tracés pour la variable catégorielle si elle a déjà été traitée
        if root_name in treated_root_names:
            continue

        # Récupération des infos disponibles dans les importances 
        available_cols = [col for col in all_categ_cols if col in df_approved_cat_rates.columns]
        if len(available_cols) == 0:
            continue

        values = df_approved_cat_rates[available_cols].values[0, :]
        df_cat = pd.DataFrame({crit: all_categ_vals,
                               'PERC_APPROVED': values})
        df_cat['PERC_APPROVED'] = df_cat['PERC_APPROVED'] * 100.
        df_cat['PERC_REFUSED'] = 100. - df_cat['PERC_APPROVED']

        # Figures
        fig = plt.figure()
        
        rows = 1 #len(crit_names)
        cols = 2
        grid = plt.GridSpec(rows, cols, wspace=0.8, hspace=0.4)

        # Approved
        fig_ax1 = fig.add_subplot(grid[ind, 0])
        df_cat1 = df_cat.sort_values(by='PERC_APPROVED', ascending=False)
        clrs = [color_approved if (x==client_type) else color_lambda for x in df_cat1[crit].values ]
        sns.barplot(ax=fig_ax1, data=df_cat1.iloc[:N_CAT_MAX], x="PERC_APPROVED", y=crit, palette=clrs)
        fig_ax1.set_xlabel("% d'approbation");
        fig_ax1.set_ylabel("Catégorie");
        fig_ax1.set_title(f"Critère {root_name}");
        
        # Refused
        fig_ax2 = fig.add_subplot(grid[ind, 1])
        df_cat2 = df_cat.sort_values(by='PERC_REFUSED', ascending=False)
        clrs = [color_approved if (x==client_type) else color_lambda for x in df_cat2[crit].values ]
        sns.barplot(ax=fig_ax2, data=df_cat2.iloc[:N_CAT_MAX], x='PERC_REFUSED', y=crit, palette=clrs)
        fig_ax2.set_xlabel("% de refus");
        fig_ax2.set_ylabel("");
        fig_ax2.set_title(f"Critère {root_name}");
        
        figs.append(fig)

        # On mémorise que la variable catégorielle a été traitée
        treated_root_names.append(root_name)
    return figs

# @st.cache
def get_client_info():
    # Récupération des infos d'un client random
    res = re.get(f"{PREDICTION_API_URL}/api/client")
    if res.status_code != 200:
        st.write("❌ Erreur de communication des infos clients.")
        return
    json_str = json.dumps(res.json())
    x_client = json.loads(json_str)
    st.session_state.x_client = x_client
    st.session_state.num_vars, st.session_state.categ_vars, st.session_state.mapping = interpret_input_df_client_vars(x_client)
    st.session_state.x_gui = decode_x_test(st.session_state.x_client, st.session_state.num_vars, st.session_state.categ_vars, st.session_state.mapping)


def get_ihm_info():

    if not 'x_gui' in st.session_state:
        return


    categ_vars = st.session_state.categ_vars
    x_gui = st.session_state.x_gui

    info_gui = {}

    container = st.session_state.client_container
    container.header('Etat civil')
    aa = container.number_input("Numéro de client", value=x_gui['SK_ID_CURR'], min_value=100001, max_value=456255, format="%d", help="Tel qu'indiqué dans application_{train|test}.csv") # info_gui['SK_ID_CURR']
    info_gui['CODE_GENDER_NR'] = container.radio("Sexe :", ("F", "M"), index=1-x_gui['CODE_GENDER_NR'], horizontal=True)
    info_gui['DAYS_BIRTH'] = container.date_input("Date de naissance :", value=x_gui['DAYS_BIRTH'])
    info_gui['NAME_FAMILY_STATUS'] = container.selectbox("Statut matrimonial :", categ_vars['NAME_FAMILY_STATUS'], index=x_gui['NAME_FAMILY_STATUS'])

    container.header('Informations socio-professionnelles')
    info_gui['CNT_FAM_MEMBERS_NR'] = container.number_input("Composition du foyer :", min_value=0, max_value=12, value=int(x_gui['CNT_FAM_MEMBERS_NR']), format="%d")
    info_gui['NAME_EDUCATION_TYPE'] = container.selectbox("Niveau d'études :", categ_vars['NAME_EDUCATION_TYPE'], index=x_gui['NAME_EDUCATION_TYPE'])
    info_gui['AMT_INCOME_TOTAL'] = container.slider("Revenus annuels :", min_value=0., max_value=x_gui['AMT_INCOME_TOTAL']*3, step=1000., value=x_gui['AMT_INCOME_TOTAL'])
    info_gui['NAME_INCOME_TYPE'] = container.selectbox("Type d'emploi :", categ_vars['NAME_INCOME_TYPE'], index=x_gui['NAME_INCOME_TYPE'])
    info_gui['OCCUPATION_TYPE'] = container.selectbox("Poste :", categ_vars['OCCUPATION_TYPE'], index=x_gui['OCCUPATION_TYPE'])
    info_gui['DAYS_EMPLOYED'] = container.date_input("Date de prise d'emploi :", value=x_gui['DAYS_EMPLOYED'])
    info_gui['ORGANIZATION_TYPE'] = container.selectbox("Domaine professionnel :", categ_vars['ORGANIZATION_TYPE'], index=x_gui['ORGANIZATION_TYPE'])
    info_gui['FLAG_OWN_REALTY_NR'] = container.radio("Propriétaire foncier :", ('Non', 'Oui'), index=x_gui['FLAG_OWN_REALTY_NR'])
    info_gui['FLAG_OWN_CAR_NR'] = container.radio("Propriétaire d'un véhicule :", ('Non', 'Oui'), index=x_gui['FLAG_OWN_CAR_NR'])

    container.header('Informations prêt en cours')
    info_gui['NAME_CONTRACT_TYPE'] = container.selectbox("Type de prêt :", categ_vars['NAME_CONTRACT_TYPE'], index=x_gui['NAME_CONTRACT_TYPE'])
    info_gui['AMT_CREDIT'] = container.slider("Montant du prêt :", min_value=0., max_value=x_gui['AMT_CREDIT']*3, step=1000., value=x_gui['AMT_CREDIT'])
    info_gui['AMT_ANNUITY'] = container.slider("Montant annuel de remboursement :", min_value=0., max_value=x_gui['AMT_ANNUITY']*4, step=100., value=x_gui['AMT_ANNUITY']) # TODO : transformer en vraie mensualité
    info_gui['AMT_GOODS_PRICE'] = container.slider("Montant du bien :", min_value=0., max_value=x_gui['AMT_GOODS_PRICE']*2.5, step=1000., value=x_gui['AMT_GOODS_PRICE'])
    st.session_state.info_gui = info_gui
    

def launch_prediction():
    st.session_state.prediction = True
    
    # Mise à jour du client avec les infos IHM
    st.session_state.x_client = update_x_test(st.session_state.x_client, st.session_state.info_gui, 
                             st.session_state.num_vars, st.session_state.categ_vars, st.session_state.mapping)
    st.session_state.x_gui = decode_x_test(st.session_state.x_client, st.session_state.num_vars, st.session_state.categ_vars, st.session_state.mapping)

    # Encodage des infos IHM
    x_client = st.session_state.x_client
    values = {'x_client': x_client}

    res = re.get(f"{PREDICTION_API_URL}/api/prediction", json=values) # , verify=False
    if res.status_code != 200:
        st.write("❌ Erreur de communication de la prédiction.")
        return
    json_str = json.dumps(res.json())
    resp = json.loads(json_str)
    risk = resp['decision']['risk']
    risk_threshold = resp['decision']['risk_threshold']
    client_shap_values = resp['decision_criteria']['client_shap_values']
    criteria_names = resp['decision_criteria']['feature_names']
    crit_means = resp['people_indicators']['people_means']
    approved_cat_rates = resp['people_indicators']['approved_cat_rates']
    if risk > risk_threshold:
        client_approved = False
        icon_decision = Image.open(os.path.join(IMG_PATH, 'no.jpg'))
        str_decision = "Prêt refusé"
    else:
        client_approved = True
        icon_decision = Image.open(os.path.join(IMG_PATH, 'yes.jpg'))
        str_decision = "Prêt accordé"


    # container = st.session_state.report_container
    with st.container():
        st.title("Rapport de décision")
        st.markdown("""---""")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(icon_decision, width=80)
            st.subheader(str_decision)
        with col2:
            # Niveau de risque / seuil
            st.markdown(f"Risque de défaut de paiement / seuil de décision : **{risk:.2f} / {risk_threshold:.2f}**")
        st.markdown("""---""")

        # Eléments de décision
        st.subheader("Eléments de décision")
        fig, most_impt_crit_names = make_decision_criteria_graph(N_CRIT, client_shap_values, criteria_names, client_approved,
                                                                 st.session_state.num_vars, st.session_state.categ_vars,
                                                                 st.session_state.mapping, st.session_state.x_gui)
        st.pyplot(fig)
        st.markdown("""---""")

        # Positionnement du client, variables numériques
        st.subheader("Positionnement du client")
        st.markdown("**Critères numériques**")
        most_impt_num_crit_names = [crit for crit in most_impt_crit_names if crit in st.session_state.num_vars]
        fig = make_num_client_positioning(most_impt_num_crit_names, crit_means, x_client)
        if fig is None or len(most_impt_num_crit_names)<3:
            st.write("⚠ Pas assez de critères numériques dans la décision.")
        else:
            st.pyplot(fig)

        # st.image("separator")
        st.markdown("""---""")

        st.markdown("**Critères catégoriels**")
        # Positionnement du client, variables catégorielles
        most_impt_cat_crit_names = [crit for crit in most_impt_crit_names if crit not in most_impt_num_crit_names] # Mieux st.session_state.encoded_categ_vars
        figs = make_cat_client_positioning(most_impt_cat_crit_names, approved_cat_rates, 
                                           st.session_state.categ_vars, st.session_state.mapping,
                                           st.session_state.x_gui, client_approved=client_approved)
        if len(figs)==0:
            st.write("⚠ Pas assez de critères catégoriels dans la décision.")
        else:
            for fig in figs:
                st.pyplot(fig)
        st.markdown("""---""")



if not 'prediction' in st.session_state:
    st.title("Prêt à dépenser - tableau de bord")
    st.markdown("""---""")
    st.write("Application web prédisant le risque de défaut de paiement d'un client. L'application utilise un modèle de Machine Learning, servi par une API.")
    header_img = Image.open(os.path.join(IMG_PATH, "header.png"))
    st.image(header_img, caption=' ')
    st.markdown("""---""")


# Récupération des paramètres d'affichage

# Panneau latéral
st.sidebar.title("Informations client")
st.session_state.client_container = st.sidebar

client_img = Image.open(os.path.join(IMG_PATH, "client.png"))
st.sidebar.image(client_img, width=100)

st.sidebar.button("Nouveau client", on_click=get_client_info)

get_ihm_info()

if 'info_gui' in st.session_state:
    st.sidebar.button("Décision", on_click=launch_prediction)