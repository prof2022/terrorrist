
#Import the required libraries for the machine learning application.
from sklearn.preprocessing import LabelEncoder
from numpy.core.numeric import True_
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

import streamlit as st
import pickle

st.title('Streamlit WebApp For predicting Terrorist Attack')

    

def load():
    data= pd.read_csv("subAfricaDf.csv")
    
    return data
df = load()

        
features = df[['Country', 'AttackType', 'Target_type', 'Group', 'Weapon_type', 'Suicide']]
target = df[['Success']]





cols = ['Country', 'AttackType', 'Target_type', 'Group', 'Weapon_type', 'Suicide']
st.sidebar.markdown(
            '<p class="header-style">Terrorist Attack Featutes</p>',
            unsafe_allow_html=True
        )
Country = st.sidebar.selectbox(
            f"Select {cols[0]}",
            sorted(features[cols[0]].unique())
        )


AttackType = st.sidebar.selectbox(
            f"Select {cols[1]}",
            sorted(features[cols[1]].unique())
        )

Target_type = st.sidebar.selectbox(
            f"Select {cols[2]}",
                        sorted(features[cols[2]].unique())
        )
Group = st.sidebar.selectbox(
            f"Select {cols[3]}",
            sorted(features[cols[3]].unique())
        )

Weapon_type = st.sidebar.selectbox(
            f"Select {cols[4]}",
            sorted(features[cols[4]].unique())
        )
Suicide = st.sidebar.selectbox(
            f"Select {cols[5]}",
                        sorted(features[cols[5]].unique())
        )

#label encoding
label= LabelEncoder()


df['Country_n'] = label.fit_transform(df['Country'])
df['Attack_Type_n'] = label.fit_transform(df['AttackType'])
df['Target_Type_n'] = label.fit_transform(df['Target_type'])
df['Attack_Group_n'] = label.fit_transform(df['Group'])
df['Weapon_n'] = label.fit_transform(df['Weapon_type'])

df = df.drop(['Country', 'AttackType', 'Target_type', 'Group', 'Weapon_type' ], axis='columns')
        
features = df[['Country_n', 'Attack_Type_n', 'Target_Type_n', 'Attack_Group_n', 'Weapon_n', 'Suicide']]
target = df[['Success']]



# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    pickle_in = open("terrorist_rf2.pkl","rb")
    clf=pickle.load(pickle_in)

 #Store inputs into dataframe
    X = [Country, AttackType, Target_type, Group, Weapon_type, Suicide]
    X_n = label.fit_transform(X)
    X_n = np.array(X_n).reshape(1, -1)

    prediction = clf.predict(X_n)
    
    if prediction == 1:
        prediction=" a Successful Terrorist Attack"
    else:
        prediction=" an Unsuccessful Terrorist Attack"
   
    
    #Output prediction
    st.text(f"The model predicts  {prediction}")
  