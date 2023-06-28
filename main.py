import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# App simple pour la prévision des fleurs d'iris
Cette application prédit le type de fleur d'iris!""")

st.sidebar.header('Les Paramètres utilisateur')

def user_input_features():
    #valeur, min, max, default
    sepal_length = st.sidebar.slider('Longueur du sépale', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Largeur du sépale', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Longueur du pétale', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Largeur du pétale', 0.1, 2.5, 0.2)
    data = {
        'Longueur du sépale': sepal_length,
        'Largeur du sépale': sepal_width,
        'Longueur du pétale': petal_length,
        'Largeur du pétale': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('On veut la catégorie de la fleur d\'iris')
st.write(df)

iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)

prediction=clf.predict(df)

st.subheader("La catégorie de la fleur d'iris est:")
st.write(iris.target_names[prediction])