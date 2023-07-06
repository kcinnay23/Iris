import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF


# Définition de la classe PDF
class PDF(FPDF):
    def header(self):
        self.set_fill_color(255, 255, 255)  # Couleur de fond blanche
        self.rect(10, 20, self.w - 20, 10, 'F')  # Rectangle pour le fond

        # Ajouter l'image à gauche
        self.image('Gsmile.png', x=15, y=10, w=20)
        # Ajouter l'image à droite
        self.image('Gsmile.png', x=self.w - 35, y=10, w=20)

        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 0, 0)  # Couleur du texte noir
        self.cell(self.w - 60, 10, 'Give Smile Prediction', 0, 0, 'C')

        self.set_line_width(0.5)  # Largeur de la ligne
        self.dashed_line(10, 32, self.w - 10, 32, dash_length=2, space_length=2)  # Ligne de délimitation en pointillés

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=True, align='C')

    def chapter_body(self, content):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, content, align='C')


# Création de l'objet PDF
pdf = PDF()

st.write("""
# App simple pour la prévision des fleurs d'iris
Cette application prédit le type de fleur d'iris!
""")

st.sidebar.header('Paramètres utilisateur')


def user_input_features():
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

st.subheader('Catégorie de la fleur d\'iris')
st.write(df)

iris = datasets.load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)

prediction = clf.predict(df)

# Ajouter une page au PDF
pdf.add_page()

# Entête
pdf.header()

# Espace avant le résultat
pdf.ln(40)

# Centrer le résultat de l'analyse
pdf.set_font('Arial', '', 12)
pdf.set_xy(10, pdf.get_y())
pdf.cell(0, 10, f"La catégorie de la fleur d'iris est : {iris.target_names[prediction][0]}", 0, 1, 'C')

# Aperçu du résultat
st.subheader("Aperçu du résultat :")
st.write(f"La catégorie de la fleur d'iris est : {iris.target_names[prediction][0]}")

# Téléchargement du PDF
pdf_output = pdf.output(dest='S').encode('latin-1')
st.download_button("Télécharger les résultats (PDF)", data=pdf_output, file_name="resultats_prediction.pdf")
