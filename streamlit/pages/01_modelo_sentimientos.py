import streamlit as st 


#if st.checkbox("mostrar texto"):
    #st.write("hola")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle
import os





# Directorio donde se encuentran los modelos
model_dir = r"C:\Users\pc\Desktop\HENRY\STREAMLIT PF\models"

def cargar_modelo(nombre_modelo):
  """Carga el vectorizador y el modelo desde un archivo .pkl.

  Args:
    nombre_modelo: El nombre del modelo (sin extensión .pkl).

  Returns:
    Una tupla conteniendo el vectorizador y el modelo.
  """
  vectorizador = pickle.load(open(os.path.join(model_dir, f"{nombre_modelo}_vectorizer.pkl"), 'rb'))
  modelo = pickle.load(open(os.path.join(model_dir, f"{nombre_modelo}_model.pkl"), 'rb'))
  return vectorizador, modelo

# Crear la interfaz de usuario de Streamlit
st.title("Análisis de Sentimientos")
st.write("Ingresa un texto y el modelo te dirá si es positivo, negativo o neutral.")

# Selección del modelo
modelo_seleccionado = st.selectbox("Selecciona un modelo", os.listdir(model_dir))
modelo_seleccionado = modelo_seleccionado.replace("_vectorizer.pkl", "")

# Obtener el texto del usuario
texto = st.text_area("Ingrese el texto aquí:")

# Botón para realizar la predicción
if st.button("Analizar Sentimiento"):
  # Cargar el modelo seleccionado
  vectorizador, modelo = cargar_modelo(modelo_seleccionado)

  # Vectorizar el texto
  texto_vectorizado = vectorizador.transform([texto])

  # Hacer la predicción
  prediccion = modelo.predict(texto_vectorizado)[0]

  # Mostrar el resultado
  st.write(f"El sentimiento del texto es: {prediccion}")
















#ejemplo de como crear botones 
#if st.checkbox("vista de datos head o tail"):
#    if st.button("mostrar head"):
#        st.write(processed_files.head())
#    if st.button("mostrar tail"):
#        st.write(df.tail())





