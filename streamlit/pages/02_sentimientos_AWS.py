import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle
import boto3
import io

# AWS S3 Configuration
S3_BUCKET = 'target-databite'  # Replace with your actual S3 bucket name
S3_MODEL_PREFIX = 'PKL-analisis-sentimiento/'  # Path to models in the S3 bucket

def load_model_from_s3(model_name):
    """
    Load vectorizer and model from AWS S3.
    
    Args:
        model_name (str): Base name of the model (without _vectorizer.pkl or _model.pkl)
    
    Returns:
        Tuple of (vectorizer, model)
    """
    # Create S3 client
    s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAT45W6OH5BCHZYS65',
    aws_secret_access_key='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'
)
    try:
        # Load vectorizer from S3
        vectorizer_response = s3.get_object(
            Bucket=S3_BUCKET, 
            Key=f'{S3_MODEL_PREFIX}{model_name}_vectorizer.pkl'
        )
        vectorizer_bytes = vectorizer_response['Body'].read()
        vectorizer = pickle.load(io.BytesIO(vectorizer_bytes))
        
        # Load model from S3
        model_response = s3.get_object(
            Bucket=S3_BUCKET, 
            Key=f'{S3_MODEL_PREFIX}{model_name}_model.pkl'
        )
        model_bytes = model_response['Body'].read()
        model = pickle.load(io.BytesIO(model_bytes))
        
        return vectorizer, model
    
    except Exception as e:
        st.error(f"Error loading model from S3: {e}")
        return None, None

def list_models_in_s3():
    """
    List available models in the S3 bucket.
    
    Returns:
        List of model names
    """
    s3 = boto3.client('s3')
    
    try:
        # List objects in the models prefix
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODEL_PREFIX)
        
        # Extract unique model names (removing _vectorizer.pkl and _model.pkl)
        models = set()
        for obj in response.get('Contents', []):
            filename = obj['Key'].replace(S3_MODEL_PREFIX, '').replace('_vectorizer.pkl', '').replace('_model.pkl', '')
            models.add(filename)
        
        return sorted(list(models))
    
    except Exception as e:
        st.error(f"Error listing models in S3: {e}")
        return []

# Streamlit App
def main():
    st.title("Análisis de Sentimientos")
    st.write("Ingresa un texto y el modelo te dirá si es positivo, negativo o neutral.")

    # Fetch and display available models from S3
    available_models = list_models_in_s3()
    
    if not available_models:
        st.error("No se encontraron modelos. Verifica tu configuración de S3.")
        return

    # Select model
    modelo_seleccionado = st.selectbox("Selecciona un modelo", available_models)

    # Get user input text
    texto = st.text_area("Ingrese el texto aquí:")

    # Prediction button
    if st.button("Analizar Sentimiento"):
        # Load the selected model from S3
        vectorizador, modelo = load_model_from_s3(modelo_seleccionado)
        
        if vectorizador is None or modelo is None:
            st.error("No se pudo cargar el modelo. Verifica tu configuración.")
            return

        # Vectorize and predict
        texto_vectorizado = vectorizador.transform([texto])
        prediccion = modelo.predict(texto_vectorizado)[0]

        # Display result
        st.write(f"El sentimiento del texto es: {prediccion}")

# Run the Streamlit app
if __name__ == "__main__":
    main()



