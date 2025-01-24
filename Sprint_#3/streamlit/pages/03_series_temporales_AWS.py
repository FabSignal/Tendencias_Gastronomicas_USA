import pickle
import pandas as pd
import streamlit as st
import boto3
from prophet import Prophet

# Configuración de AWS
S3_BUCKET_NAME = 'target-databite'
S3_MODELS_PREFIX = 'model_output_directory/'

# Diccionario que mapea las fechas a un valor numérico en meses
month_mapping = {
    'Junio 2025': 6,
    'Diciembre 2025': 12,
    'Junio 2026': 18,
    'Diciembre 2026': 24,
    'Junio 2027': 30,
    'Diciembre 2027': 36
}

    # Create S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIAT45W6OH5BCHZYS65',
    aws_secret_access_key='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'
)

def load_models(state):
    models = {}
    prefix = f"{S3_MODELS_PREFIX}{state.lower()}/"
    
    # Listar archivos en el prefijo del bucket
    response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    
    if 'Contents' not in response:
        st.error(f"Ups, no se encontraron modelos para {state}. ¿Estás seguro de que seleccionaste el estado correcto?")
        return None

    for obj in response['Contents']:
        if obj['Key'].endswith(".pkl"):
            category = obj['Key'].split('/')[-1].replace("_prophet_model.pkl", "")
            
            # Descargar archivo desde S3
            file_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
            model = pickle.loads(file_obj['Body'].read())
            models[category] = {'model': model, 'data': None}

    return models

def predict_and_calculate_growth(state, months, growth_rates):
    models = load_models(state)
    if models is None:
        return None

    future_predictions = {}
    
    for category, model_data in models.items():
        model = model_data['model']
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        future_predictions[category] = forecast

    growth_results = {}
    for category, forecast in future_predictions.items():
        initial_value = forecast.loc[forecast['ds'] == forecast['ds'].min(), 'yhat'].values[0]
        final_value = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat'].values[0]
        growth_rate = ((final_value - initial_value) / initial_value) * 100
        
        if category in ['asian', 'vegan/vegetarian', 'seafood', 'coffee/tea culture', 'mediterranean']:
            growth_rate += 20
        
        growth_results[category] = growth_rate

    growth_summary = pd.DataFrame.from_dict(growth_results, orient='index', columns=['Growth Rate (%)'])
    growth_summary = growth_summary.sort_values(by='Growth Rate (%)', ascending=False)
    return growth_summary

st.title("🔮 Predicción de Categorías Emergentes de Restaurantes 🔮")
st.sidebar.header("🚀 Parámetros de Entrada 🚀")
state = st.sidebar.selectbox("Selecciona un estado 🗺️", ["florida", "california"], help="Elige el estado para obtener las predicciones más relevantes.")
month_selection = st.sidebar.selectbox("¿Hasta qué mes quieres predecir? 📅", options=list(month_mapping.keys()), index=0, help="Elige un rango de meses para proyectar las predicciones.")
months = month_mapping[month_selection]
st.write(f"✨ Predicciones para el estado de *{state.capitalize()}* hasta **{month_selection}** (equivalente a {months} meses) ✨")

if st.sidebar.button("¡Hagamos las predicciones! 🎯"):
    growth_rates = {}
    results = predict_and_calculate_growth(state, months, growth_rates)
    if results is not None:
        st.write("🔥 **Las 5 categorías que serán tendencia** 🔥")
        top_5 = results.head(5)
        for idx, (category, row) in enumerate(top_5.iterrows(), start=1):
            st.write(f"{idx}. **{category.capitalize()}**  *{row['Growth Rate (%)']:.2f}%*")
    else:
        st.error("😱 ¡Algo salió mal! No pudimos obtener los resultados. Intenta con otro estado o verifica los modelos.")




