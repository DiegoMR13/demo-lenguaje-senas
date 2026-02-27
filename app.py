import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cargar variables de entorno (API Key de Gemini)
load_dotenv()
# Intenta obtener la clave del entorno local (.env) o de los secretos de Streamlit Cloud
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ API Key de Gemini no encontrada. La función de 'Mejorar Texto' podría fallar.")

st.set_page_config(page_title="Traductor de Señas", layout="centered")
st.title("Traductor de Lenguaje de Señas 🤟")

# Inicializar memoria de la sesión
if 'fotos' not in st.session_state:
    st.session_state.fotos = []
if 'letras' not in st.session_state:
    st.session_state.letras = []

# 2. Cargar tu modelo local .h5
@st.cache_resource
def cargar_modelo():
    # Asegúrate de que este nombre sea exactamente el de tu archivo en el repositorio
    return tf.keras.models.load_model("modelo_lenguaje_senas.h5")

try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Verifica que el archivo .h5 esté subido a GitHub.")
    st.stop()

# Mapeo de índices a letras (Lenguaje de señas americano, omitiendo J=9 y Z=25)
mapa_letras = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I',
    10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
    18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'
}

# 3. La Cámara
foto_capturada = st.camera_input("Haz la seña y toma la foto")

# 4. Lógica de Predicción al tomar la foto
if foto_capturada:
    # Abrir y preprocesar la imagen
    img = Image.open(foto_capturada)
    
    # Convertir a escala de grises y redimensionar a 28x28
    img_procesada = img.convert('L').resize((28, 28)) 
    
    # Convertir a matriz, normalizar y darle la forma que espera el modelo
    img_array = np.array(img_procesada) / 255.0 
    img_array = img_array.reshape(1, 28, 28, 1) 
    
    # Predicción matemática
    prediccion_array = modelo.predict(img_array)
    indice_predicho = np.argmax(prediccion_array)
    
    # Obtener la letra usando el diccionario (devuelve "?" si hay un índice inesperado)
    letra_predicha = mapa_letras.get(indice_predicho, "?")
    
    # Guardar en memoria (guardamos la foto a color original para la galería visual)
    st.session_state.fotos.append(img)
    st.session_state.letras.append(letra_predicha)

# 5. Galería Visual y Texto Crudo
if st.session_state.fotos:
    st.divider()
    st.write("### Secuencia detectada:")
    
    # Mostrar fotos en columnas
    columnas = st.columns(len(st.session_state.fotos))
    for i, col in enumerate(columnas):
        with col:
            st.image(st.session_state.fotos[i], use_container_width=True)
            st.markdown(f"<h3 style='text-align: center;'>{st.session_state.letras[i]}</h3>", unsafe_allow_html=True)
            
    texto_crudo = "".join(st.session_state.letras)
    st.write(f"**Texto crudo acumulado:** `{texto_crudo}`")

    # 6. Corregir con IA Generativa
    if st.button("✨ Mejorar Texto", type="primary"):
        with st.spinner("Procesando con IA..."):
            try:
                modelo_ia = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Tengo esta secuencia de letras obtenidas de un modelo de lenguaje de señas: '{texto_crudo}'. Agrúpala en palabras con sentido, si encuentras un '?' eliminalo del texto antes de continuar con las siguientes indicaciones, corrige pequeños errores de predicción y devuelve ÚNICAMENTE la frase final coherente. Si detectas que es una pregunta, ponle los signos adecuados."
                respuesta = modelo_ia.generate_content(prompt)
                
                st.success("### Frase Final:")
                st.info(f"**{respuesta.text.strip()}**")
            except Exception as e:
                st.error("Error al conectar con la IA de texto. Revisa la configuración de tu API Key.")
    
    if st.button("🗑️ Borrar Todo"):
        st.session_state.fotos = []
        st.session_state.letras = []
        st.rerun()

