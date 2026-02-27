import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cargar variables de entorno (API Key de Gemini)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Traductor de Señas", layout="centered")
st.title("Traductor de Lenguaje de Señas 🤟")

if 'fotos' not in st.session_state:
    st.session_state.fotos = []
if 'letras' not in st.session_state:
    st.session_state.letras = []

# 2. Cargar tu modelo local .h5
@st.cache_resource
def cargar_modelo():
    # ¡Asegúrate de que este nombre sea exactamente el de tu archivo!
    return tf.keras.models.load_model("modelo_lenguaje_senas.h5")

modelo = cargar_modelo()

# 3. La Cámara
foto_capturada = st.camera_input("Haz la seña y toma la foto")

# 4. Lógica de Predicción al tomar la foto
if foto_capturada:
    img = Image.open(foto_capturada)
    
    # IMPORTANTE: Cambia '224, 224' por el tamaño que usaste al entrenar
    img_resized = img.resize((224, 224)) 
    img_array = np.array(img_resized) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    
    prediccion_array = modelo.predict(img_array)
    
    # Cambia esto por las letras reales con las que entrenaste tu modelo
    clases = ['A', 'B', 'C', 'D', 'E'] 
    indice_predicho = np.argmax(prediccion_array)
    letra_predicha = clases[indice_predicho]
    
    st.session_state.fotos.append(img)
    st.session_state.letras.append(letra_predicha)

# 5. Galería Visual y Texto Crudo
if st.session_state.fotos:
    st.divider()
    st.write("### Secuencia detectada:")
    
    columnas = st.columns(len(st.session_state.fotos))
    for i, col in enumerate(columnas):
        with col:
            st.image(st.session_state.fotos[i], use_container_width=True)
            st.markdown(f"<h3 style='text-align: center;'>{st.session_state.letras[i]}</h3>", unsafe_allow_html=True)
            
    texto_crudo = "".join(st.session_state.letras)
    st.write(f"**Texto crudo acumulado:** `{texto_crudo}`")

    # 6. Corregir con IA Generativa
    if st.button("✨ Generar Frase Final", type="primary"):
        with st.spinner("Procesando con IA..."):
            try:
                modelo_ia = genai.GenerativeModel('gemini-2.5-flash')
                prompt = f"Tengo esta secuencia de letras de un lenguaje de señas: '{texto_crudo}'. Agrúpala en palabras con sentido, corrige errores y devuelve ÚNICAMENTE la frase final coherente."
                respuesta = modelo_ia.generate_content(prompt)
                
                st.success("### Frase Final:")
                st.info(f"**{respuesta.text.strip()}**")
            except Exception as e:
                st.error("Error al conectar con la IA de texto. Verifica tu API Key.")
    
    if st.button("🗑️ Borrar Todo"):
        st.session_state.fotos = []
        st.session_state.letras = []
        st.rerun()