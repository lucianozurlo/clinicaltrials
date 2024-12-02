import os
import sys
import logging
import json
import hnswlib
import numpy as np
import time
import hashlib
import random
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from tenacity import retry, wait_exponential, stop_after_attempt
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configurar variables de entorno
load_dotenv()

# Modelo de transformer
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# Variables de Streamlit
if "historial" not in st.session_state:
    st.session_state.historial = []  # Lista para almacenar el historial de preguntas y respuestas

if "input_pregunta" not in st.session_state:
    st.session_state.input_pregunta = ""  # Valor inicial vacío para el campo de entrada

# Cargar documentos (similar a tu código previo)
def load_documents(source, is_directory=False):
    loaded_files = []
    if not os.path.exists(source):
        logging.error(f"La fuente '{source}' no existe.")
        raise FileNotFoundError(f"La fuente '{source}' no se encontró.")
    if is_directory:
        for filename in os.listdir(source):
            filepath = os.path.join(source, filename)
            if os.path.isfile(filepath) and filepath.endswith(('.txt', '.json', '.pdf')):
                content = extract_content(filepath)
                if content:
                    loaded_files.append({"filename": filename, "content": content})
    else:
        content = extract_content(source)
        if content:
            loaded_files.append({"filename": os.path.basename(source), "content": content})
    logging.info(f"{len(loaded_files)} documentos cargados.")
    return loaded_files

def extract_content(filepath):
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            units = content.split("\n-----\n")
            return units
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as file:
                return json.load(file)
        elif filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            return ''.join(page.extract_text() or '' for page in reader.pages)
    except Exception as e:
        logging.error(f"Error al extraer contenido de '{filepath}': {e}")
        return None

# Cargar documentos
ruta_fuente = 'data'
documentos = load_documents(ruta_fuente, is_directory=True)

# Crear el índice HNSW para los documentos cargados
def crear_indice_hnsw(documentos, model):
    embeddings = []
    for doc in documentos:
        for fragmento in doc['content']:
            emb = model.encode([fragmento])
            embeddings.append(emb[0])
    embeddings = np.array(embeddings)

    # Crear el índice HNSW
    dim = embeddings.shape[1]  # Dimensión de los embeddings
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=10000, ef_construction=200, M=16)
    index.add_items(embeddings)

    logging.info("Índice HNSW creado y cargado con los embeddings.")
    return index

# Crear el índice
index_archivos = crear_indice_hnsw(documentos, model)

# Configurar Gemini LLM
def configure_gemini():
    global gemini_llm
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("La clave API de Gemini no está configurada.")
        raise EnvironmentError("Configura GEMINI_API_KEY en tu archivo .env.")
    gemini_llm = Gemini(api_key=api_key)
    logging.info("Gemini configurado correctamente.")

configure_gemini()

# Generar embedding de la pregunta
def generate_embedding(texto):
    try:
        embedding = model.encode([texto])
        logging.info(f"Embedding generado para el texto: {texto}")
        return embedding
    except Exception as e:
        logging.error(f"Error al generar el embedding: {e}")
        return np.zeros((1, 384))

# Obtener contexto relevante de los documentos
def obtener_contexto(pregunta, index, top_k=5):
    try:
        pregunta_emb = generate_embedding(pregunta)
        logging.info("Embedding generado para la pregunta.")
        results = index.knn_query(pregunta_emb[0], k=top_k)  # Consulta KNN
        texto = ""
        for idx in results[0]:
            doc = documentos[idx // len(documentos[0]['content'])]  # Obtiene el documento original
            fragmento = doc['content'][idx % len(doc['content'])]
            texto += fragmento + "\n"
        logging.info("Contexto relevante recuperado para la pregunta.")
        return texto
    except Exception as e:
        logging.error(f"Error al obtener el contexto: {e}")
        return ""

# Función para procesar la pregunta y generar la respuesta
def procesar_pregunta():
    if st.session_state.input_pregunta.strip():  # Verificar si el input no está vacío
        # Generar la respuesta
        respuesta = responder_pregunta(st.session_state.input_pregunta)

        # Guardar la pregunta y la respuesta en el historial
        st.session_state.historial.append({
            "pregunta": st.session_state.input_pregunta,
            "respuesta": respuesta
        })

        # Limpiar el campo de entrada
        st.session_state.input_pregunta = ""  # Restablecer a vacío
    else:
        st.warning("Por favor, ingresá una pregunta válida.")

# Lógica para responder la pregunta
def responder_pregunta(pregunta):
    contexto = obtener_contexto(pregunta, index_archivos)
    respuesta = f"Contexto relevante encontrado:\n{contexto}\n\nPregunta procesada: {pregunta}"
    return respuesta

# Categorizar la pregunta (si es necesario)
def categorizar_pregunta(pregunta):
    categorias = {
        "tratamiento": ["tratamiento", "medicación", "cura", "terapia", "fármaco"],
        "ensayo": ["ensayo", "estudio", "prueba", "investigación", "trial"],
        "resultado": ["resultado", "efectividad", "resultados", "éxito", "fracaso"],
        "prevención": ["prevención", "previene", "evitar", "reducción de riesgo"]
    }
    for categoria, palabras in categorias.items():
        if any(palabra in pregunta.lower() for palabra in palabras):
            return categoria
    return "general"

# Configuración Streamlit
if __name__ == "__main__":
    # Configurar el título y la descripción de la aplicación
    st.title("Chatbot sobre Ensayos Clínicos")
    st.write("Hacé preguntas sobre ensayos clínicos y obtené respuestas basadas en el contenido cargado.")

    # Campo de entrada para la pregunta del usuario
    with st.form("my_form"):
        st.text_input(
            "Hacé una pregunta:",
            key="input_pregunta",
            placeholder="Tu pregunta aquí..."
        )
        st.form_submit_button("Enviar", on_click=procesar_pregunta)

    # Mostrar el historial de preguntas y respuestas
    for intercambio in reversed(st.session_state.historial):
        st.write(f"**Pregunta:** {intercambio['pregunta']}")
        with st.spinner('Generando respuesta...'):
            st.write(f"**Respuesta:** {intercambio['respuesta']}")
        st.markdown("---")  # Separador entre interacciones

